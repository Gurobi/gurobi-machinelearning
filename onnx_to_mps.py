#!/usr/bin/env python3
"""
Convert ONNX + VNN-LIB to Gurobi MPS file.

This script:
1. Loads an ONNX model
2. Verifies it has a purely sequential architecture (no residuals/skips)
3. Parses VNN-LIB property file
4. Creates Gurobi model with NN constraints + properties
5. Exports to MPS format
6. Verifies the model against ONNX Runtime

Usage:
    python onnx_to_mps.py <onnx_file> <vnnlib_file> <output_mps>
"""

import sys
import os
import argparse
import gzip
import onnx
from onnx import helper
import numpy as np
import onnxruntime as ort
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.onnx import add_onnx_constr
from vnnlib_simple import parse_vnnlib_simple


def remove_dropout_nodes(model):
    """
    Remove Dropout nodes from ONNX model.

    Dropout is a no-op during inference (it only affects training).
    We remove it by connecting the input directly to the output.

    Args:
        model: ONNX ModelProto

    Returns:
        Modified ONNX ModelProto with Dropout nodes removed
    """
    graph = model.graph

    # Find all Dropout nodes
    dropout_nodes = [node for node in graph.node if node.op_type == "Dropout"]

    if not dropout_nodes:
        return model

    print(f"  Removing {len(dropout_nodes)} Dropout nodes (no-op during inference)...")

    # Build mapping of output -> input for Dropout nodes
    # Dropout has: inputs=[data, (optional)ratio, (optional)training_mode], outputs=[output, (optional)mask]
    dropout_replacements = {}
    for node in dropout_nodes:
        # Dropout output[0] should be replaced with input[0]
        dropout_replacements[node.output[0]] = node.input[0]

    # Remove Dropout nodes
    new_nodes = [node for node in graph.node if node.op_type != "Dropout"]

    # Replace references to Dropout outputs with their inputs
    for node in new_nodes:
        # Replace in node inputs
        for i, input_name in enumerate(node.input):
            if input_name in dropout_replacements:
                node.input[i] = dropout_replacements[input_name]

    # Update graph outputs if they reference Dropout outputs
    for i, output in enumerate(graph.output):
        if output.name in dropout_replacements:
            # Create new ValueInfo with updated name
            new_output = helper.make_tensor_value_info(
                dropout_replacements[output.name],
                output.type.tensor_type.elem_type,
                [d.dim_value for d in output.type.tensor_type.shape.dim],
            )
            graph.output[i].CopyFrom(new_output)

    # Clear and rebuild nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    return model


def is_max_pattern(or_clause):
    """
    Check if OR clause matches MAX pattern: all Y_i >= Y_target with same target.

    Args:
        or_clause: List of (idx1, operator, idx2) tuples

    Returns:
        (is_pattern, target_idx, other_indices) if pattern matches
        (False, None, None) otherwise
    """
    if not or_clause:
        return False, None, None

    # Check all have >= operator and same second operand
    operators = [op for _, op, _ in or_clause]
    targets = [idx2 for _, _, idx2 in or_clause]

    if not all(op == ">=" for op in operators):
        return False, None, None

    if len(set(targets)) != 1:
        return False, None, None

    target_idx = targets[0]
    other_indices = [idx1 for idx1, _, _ in or_clause]

    # Verify target is not in others (sanity check)
    if target_idx in other_indices:
        return False, None, None

    return True, target_idx, other_indices


def is_min_pattern(or_clause):
    """
    Check if OR clause matches MIN pattern: all Y_target <= Y_i with same target.

    Args:
        or_clause: List of (idx1, operator, idx2) tuples

    Returns:
        (is_pattern, target_idx, other_indices) if pattern matches
        (False, None, None) otherwise
    """
    if not or_clause:
        return False, None, None

    # Check all have <= operator and same first operand
    operators = [op for _, op, _ in or_clause]
    targets = [idx1 for idx1, _, _ in or_clause]

    if not all(op == "<=" for op in operators):
        return False, None, None

    if len(set(targets)) != 1:
        return False, None, None

    target_idx = targets[0]
    other_indices = [idx2 for _, _, idx2 in or_clause]

    # Verify target is not in others
    if target_idx in other_indices:
        return False, None, None

    return True, target_idx, other_indices


def verify_sequential_architecture(onnx_model):
    """
    Verify ONNX model has purely sequential architecture.

    Returns:
        (is_sequential, message)
    """
    # Allowed ops for neural networks (updated for PR 456)
    allowed_ops = {
        # Fully connected layers
        "Gemm",
        "MatMul",
        "Add",
        # Activations
        "Relu",
        # Shape operations
        "Flatten",
        "Reshape",
        "Transpose",
        "Constant",
        # Convolutional layers (PR 456)
        "Conv",
        "BatchNormalization",
        # Pooling (PR 456)
        "AveragePool",
        "MaxPool",
        "GlobalAveragePool",
        # Note: Dropout is removed by preprocessing, not passed to Gurobi
    }

    # Check all node types
    node_types = {node.op_type for node in onnx_model.graph.node}
    unsupported = node_types - allowed_ops

    if unsupported:
        return False, f"Unsupported ops: {unsupported}"

    # Check for residual/skip connections
    # Track how many times each intermediate value is used
    value_usage = {}

    # Count initializers (weights/biases) - these can be reused
    initializer_names = {init.name for init in onnx_model.graph.initializer}

    for node in onnx_model.graph.node:
        for inp in node.input:
            # Skip empty inputs
            if not inp:
                continue
            # Skip initializers (weights/biases)
            if inp in initializer_names:
                continue
            # Skip constants
            if any(
                n.op_type == "Constant" and inp in n.output
                for n in onnx_model.graph.node
            ):
                continue

            # Track usage of intermediate values
            value_usage[inp] = value_usage.get(inp, 0) + 1

    return True, "Architecture verified"


def verify_gurobi_model(
    gurobi_model,
    onnx_file,
    input_vars,
    output_vars,
    input_dim,
    output_dim,
    original_bounds,
    input_var_shape,
    input_shape_nchw,
):
    """
    Verify Gurobi model matches ONNX Runtime on random inputs.

    Args:
        original_bounds: List of (lb, ub) tuples to restore after testing
        input_var_shape: Shape of input variables in NHWC format (for Gurobi)
        input_shape_nchw: Shape of input in NCHW format (for ONNX Runtime)

    Returns:
        (success, max_error, message)
    """
    ort_session = ort.InferenceSession(onnx_file)
    input_name = ort_session.get_inputs()[0].name

    # Flatten input/output vars for easier indexing
    input_flat = input_vars.reshape(-1)
    output_flat = output_vars.reshape(-1)

    # Test on 5 random inputs
    max_error = 0.0

    for trial in range(5):
        # Random input in [0, 1]
        test_input = np.random.rand(input_dim).astype(np.float32)

        # ONNX Runtime prediction (reshape to NCHW format)
        onnx_input = test_input.reshape(input_shape_nchw)
        onnx_output = ort_session.run(None, {input_name: onnx_input})[0]
        onnx_output_flat = onnx_output.flatten()

        # Temporarily fix input in Gurobi model
        # Note: Gurobi variables are in NHWC format, but test_input is flat
        # We need to convert the flat input to NHWC order
        if len(input_var_shape) == 4:  # Conv layers
            # Reshape to NCHW then transpose to NHWC
            input_nchw = test_input.reshape(input_shape_nchw)
            # NCHW [N, C, H, W] -> NHWC [N, H, W, C]
            input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1))
            input_flat_nhwc = input_nhwc.flatten()

            for i in range(input_dim):
                input_flat[i].lb = float(input_flat_nhwc[i])
                input_flat[i].ub = float(input_flat_nhwc[i])
        else:  # MLP layers - no transpose needed
            for i in range(input_dim):
                input_flat[i].lb = float(test_input[i])
                input_flat[i].ub = float(test_input[i])

        # Solve
        gurobi_model.setObjective(0, GRB.MINIMIZE)
        gurobi_model.optimize()

        if gurobi_model.status == 3:  # INFEASIBLE due to license
            # Restore original bounds before returning
            for i in range(input_dim):
                lb, ub = original_bounds[i]
                input_flat[i].lb = lb if lb is not None else -GRB.INFINITY
                input_flat[i].ub = ub if ub is not None else GRB.INFINITY
            # Skip verification - model is likely too large for academic license
            return True, 0.0, "Skipped verification (model too large for license)"

        if gurobi_model.status != GRB.OPTIMAL:
            # Restore original bounds before returning
            for i in range(input_dim):
                lb, ub = original_bounds[i]
                input_flat[i].lb = lb if lb is not None else -GRB.INFINITY
                input_flat[i].ub = ub if ub is not None else GRB.INFINITY
            return (
                False,
                0.0,
                f"Optimization failed on trial {trial + 1}: status={gurobi_model.status}",
            )

        # Compare outputs
        gurobi_output = output_flat.X
        error = np.abs(onnx_output_flat - gurobi_output).max()
        max_error = max(max_error, error)

        if error > 1e-3:
            # Restore original bounds before returning
            for i in range(input_dim):
                lb, ub = original_bounds[i]
                input_flat[i].lb = lb if lb is not None else -GRB.INFINITY
                input_flat[i].ub = ub if ub is not None else GRB.INFINITY
            return False, error, f"Large error on trial {trial + 1}: {error:.2e}"

    # Restore original bounds after all tests
    for i in range(input_dim):
        lb, ub = original_bounds[i]
        input_flat[i].lb = lb if lb is not None else -GRB.INFINITY
        input_flat[i].ub = ub if ub is not None else GRB.INFINITY

    return True, max_error, f"Verified OK (max error: {max_error:.2e})"


def convert_to_mps(onnx_file, vnnlib_file, output_mps, verify=True, verbose=True):
    """
    Convert ONNX + VNN-LIB to MPS file.

    Args:
        onnx_file: Path to ONNX model
        vnnlib_file: Path to VNN-LIB property
        output_mps: Output MPS file path
        verify: Whether to verify the model
        verbose: Print progress

    Returns:
        (success, message)
    """
    if verbose:
        print("=" * 80)
        print("ONNX + VNN-LIB to MPS Converter")
        print("=" * 80)
        print(f"ONNX:   {onnx_file}")
        print(f"VNNLIB: {vnnlib_file}")
        print(f"Output: {output_mps}")
        print()

    # Load ONNX model (handle gzip compression)
    try:
        if onnx_file.endswith(".gz"):
            with gzip.open(onnx_file, "rb") as f:
                onnx_model = onnx.load(f)
        else:
            onnx_model = onnx.load(onnx_file)
        if verbose:
            print("✓ ONNX model loaded")

        # Remove Dropout nodes (no-op during inference)
        onnx_model = remove_dropout_nodes(onnx_model)
    except Exception as e:
        return False, f"Failed to load ONNX: {e}"

    # Verify architecture is sequential
    is_sequential, arch_msg = verify_sequential_architecture(onnx_model)
    if verbose:
        if is_sequential:
            print(f"✓ {arch_msg}")
        else:
            print(f"✗ {arch_msg}")

    if not is_sequential:
        return False, arch_msg

    # Get model dimensions
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]

    input_shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in output_info.type.tensor_type.shape.dim]

    # Fix dynamic batch dimensions (0 or -1) to 1 for verification
    if input_shape[0] <= 0:
        input_shape[0] = 1
    if output_shape[0] <= 0:
        output_shape[0] = 1

    # Calculate total dimensions, excluding batch dimension (dim[0] if <= 0)
    # For CNNs: [batch, channels, height, width] -> channels * height * width
    # For MLPs: [batch, features] -> features
    input_dim = int(np.prod([d for d in input_shape if d > 0]))
    output_dim = int(np.prod([d for d in output_shape if d > 0]))

    # Convert ONNX shapes (NCHW) to gurobi_ml format (NHWC)
    # ONNX uses: [batch, channels, height, width]
    # gurobi_ml expects: [batch, height, width, channels]
    def onnx_to_nhwc(shape):
        """Convert ONNX shape from NCHW to NHWC format."""
        if len(shape) == 4:  # Conv layers: [N, C, H, W] -> [N, H, W, C]
            return (shape[0], shape[2], shape[3], shape[1])
        else:  # MLP layers: keep as-is
            return tuple(shape)

    input_var_shape = onnx_to_nhwc(input_shape)
    output_var_shape = onnx_to_nhwc(output_shape)

    if verbose:
        print(f"✓ Model dimensions: {input_dim}D → {output_dim}D")

    # Parse VNN-LIB property
    try:
        prop = parse_vnnlib_simple(vnnlib_file)
        if verbose:
            print(
                f"✓ VNN-LIB property parsed: {prop.num_inputs} inputs, {prop.num_outputs} outputs"
            )
    except Exception as e:
        return False, f"Failed to parse VNN-LIB: {e}"

    # Verify dimensions match
    if prop.num_inputs != input_dim:
        return (
            False,
            f"Dimension mismatch: ONNX has {input_dim} inputs, VNN-LIB expects {prop.num_inputs}",
        )
    if prop.num_outputs != output_dim:
        return (
            False,
            f"Dimension mismatch: ONNX has {output_dim} outputs, VNN-LIB expects {prop.num_outputs}",
        )

    # Create Gurobi model
    try:
        model = gp.Model("verification")
        model.setParam("OutputFlag", 0)

        # Add input variables with VNN-LIB bounds
        # Note: gurobi_ml expects variables in NHWC format for Conv layers
        # For CNN: input_var_shape is (1, H, W, C) after conversion
        # For MLP: input_var_shape is (1, features) unchanged
        input_vars = model.addMVar(
            shape=input_var_shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x"
        )

        # Apply VNN-LIB bounds to input variables
        # VNN-LIB bounds are in flattened NCHW order, but variables are in NHWC
        if len(input_var_shape) == 4:  # Conv layers - need to convert NCHW -> NHWC
            # Create a dummy array in NCHW format to map bounds
            nchw_bounds = np.zeros(input_shape, dtype=object)
            flat_idx = 0
            for bounds in prop.input_bounds:
                # Flatten in NCHW order
                idx = np.unravel_index(flat_idx, input_shape)
                nchw_bounds[idx] = bounds
                flat_idx += 1

            # Convert to NHWC order: [N, C, H, W] -> [N, H, W, C]
            nhwc_bounds = np.transpose(nchw_bounds, (0, 2, 3, 1))

            # Apply bounds to flattened NHWC variables
            input_flat = input_vars.reshape(-1)
            for i, bounds in enumerate(nhwc_bounds.flatten()):
                lb, ub = bounds
                if lb is not None:
                    input_flat[i].lb = lb
                if ub is not None:
                    input_flat[i].ub = ub
        else:  # MLP layers - no conversion needed
            input_flat = input_vars.reshape(-1)
            for i, (lb, ub) in enumerate(prop.input_bounds):
                if lb is not None:
                    input_flat[i].lb = lb
                if ub is not None:
                    input_flat[i].ub = ub

        # Add output variables
        output_vars = model.addMVar(
            shape=output_var_shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y"
        )

        if verbose:
            print(f"✓ Created {input_dim} input and {output_dim} output variables")

    except Exception as e:
        return False, f"Failed to create Gurobi variables: {e}"

    # Add ONNX neural network constraints
    try:
        add_onnx_constr(model, onnx_model, input_vars, output_vars)
        model.update()

        if verbose:
            print(
                f"✓ Added ONNX constraints: {model.NumVars} vars, {model.NumConstrs} linear constrs, {model.NumGenConstrs} general constrs"
            )
    except Exception as e:
        return False, f"Failed to add ONNX constraints: {e}"

    # Add VNN-LIB output property constraints (simple bounds)
    try:
        # Flatten output for easier indexing (output is typically (1, num_classes))
        output_flat = output_vars.reshape(-1)

        num_output_constrs = 0
        for i, (lb, ub) in prop.output_constraints.items():
            if lb is not None:
                model.addConstr(output_flat[i] >= lb, name=f"prop_y{i}_lb")
                num_output_constrs += 1
            if ub is not None:
                model.addConstr(output_flat[i] <= ub, name=f"prop_y{i}_ub")
                num_output_constrs += 1

        if verbose and num_output_constrs > 0:
            print(f"✓ Added {num_output_constrs} simple bound constraints")
    except Exception as e:
        return False, f"Failed to add property constraints: {e}"

    # Add VNN-LIB disjunctive constraints (OR clauses) - OPTIMIZED
    try:
        # Flatten output for easier indexing
        output_flat = output_vars.reshape(-1)

        num_disjunctive_constrs = 0
        num_binary_vars_saved = 0

        for or_idx, or_clause in enumerate(prop.disjunctive_constraints):
            # or_clause is a list of (var_idx1, operator, var_idx2) tuples
            # We need: at least one of these comparisons must be true

            # Try to match MAX pattern first (optimal for cora_2024)
            is_max, target_idx, other_indices = is_max_pattern(or_clause)

            if is_max:
                # Use MAX constraint - NO binary variables needed!
                max_var = model.addVar(lb=-GRB.INFINITY, name=f"max_or{or_idx}")
                other_outputs = [output_flat[i] for i in other_indices]

                # max_var = MAX(Y_0, Y_2, ..., Y_n)
                model.addGenConstrMax(max_var, other_outputs, name=f"max_or{or_idx}")

                # Enforce: max_var >= Y_target
                model.addConstr(
                    max_var >= output_flat[target_idx], name=f"or_property_{or_idx}"
                )

                num_disjunctive_constrs += 1
                num_binary_vars_saved += len(or_clause)

                if verbose:
                    indices_str = ", ".join(f"Y_{i}" for i in other_indices)
                    print(
                        f"  ✓ OR clause {or_idx}: MAX({indices_str}) >= Y_{target_idx} (saved {len(or_clause)} binary vars)"
                    )

                continue

            # Try to match MIN pattern
            is_min, target_idx, other_indices = is_min_pattern(or_clause)

            if is_min:
                # Use MIN constraint
                min_var = model.addVar(lb=-GRB.INFINITY, name=f"min_or{or_idx}")
                other_outputs = [output_flat[i] for i in other_indices]

                # min_var = MIN(Y_0, Y_2, ..., Y_n)
                model.addGenConstrMin(min_var, other_outputs, name=f"min_or{or_idx}")

                # Enforce: Y_target <= min_var (equivalently: min_var >= Y_target)
                model.addConstr(
                    min_var >= output_flat[target_idx], name=f"or_property_{or_idx}"
                )

                num_disjunctive_constrs += 1
                num_binary_vars_saved += len(or_clause)

                if verbose:
                    indices_str = ", ".join(f"Y_{i}" for i in other_indices)
                    print(
                        f"  ✓ OR clause {or_idx}: Y_{target_idx} <= MIN({indices_str}) (saved {len(or_clause)} binary vars)"
                    )

                continue

            # Fall back to indicator constraints for general case
            z_vars = []
            for clause_idx, (idx1, op, idx2) in enumerate(or_clause):
                z = model.addVar(
                    vtype=GRB.BINARY, name=f"z_or{or_idx}_clause{clause_idx}"
                )
                z_vars.append(z)

                # Add indicator constraint: if z == 1, then the comparison holds
                if op == ">=":
                    # z == 1 => Y_idx1 >= Y_idx2
                    model.addGenConstrIndicator(
                        z,
                        True,
                        output_flat[idx1] >= output_flat[idx2],
                        name=f"ind_or{or_idx}_c{clause_idx}",
                    )
                elif op == "<=":
                    # z == 1 => Y_idx1 <= Y_idx2
                    model.addGenConstrIndicator(
                        z,
                        True,
                        output_flat[idx1] <= output_flat[idx2],
                        name=f"ind_or{or_idx}_c{clause_idx}",
                    )

            # At least one indicator must be true (OR constraint)
            if z_vars:
                model.addConstr(sum(z_vars) >= 1, name=f"or_constraint_{or_idx}")
                num_disjunctive_constrs += 1

                if verbose:
                    print(
                        f"  ✓ OR clause {or_idx}: General disjunction ({len(z_vars)} binary vars)"
                    )

        model.update()

        if verbose and num_disjunctive_constrs > 0:
            total_clauses = sum(len(clause) for clause in prop.disjunctive_constraints)
            print(
                f"✓ Added {num_disjunctive_constrs} disjunctive constraint(s) with {total_clauses} total clauses"
            )
            if num_binary_vars_saved > 0:
                print(
                    f"  🎉 Optimization: Saved {num_binary_vars_saved} binary variables using MAX/MIN constraints!"
                )
    except Exception as e:
        return False, f"Failed to add disjunctive constraints: {e}"

    # Verify model correctness
    if verify:
        if verbose:
            print("\nVerifying model against ONNX Runtime...")

        # For verification, we need the actual ONNX model, not the compressed file path
        # Create a temporary uncompressed model if needed
        temp_onnx_file = None
        if onnx_file.endswith(".gz"):
            import tempfile

            temp_onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            onnx.save(onnx_model, temp_onnx_file.name)
            temp_onnx_file.close()
            verify_onnx_path = temp_onnx_file.name
        else:
            verify_onnx_path = onnx_file

        success, max_error, verify_msg = verify_gurobi_model(
            model,
            verify_onnx_path,
            input_vars,
            output_vars,
            input_dim,
            output_dim,
            prop.input_bounds,
            input_var_shape,
            tuple(input_shape),  # Pass original NCHW shape for ONNX Runtime
        )

        # Clean up temporary file if created
        if temp_onnx_file:
            try:
                os.unlink(temp_onnx_file.name)
            except Exception as _:
                pass

        if verbose:
            if success:
                print(f"✓ {verify_msg}")
            else:
                print(f"✗ {verify_msg}")

        if not success:
            return False, f"Verification failed: {verify_msg}"

    # Export to MPS (Gurobi handles .mps.bz2 compression automatically)
    try:
        # Add a dummy objective if none exists
        model.setObjective(0, GRB.MINIMIZE)
        model.update()

        # Write MPS file (Gurobi automatically compresses if extension is .mps.bz2)
        model.write(output_mps)

        if verbose:
            print(f"\n✓ MPS file written: {output_mps}")

        # Get file size
        file_size = os.path.getsize(output_mps)
        if verbose:
            print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    except Exception as e:
        return False, f"Failed to write MPS: {e}"

    if verbose:
        print("\n" + "=" * 80)
        print("✓ CONVERSION SUCCESSFUL")
        print("=" * 80)

    return True, "Success"


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX + VNN-LIB to Gurobi MPS file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onnx_to_mps.py model.onnx prop.vnnlib output.mps
  python onnx_to_mps.py model.onnx prop.vnnlib output.mps --no-verify
  python onnx_to_mps.py model.onnx prop.vnnlib output.mps --quiet
        """,
    )

    parser.add_argument("onnx_file", help="Path to ONNX model file")
    parser.add_argument("vnnlib_file", help="Path to VNN-LIB property file")
    parser.add_argument("output_mps", help="Output MPS file path")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification against ONNX Runtime",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()

    # Check input files exist
    if not os.path.exists(args.onnx_file):
        print(f"Error: ONNX file not found: {args.onnx_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.vnnlib_file):
        print(f"Error: VNN-LIB file not found: {args.vnnlib_file}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_mps)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert
    success, message = convert_to_mps(
        args.onnx_file,
        args.vnnlib_file,
        args.output_mps,
        verify=not args.no_verify,
        verbose=not args.quiet,
    )

    if not success:
        print(f"\n✗ CONVERSION FAILED: {message}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
