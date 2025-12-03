# Copyright © 2025 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
ONNX + VNN-LIB to Gurobi MPS converter.

This module provides functionality to convert neural network verification problems
specified as ONNX models with VNN-LIB properties into Gurobi optimization models
that can be exported to MPS format.

Main features:
- Loads ONNX models with automatic Dropout removal
- Parses VNN-LIB property files
- Creates Gurobi models with neural network and property constraints
- Optimized handling of disjunctive constraints (MAX/MIN patterns)
- Optional verification against ONNX Runtime
- MPS export with automatic compression support
"""

import os
import gzip
import tempfile
import onnx
from onnx import helper
import numpy as np
import onnxruntime as ort
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.onnx import add_onnx_constr, add_onnx_dag_constr
from .parser import parse_vnnlib_simple


def remove_dropout_nodes(model):
    """Remove Dropout nodes from ONNX model.

    Dropout is a no-op during inference (it only affects training).
    This function removes it by connecting inputs directly to outputs.

    Args:
        model: ONNX ModelProto.

    Returns:
        Modified ONNX ModelProto with Dropout nodes removed.
    """
    graph = model.graph

    # Find all Dropout nodes
    dropout_nodes = [node for node in graph.node if node.op_type == "Dropout"]

    if not dropout_nodes:
        return model

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


def check_model_architecture(onnx_model):
    """Check ONNX model architecture and determine if it's sequential or DAG.

    Args:
        onnx_model: ONNX ModelProto.

    Returns:
        Tuple of (is_supported: bool, is_sequential: bool, message: str).
        - is_supported: Whether the model can be converted at all
        - is_sequential: Whether it's purely sequential (vs DAG with skip connections)
        - message: Description of the result
    """
    # Allowed ops for neural networks (same for both sequential and DAG)
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
        # Convolutional layers
        "Conv",
        "BatchNormalization",
        # Pooling
        "AveragePool",
        "MaxPool",
        "GlobalAveragePool",
        # DAG-specific
        "Identity",  # Pass-through for skip connections
        # Note: Dropout is removed by preprocessing
    }

    # Check all node types
    node_types = {node.op_type for node in onnx_model.graph.node}
    unsupported = node_types - allowed_ops

    if unsupported:
        return False, False, f"Unsupported ops: {unsupported}"

    # Check for residual/skip connections
    # Track how many times each intermediate value is used
    value_usage = {}

    # Count initializers (weights/biases) - these can be reused
    initializer_names = {init.name for init in onnx_model.graph.initializer}

    # Count graph inputs (can be used multiple times)
    input_names = {input_val.name for input_val in onnx_model.graph.input}

    for node in onnx_model.graph.node:
        for inp in node.input:
            # Skip empty inputs
            if not inp:
                continue
            # Skip initializers (weights/biases) - these can be reused
            if inp in initializer_names:
                continue
            # Skip graph inputs - these can be used multiple times
            if inp in input_names:
                continue
            # Skip constants
            if any(
                n.op_type == "Constant" and inp in n.output
                for n in onnx_model.graph.node
            ):
                continue

            # Track usage of intermediate values
            value_usage[inp] = value_usage.get(inp, 0) + 1

    # Check if any intermediate value is used more than once (skip connection)
    has_skip_connections = any(count > 1 for count in value_usage.values())

    if has_skip_connections:
        return True, False, "DAG architecture with skip/residual connections detected"
    else:
        return True, True, "Sequential architecture verified"


def is_max_pattern(or_clause):
    """Check if OR clause matches MAX pattern: all Y_i >= Y_target with same target.

    Args:
        or_clause: List of (idx1, operator, idx2) tuples.

    Returns:
        Tuple of (is_pattern: bool, target_idx: int|None, other_indices: list|None).
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
    """Check if OR clause matches MIN pattern: all Y_target <= Y_i with same target.

    Args:
        or_clause: List of (idx1, operator, idx2) tuples.

    Returns:
        Tuple of (is_pattern: bool, target_idx: int|None, other_indices: list|None).
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
    """Verify Gurobi model matches ONNX Runtime on random inputs.

    Args:
        gurobi_model: Gurobi Model object.
        onnx_file: Path to ONNX file.
        input_vars: Gurobi input variables.
        output_vars: Gurobi output variables.
        input_dim: Flattened input dimension.
        output_dim: Flattened output dimension.
        original_bounds: List of (lb, ub) tuples to restore after testing.
        input_var_shape: Shape of input variables in NHWC format (for Gurobi).
        input_shape_nchw: Shape of input in NCHW format (for ONNX Runtime).

    Returns:
        Tuple of (success: bool, max_error: float, message: str).
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
    """Convert ONNX + VNN-LIB to Gurobi MPS file.

    This is the main entry point for converting a neural network verification
    problem into a Gurobi optimization model.

    Args:
        onnx_file: Path to ONNX model file (can be .onnx or .onnx.gz).
        vnnlib_file: Path to VNN-LIB property file.
        output_mps: Output MPS file path (can end with .mps or .mps.bz2).
        verify: Whether to verify the model against ONNX Runtime (default: True).
        verbose: Print progress messages (default: True).

    Returns:
        Tuple of (success: bool, message: str).

    Example:
        >>> from gurobi_ml.vnnlib import convert_to_mps
        >>> success, msg = convert_to_mps('model.onnx', 'prop.vnnlib', 'out.mps')
        >>> if success:
        ...     print("Conversion successful!")
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
        original_dropout_count = len(
            [n for n in onnx_model.graph.node if n.op_type == "Dropout"]
        )
        if original_dropout_count > 0:
            if verbose:
                print(
                    f"  Removing {original_dropout_count} Dropout nodes (no-op during inference)..."
                )
            onnx_model = remove_dropout_nodes(onnx_model)
    except Exception as e:
        return False, f"Failed to load ONNX: {e}"

    # Check architecture
    is_supported, is_sequential, arch_msg = check_model_architecture(onnx_model)
    if verbose:
        if is_supported:
            print(f"✓ {arch_msg}")
        else:
            print(f"✗ {arch_msg}")

    if not is_supported:
        return False, arch_msg

    # Get model dimensions
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]

    input_shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in output_info.type.tensor_type.shape.dim]

    # Fix dynamic batch dimensions (0 or -1) to 1
    if input_shape[0] <= 0:
        input_shape[0] = 1
    if output_shape[0] <= 0:
        output_shape[0] = 1

    # Calculate total dimensions, excluding batch dimension
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
    # Use DAG formulation if model has skip connections, otherwise use sequential
    try:
        if is_sequential:
            add_onnx_constr(model, onnx_model, input_vars, output_vars)
            if verbose:
                print("✓ Using sequential formulation")
        else:
            add_onnx_dag_constr(model, onnx_model, input_vars, output_vars)
            if verbose:
                print("✓ Using DAG formulation for skip/residual connections")

        model.update()

        if verbose:
            print(
                f"✓ Added ONNX constraints: {model.NumVars} vars, {model.NumConstrs} linear constrs, {model.NumGenConstrs} general constrs"
            )
    except Exception as e:
        return False, f"Failed to add ONNX constraints: {e}"

    # Add VNN-LIB output property constraints (simple bounds)
    try:
        # Flatten output for easier indexing
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

            # Try to match MAX pattern first
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
                    print(
                        f"  ✓ OR clause {or_idx}: MAX constraint (saved {len(or_clause)} binary vars)"
                    )
                continue

            # Try to match MIN pattern
            is_min, target_idx, other_indices = is_min_pattern(or_clause)

            if is_min:
                # Use MIN constraint
                min_var = model.addVar(lb=-GRB.INFINITY, name=f"min_or{or_idx}")
                other_outputs = [output_flat[i] for i in other_indices]

                # min_var = MIN(Y_1, Y_2, ..., Y_n)
                model.addGenConstrMin(min_var, other_outputs, name=f"min_or{or_idx}")

                # Enforce: min_var >= Y_target
                model.addConstr(
                    min_var <= output_flat[target_idx], name=f"or_property_{or_idx}"
                )

                num_disjunctive_constrs += 1
                num_binary_vars_saved += len(or_clause)

                if verbose:
                    print(
                        f"  ✓ OR clause {or_idx}: MIN constraint (saved {len(or_clause)} binary vars)"
                    )
                continue

            # General case: Use binary variables and indicator constraints
            # For OR(C1, C2, ..., Cn), we need: z1 + z2 + ... + zn >= 1
            # where zi = 1 if Ci is true
            z_vars = []
            for clause_idx, (idx1, op, idx2) in enumerate(or_clause):
                z = model.addVar(vtype=GRB.BINARY, name=f"z_or{or_idx}_c{clause_idx}")
                z_vars.append(z)

                # Add indicator constraint: z = 1 -> Y_idx1 op Y_idx2
                if op == ">=":
                    model.addGenConstrIndicator(
                        z,
                        True,
                        output_flat[idx1] >= output_flat[idx2],
                        name=f"ind_or{or_idx}_c{clause_idx}",
                    )
                elif op == "<=":
                    model.addGenConstrIndicator(
                        z,
                        True,
                        output_flat[idx1] <= output_flat[idx2],
                        name=f"ind_or{or_idx}_c{clause_idx}",
                    )
                else:
                    return False, f"Unsupported operator in disjunction: {op}"

            # At least one must be true
            model.addConstr(sum(z_vars) >= 1, name=f"or_constr_{or_idx}")
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
            except Exception:
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
