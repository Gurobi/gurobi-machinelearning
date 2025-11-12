"""
VNN-LIB Parser for CerSyVe Benchmark Properties

This module parses VNN-LIB files from the CerSyVe benchmark and extracts
input bounds and output property constraints for neural network verification.
"""

import re
from dataclasses import dataclass


@dataclass
class VNNLIBProperty:
    """Represents a VNN-LIB property specification."""

    # Input bounds
    input_bounds: list[
        tuple[float, float]
    ]  # [(lower, upper), ...] for each input dimension

    # Output property constraints
    # For CerSyVe, typically: Y_0 <= 0 and Y_1 >= 0
    output_constraints: dict[
        int, tuple[float | None, float | None]
    ]  # {index: (lower, upper)}

    # Disjunctive output constraints (OR clauses)
    # List of disjunctions, each containing a list of (var_idx1, op, var_idx2) tuples
    # Example: [(0, '>=', 1), (2, '>=', 1), ...] means (Y_0 >= Y_1) OR (Y_2 >= Y_1) OR ...
    disjunctive_constraints: list[list[tuple[int, str, int]]]  # List of OR clauses

    # Metadata
    property_name: str
    num_inputs: int
    num_outputs: int

    def __repr__(self):
        return (
            f"VNNLIBProperty(name='{self.property_name}', "
            f"inputs={self.num_inputs}, outputs={self.num_outputs})"
        )


class VNNLIBParser:
    """Parser for VNN-LIB property files."""

    # Regular expressions for parsing
    COMMENT_PATTERN = re.compile(r"^\s*;")
    DECLARE_PATTERN = re.compile(r"\(declare-const\s+([XY])_(\d+)\s+Real\)")
    ASSERT_PATTERN = re.compile(
        r"\(assert\s+\((<=|>=)\s+([XY])_(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\)"
    )

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset parser state."""
        self.input_vars = set()
        self.output_vars = set()
        self.input_bounds = {}  # {index: [lower, upper]}
        self.output_bounds = {}  # {index: [lower, upper]}
        self.disjunctive_constraints = []  # List of OR clauses
        self.property_name = "unknown"

    def parse_disjunctive_constraint(self, content: str) -> list[tuple[int, str, int]]:
        """Parse a disjunctive constraint of the form (assert (or (and (>= Y_i Y_j)) ...))

        Args:
            content: Full file content as string

        Returns:
            List of (var_idx1, operator, var_idx2) tuples
        """
        # Find all (assert (or blocks by matching parentheses
        i = 0
        while i < len(content):
            # Look for "(assert (or"
            if content[i : i + 11] == "(assert (or":
                # Found start, now find matching closing parens
                start_idx = i
                paren_count = 0
                j = i
                while j < len(content):
                    if content[j] == "(":
                        paren_count += 1
                    elif content[j] == ")":
                        paren_count -= 1
                        if paren_count == 0:
                            # Found end of (assert block
                            or_block = content[start_idx : j + 1]

                            # Parse individual clauses within this OR block
                            clause_pattern = (
                                r"\(and\s+\((>=|<=)\s+Y_(\d+)\s+Y_(\d+)\)\)"
                            )
                            clauses = re.findall(clause_pattern, or_block)

                            if clauses:
                                # Convert to (idx1, op, idx2) tuples
                                parsed_clauses = [
                                    (int(idx1), op, int(idx2))
                                    for op, idx1, idx2 in clauses
                                ]
                                self.disjunctive_constraints.append(parsed_clauses)

                            i = j
                            break
                    j += 1
            i += 1

        return self.disjunctive_constraints

    def parse_file(self, filepath: str) -> VNNLIBProperty:
        """Parse a VNN-LIB file and return a property specification.

        Args:
            filepath: Path to the .vnnlib file

        Returns:
            VNNLIBProperty object with parsed constraints
        """
        self.reset()

        with open(filepath) as f:
            content = f.read()

        lines = content.split("\n")

        # Extract property name from first comment line
        if lines and self.COMMENT_PATTERN.match(lines[0]):
            self.property_name = lines[0].strip().lstrip(";").strip()

        # Parse disjunctive constraints from full content (multi-line OR clauses)
        self.parse_disjunctive_constraint(content)

        # Parse each line for simple constraints
        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            # Parse variable declarations
            declare_match = self.DECLARE_PATTERN.match(line)
            if declare_match:
                var_type, var_index = declare_match.groups()
                var_index = int(var_index)

                if var_type == "X":
                    self.input_vars.add(var_index)
                elif var_type == "Y":
                    self.output_vars.add(var_index)
                continue

            # Parse assertions (bounds)
            assert_match = self.ASSERT_PATTERN.match(line)
            if assert_match:
                operator, var_type, var_index, value = assert_match.groups()
                var_index = int(var_index)
                value = float(value)

                if var_type == "X":
                    # Input bounds
                    if var_index not in self.input_bounds:
                        self.input_bounds[var_index] = [None, None]

                    if operator == ">=":
                        self.input_bounds[var_index][0] = value  # lower bound
                    elif operator == "<=":
                        self.input_bounds[var_index][1] = value  # upper bound

                elif var_type == "Y":
                    # Output bounds
                    if var_index not in self.output_bounds:
                        self.output_bounds[var_index] = [None, None]

                    if operator == ">=":
                        self.output_bounds[var_index][0] = value  # lower bound
                    elif operator == "<=":
                        self.output_bounds[var_index][1] = value  # upper bound

        # Build property object
        num_inputs = len(self.input_vars)
        num_outputs = len(self.output_vars)

        # Create ordered input bounds list
        input_bounds_list = []
        for i in range(num_inputs):
            if i in self.input_bounds:
                lower = self.input_bounds[i][0]
                upper = self.input_bounds[i][1]
                input_bounds_list.append((lower, upper))
            else:
                # No bounds specified - use None
                input_bounds_list.append((None, None))

        # Create output constraints dict
        output_constraints = {}
        for i in range(num_outputs):
            if i in self.output_bounds:
                lower = self.output_bounds[i][0]
                upper = self.output_bounds[i][1]
                output_constraints[i] = (lower, upper)
            else:
                output_constraints[i] = (None, None)

        return VNNLIBProperty(
            input_bounds=input_bounds_list,
            output_constraints=output_constraints,
            disjunctive_constraints=self.disjunctive_constraints,
            property_name=self.property_name,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
        )

    @staticmethod
    def summarize_property(prop: VNNLIBProperty) -> str:
        """Generate a human-readable summary of a property.

        Args:
            prop: VNNLIBProperty object

        Returns:
            String summary
        """
        lines = [
            f"Property: {prop.property_name}",
            f"Inputs: {prop.num_inputs}, Outputs: {prop.num_outputs}",
            "",
            "Input Bounds:",
        ]

        for i, (lower, upper) in enumerate(prop.input_bounds):
            lower_str = f"{lower:g}" if lower is not None else "-∞"
            upper_str = f"{upper:g}" if upper is not None else "+∞"
            lines.append(f"  X_{i}: [{lower_str}, {upper_str}]")

        lines.append("")
        lines.append("Output Constraints:")

        for i, (lower, upper) in prop.output_constraints.items():
            constraints = []
            if lower is not None:
                constraints.append(f"Y_{i} >= {lower:g}")
            if upper is not None:
                constraints.append(f"Y_{i} <= {upper:g}")

            if constraints:
                lines.append(f"  {' and '.join(constraints)}")
            else:
                lines.append(f"  Y_{i}: no constraints")

        return "\n".join(lines)


def parse_vnnlib(filepath: str, input_dim: int, output_dim: int):
    """
    Parse VNN-LIB file and return constraints in simple format.

    Args:
        filepath: Path to VNN-LIB file
        input_dim: Expected input dimension
        output_dim: Expected output dimension

    Returns:
        (input_constraints, output_constraints)
        where each constraint is (var_idx, operator, value)
        operator is ">=" or "<="
    """
    parser = VNNLIBParser()
    prop = parser.parse_file(filepath)

    input_constraints = []
    output_constraints = []

    # Convert input bounds to constraints
    for i, (lower, upper) in enumerate(prop.input_bounds):
        if lower is not None:
            input_constraints.append((i, ">=", lower))
        if upper is not None:
            input_constraints.append((i, "<=", upper))

    # Convert output bounds to constraints
    for i, (lower, upper) in prop.output_constraints.items():
        if lower is not None:
            output_constraints.append((i, ">=", lower))
        if upper is not None:
            output_constraints.append((i, "<=", upper))

    return input_constraints, output_constraints


def analyze_cersyve_properties():
    """Analyze all CerSyVe VNN-LIB properties and print summary."""
    import os

    vnnlib_dir = "benchmarks/cersyve/vnnlib"

    if not os.path.exists(vnnlib_dir):
        print(f"Directory not found: {vnnlib_dir}")
        return

    parser = VNNLIBParser()
    vnnlib_files = sorted([f for f in os.listdir(vnnlib_dir) if f.endswith(".vnnlib")])

    print("=" * 80)
    print("CerSyVe VNN-LIB Properties Analysis")
    print("=" * 80)
    print()

    properties = {}

    for filename in vnnlib_files:
        filepath = os.path.join(vnnlib_dir, filename)
        prop = parser.parse_file(filepath)
        properties[filename] = prop

        print(f"\n{filename}")
        print("-" * 80)
        print(VNNLIBParser.summarize_property(prop))

    # Summary of patterns
    print("\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("\nCommon Pattern Observed:")
    print("  Input: Box constraints (lower <= X_i <= upper) for each input dimension")
    print("  Output: Safety property Y_0 <= 0 AND Y_1 >= 0")
    print("\nInterpretation:")
    print("  - Input bounds define the region of interest (state space)")
    print("  - Output property likely represents a safety condition:")
    print("    * Y_0 could be a Lyapunov/barrier function (should be non-positive)")
    print("    * Y_1 could be a control Lyapunov function (should be non-negative)")
    print("  - Verification task: Prove property holds for all inputs in bounds")
    print("  - Counterexample search: Find input violating output property")

    return properties


if __name__ == "__main__":
    analyze_cersyve_properties()
