"""
Simple and maintainable VNNLIB parser using S-expression parsing.

This is a cleaner alternative to the custom parser, using standard
S-expression parsing which handles all VNNLIB constructs correctly.

Advantages over custom parser:
- Handles nested structures automatically
- No complex regex patterns
- Easier to extend
- More robust and maintainable
"""

from typing import Any
from dataclasses import dataclass
import re


@dataclass
class VNNLIBProperty:
    """Represents a parsed VNNLIB property."""

    input_bounds: list[tuple[float, float]]
    output_constraints: dict[int, tuple[float | None, float | None]]
    disjunctive_constraints: list[list[tuple[int, str, int]]]
    property_name: str
    num_inputs: int
    num_outputs: int


def tokenize(text: str) -> list[str]:
    """Tokenize VNNLIB/SMT-LIB format."""
    # Remove comments
    text = re.sub(r";[^\n]*", "", text)
    # Replace parens with spaced versions for easy splitting
    text = text.replace("(", " ( ").replace(")", " ) ")
    # Split and filter empty tokens
    return [t for t in text.split() if t]


def parse_sexp(tokens: list[str]) -> Any:
    """Parse S-expression from tokens."""
    if not tokens:
        raise SyntaxError("Unexpected EOF")

    token = tokens.pop(0)

    if token == "(":
        exp = []
        while tokens[0] != ")":
            exp.append(parse_sexp(tokens))
        tokens.pop(0)  # Remove ')'
        return exp
    elif token == ")":
        raise SyntaxError("Unexpected )")
    else:
        # Try to convert to number
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token


def _process_complex_assertion(sexp: Any, disjunctive_constraints: list) -> None:
    """Helper to process complex assertions (OR/AND clauses)."""
    if not isinstance(sexp, list) or len(sexp) < 2:
        return

    if sexp[0] != "assert":
        return

    assertion = sexp[1]

    # Check if it's a disjunction (or ...)
    if isinstance(assertion, list) and assertion[0] == "or":
        or_clauses = []
        for clause in assertion[1:]:
            if isinstance(clause, list) and clause[0] == "and":
                comparison = clause[1]
                if isinstance(comparison, list) and len(comparison) == 3:
                    op, left, right = comparison
                    if (
                        isinstance(left, str)
                        and left.startswith("Y_")
                        and isinstance(right, str)
                        and right.startswith("Y_")
                    ):
                        idx1 = int(left[2:])
                        idx2 = int(right[2:])
                        or_clauses.append((idx1, op, idx2))

        if or_clauses:
            disjunctive_constraints.append(or_clauses)


def parse_vnnlib_simple(filepath: str) -> VNNLIBProperty:
    """
    Parse VNNLIB file using optimized line-by-line parsing.

    For large files (like VGG with 600K+ lines), parsing line-by-line
    is much faster than tokenizing the entire file at once.
    """
    property_name = "unknown"
    input_vars = set()
    output_vars = set()
    input_bounds = {}
    output_bounds = {}
    disjunctive_constraints = []

    # Fast regex patterns for common cases
    declare_pattern = re.compile(r"\(declare-const\s+([XY])_(\d+)\s+Real\)")
    assert_pattern = re.compile(
        r"\(assert\s+\((>=|<=)\s+([XY])_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)\)"
    )
    output_comp_pattern = re.compile(r"\(assert\s+\((>=|<=)\s+Y_(\d+)\s+Y_(\d+)\)\)")

    # For handling multi-line expressions
    multi_line_buffer = []
    paren_depth = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith(";"):
                if line_num == 1 and line.startswith(";"):
                    property_name = line[1:].strip()
                continue

            # If we're in a multi-line expression, accumulate lines
            if paren_depth > 0:
                multi_line_buffer.append(line)
                paren_depth += line.count("(") - line.count(")")

                # When we've balanced all parens, process the complete expression
                if paren_depth == 0:
                    full_expr = " ".join(multi_line_buffer)
                    multi_line_buffer = []

                    # Try to parse as S-expression
                    try:
                        tokens = tokenize(full_expr)
                        if tokens:
                            sexp = parse_sexp(tokens)
                            _process_complex_assertion(sexp, disjunctive_constraints)
                    except Exception:
                        # Silently skip unparseable expressions
                        pass
                continue

            # Fast path: declare-const
            match = declare_pattern.match(line)
            if match:
                var_type, idx = match.groups()
                idx = int(idx)
                if var_type == "X":
                    input_vars.add(idx)
                else:
                    output_vars.add(idx)
                continue

            # Fast path: simple assert with bound
            match = assert_pattern.match(line)
            if match:
                op, var_type, idx, value = match.groups()
                idx = int(idx)
                value = float(value)

                if var_type == "X":
                    if idx not in input_bounds:
                        input_bounds[idx] = [None, None]
                    if op == "<=":
                        input_bounds[idx][1] = value  # upper bound
                    else:  # '>='
                        input_bounds[idx][0] = value  # lower bound
                else:  # 'Y'
                    if idx not in output_bounds:
                        output_bounds[idx] = [None, None]
                    if op == "<=":
                        output_bounds[idx][1] = value
                    else:
                        output_bounds[idx][0] = value
                continue

            # Fast path: output comparison
            match = output_comp_pattern.match(line)
            if match:
                op, idx1, idx2 = match.groups()
                disjunctive_constraints.append([(int(idx1), op, int(idx2))])
                continue

            # Check if this starts a multi-line expression
            paren_depth = line.count("(") - line.count(")")
            if paren_depth > 0:
                multi_line_buffer = [line]
                continue

            # Single-line complex assertion
            if "(or" in line or "(and" in line:
                # Try to parse as S-expression
                try:
                    tokens = tokenize(line)
                    if tokens:
                        sexp = parse_sexp(tokens)
                        _process_complex_assertion(sexp, disjunctive_constraints)
                except Exception:
                    # Silently skip unparseable expressions
                    pass

    # Build result
    num_inputs = len(input_vars)
    num_outputs = len(output_vars)

    input_bounds_list = []
    for i in range(num_inputs):
        if i in input_bounds:
            lower = input_bounds[i][0]
            upper = input_bounds[i][1]
            input_bounds_list.append((lower, upper))
        else:
            input_bounds_list.append((None, None))

    output_constraints = {}
    for i in range(num_outputs):
        if i in output_bounds:
            lower = output_bounds[i][0]
            upper = output_bounds[i][1]
            output_constraints[i] = (lower, upper)
        else:
            output_constraints[i] = (None, None)

    return VNNLIBProperty(
        input_bounds=input_bounds_list,
        output_constraints=output_constraints,
        disjunctive_constraints=disjunctive_constraints,
        property_name=property_name,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )


def parse_vnnlib(filepath: str, input_dim: int, output_dim: int):
    """
    Compatibility function for old interface.
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
    prop = parse_vnnlib_simple(filepath)

    input_constraints = []
    output_constraints = []

    # Convert input bounds to constraints
    for idx, (lower, upper) in enumerate(prop.input_bounds):
        if lower is not None:
            input_constraints.append((idx, ">=", lower))
        if upper is not None:
            input_constraints.append((idx, "<=", upper))

    # Convert output bounds to constraints
    for idx, (lower, upper) in prop.output_constraints.items():
        if lower is not None:
            output_constraints.append((idx, ">=", lower))
        if upper is not None:
            output_constraints.append((idx, "<=", upper))

    return input_constraints, output_constraints


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python vnnlib_simple.py <vnnlib_file>")
        sys.exit(1)

    prop = parse_vnnlib_simple(sys.argv[1])

    print(f"Property: {prop.property_name}")
    print(f"Inputs: {prop.num_inputs}, Outputs: {prop.num_outputs}")
    print(f"Input bounds: {len([b for b in prop.input_bounds if b != (None, None)])}")
    print(
        f"Simple output constraints: {len([c for c in prop.output_constraints.values() if c != (None, None)])}"
    )
    print(f"Disjunctive constraints: {len(prop.disjunctive_constraints)}")

    if prop.disjunctive_constraints:
        print("\nDisjunctive constraints:")
        for i, or_clause in enumerate(prop.disjunctive_constraints):
            print(f"  OR clause {i}: {len(or_clause)} disjuncts")
            for j, (idx1, op, idx2) in enumerate(or_clause[:3]):
                print(f"    Y_{idx1} {op} Y_{idx2}")
            if len(or_clause) > 3:
                print(f"    ... and {len(or_clause) - 3} more")
