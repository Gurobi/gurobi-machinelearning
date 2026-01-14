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
VNN-LIB parser for neural network verification properties.

This module provides robust parsing of VNN-LIB (SMT-LIB2-based) format files used in
neural network verification benchmarks and competitions.

The parser handles:
- Input variable declarations and bounds
- Output variable declarations and bounds
- Simple conjunctive constraints (AND)
- Complex disjunctive constraints (OR)
- Optimized line-by-line parsing for large files (600K+ lines)

References:
    VNN-LIB format specification: https://www.vnnlib.org/
"""

from typing import Any
from dataclasses import dataclass
import re


@dataclass
class VNNLIBProperty:
    """Represents a parsed VNN-LIB property specification.

    Attributes:
        input_bounds: List of (lower, upper) bound tuples for each input variable.
                     None indicates unbounded in that direction.
        output_constraints: Dict mapping output index to (lower, upper) bounds.
                           Used for simple conjunctive constraints.
        disjunctive_constraints: List of OR clauses, where each clause is a list of
                                (idx1, operator, idx2) tuples representing Y_idx1 op Y_idx2.
        property_name: Name or description of the property (from first comment line).
        num_inputs: Number of input variables.
        num_outputs: Number of output variables.
    """

    input_bounds: list[tuple[float | None, float | None]]
    output_constraints: dict[int, tuple[float | None, float | None]]
    disjunctive_constraints: list[list[tuple[int, str, int]]]
    property_name: str
    num_inputs: int
    num_outputs: int


def tokenize(text: str) -> list[str]:
    """Tokenize VNN-LIB/SMT-LIB2 format text.

    Args:
        text: Raw VNN-LIB text content.

    Returns:
        List of tokens (strings).
    """
    # Remove comments
    text = re.sub(r";[^\n]*", "", text)
    # Replace parens with spaced versions for easy splitting
    text = text.replace("(", " ( ").replace(")", " ) ")
    # Split and filter empty tokens
    return [t for t in text.split() if t]


def parse_sexp(tokens: list[str]) -> Any:
    """Parse S-expression from token list.

    Args:
        tokens: List of tokens (modified in place).

    Returns:
        Parsed S-expression (nested lists and atoms).

    Raises:
        SyntaxError: If parentheses are unbalanced or expression is malformed.
    """
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
    """Process complex assertions with OR/AND clauses.

    Extracts disjunctive constraints of the form:
        (assert (or (and (>= Y_i Y_j)) (and (>= Y_k Y_l)) ...))

    Args:
        sexp: Parsed S-expression.
        disjunctive_constraints: List to append extracted constraints to (modified in place).
    """
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
    """Parse VNN-LIB file using optimized line-by-line parsing.

    This implementation is optimized for large files (e.g., VGG models with 600K+ lines)
    by parsing line-by-line rather than tokenizing the entire file at once.

    Supports:
    - Variable declarations: (declare-const X_i Real), (declare-const Y_j Real)
    - Simple bounds: (assert (>= X_i value)), (assert (<= X_i value))
    - Output comparisons: (assert (>= Y_i Y_j)), (assert (<= Y_i Y_j))
    - Complex OR clauses: (assert (or (and ...) (and ...) ...))

    Args:
        filepath: Path to VNN-LIB file.

    Returns:
        VNNLIBProperty object with parsed constraints.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file has invalid format.
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
                    complete_expr = " ".join(multi_line_buffer)
                    try:
                        tokens = tokenize(complete_expr)
                        if tokens:
                            sexp = parse_sexp(tokens)
                            _process_complex_assertion(sexp, disjunctive_constraints)
                    except Exception:
                        # Silently skip unparseable expressions
                        pass
                    multi_line_buffer = []
                continue

            # Fast path: variable declaration
            match = declare_pattern.match(line)
            if match:
                var_type, idx = match.groups()
                idx = int(idx)
                if var_type == "X":
                    input_vars.add(idx)
                else:  # 'Y'
                    output_vars.add(idx)
                continue

            # Fast path: simple bound assertion
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


def parse_vnnlib(filepath: str, input_dim: int = None, output_dim: int = None):
    """Parse VNN-LIB file and return constraints in legacy format.

    This is a compatibility function for code that expects the old constraint format.
    New code should use parse_vnnlib_simple() instead.

    Args:
        filepath: Path to VNN-LIB file.
        input_dim: Expected input dimension (optional, for validation).
        output_dim: Expected output dimension (optional, for validation).

    Returns:
        Tuple of (input_constraints, output_constraints) where each constraint
        is a tuple of (var_idx, operator, value) with operator being ">=" or "<=".
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parser.py <vnnlib_file>")
        print("\nParse and display summary of VNN-LIB property file.")
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
        print("\nDisjunctive constraints (sample):")
        for i, or_clause in enumerate(prop.disjunctive_constraints[:5]):
            print(f"  OR clause {i}: {len(or_clause)} disjuncts")
            for j, (idx1, op, idx2) in enumerate(or_clause[:3]):
                print(f"    Y_{idx1} {op} Y_{idx2}")
            if len(or_clause) > 3:
                print(f"    ... and {len(or_clause) - 3} more")
        if len(prop.disjunctive_constraints) > 5:
            print(f"  ... and {len(prop.disjunctive_constraints) - 5} more OR clauses")
