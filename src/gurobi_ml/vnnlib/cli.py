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
Command-line interface for VNN-LIB tools.

Provides a CLI for converting ONNX + VNN-LIB verification problems to MPS format
and inspecting VNN-LIB property files.
"""

import sys
import os
import argparse
from .converter import convert_to_mps
from .parser import parse_vnnlib_simple


def main_convert():
    """Main entry point for onnx-to-mps conversion tool."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX + VNN-LIB to Gurobi MPS file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gurobi-ml-vnnlib convert model.onnx prop.vnnlib output.mps
  gurobi-ml-vnnlib convert model.onnx prop.vnnlib output.mps --no-verify
  gurobi-ml-vnnlib convert model.onnx prop.vnnlib output.mps --quiet
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


def main_inspect():
    """Main entry point for VNN-LIB inspection tool."""
    parser = argparse.ArgumentParser(
        description="Inspect VNN-LIB property file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gurobi-ml-vnnlib inspect property.vnnlib
  gurobi-ml-vnnlib inspect property.vnnlib --verbose
        """,
    )

    parser.add_argument("vnnlib_file", help="Path to VNN-LIB property file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed constraints"
    )

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.vnnlib_file):
        print(f"Error: VNN-LIB file not found: {args.vnnlib_file}", file=sys.stderr)
        sys.exit(1)

    # Parse and display
    try:
        prop = parse_vnnlib_simple(args.vnnlib_file)

        print(f"Property: {prop.property_name}")
        print(f"Inputs: {prop.num_inputs}, Outputs: {prop.num_outputs}")
        print(
            f"Input bounds: {len([b for b in prop.input_bounds if b != (None, None)])}"
        )
        print(
            f"Simple output constraints: {len([c for c in prop.output_constraints.values() if c != (None, None)])}"
        )
        print(f"Disjunctive constraints: {len(prop.disjunctive_constraints)}")

        if args.verbose and prop.disjunctive_constraints:
            print("\nDisjunctive constraints (sample):")
            for i, or_clause in enumerate(prop.disjunctive_constraints[:10]):
                print(f"  OR clause {i}: {len(or_clause)} disjuncts")
                for j, (idx1, op, idx2) in enumerate(or_clause[:5]):
                    print(f"    Y_{idx1} {op} Y_{idx2}")
                if len(or_clause) > 5:
                    print(f"    ... and {len(or_clause) - 5} more")
            if len(prop.disjunctive_constraints) > 10:
                print(
                    f"  ... and {len(prop.disjunctive_constraints) - 10} more OR clauses"
                )

    except Exception as e:
        print(f"Error parsing VNN-LIB file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        description="Gurobi Machine Learning VNN-LIB Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert ONNX + VNN-LIB to MPS"
    )
    convert_parser.add_argument("onnx_file", help="Path to ONNX model file")
    convert_parser.add_argument("vnnlib_file", help="Path to VNN-LIB property file")
    convert_parser.add_argument("output_mps", help="Output MPS file path")
    convert_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification against ONNX Runtime",
    )
    convert_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output"
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect VNN-LIB property")
    inspect_parser.add_argument("vnnlib_file", help="Path to VNN-LIB property file")
    inspect_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed constraints"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert":
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

    elif args.command == "inspect":
        # Check file exists
        if not os.path.exists(args.vnnlib_file):
            print(f"Error: VNN-LIB file not found: {args.vnnlib_file}", file=sys.stderr)
            sys.exit(1)

        # Parse and display
        try:
            prop = parse_vnnlib_simple(args.vnnlib_file)

            print(f"Property: {prop.property_name}")
            print(f"Inputs: {prop.num_inputs}, Outputs: {prop.num_outputs}")
            print(
                f"Input bounds: {len([b for b in prop.input_bounds if b != (None, None)])}"
            )
            print(
                f"Simple output constraints: {len([c for c in prop.output_constraints.values() if c != (None, None)])}"
            )
            print(f"Disjunctive constraints: {len(prop.disjunctive_constraints)}")

            if args.verbose and prop.disjunctive_constraints:
                print("\nDisjunctive constraints (sample):")
                for i, or_clause in enumerate(prop.disjunctive_constraints[:10]):
                    print(f"  OR clause {i}: {len(or_clause)} disjuncts")
                    for j, (idx1, op, idx2) in enumerate(or_clause[:5]):
                        print(f"    Y_{idx1} {op} Y_{idx2}")
                    if len(or_clause) > 5:
                        print(f"    ... and {len(or_clause) - 5} more")
                if len(prop.disjunctive_constraints) > 10:
                    print(
                        f"  ... and {len(prop.disjunctive_constraints) - 10} more OR clauses"
                    )

        except Exception as e:
            print(f"Error parsing VNN-LIB file: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
