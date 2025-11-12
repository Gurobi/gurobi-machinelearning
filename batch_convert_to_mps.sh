#!/bin/bash
#
# Batch convert ONNX + VNN-LIB benchmarks to MPS files
#
# Usage:
#   ./batch_convert_to_mps.sh [output_base_dir]
#
# Example:
#   ./batch_convert_to_mps.sh mps_output

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
BENCHMARKS_DIR="benchmarks"
OUTPUT_BASE_DIR="${1:-mps_output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Consider running: source .venv-copilot/bin/activate"
    echo ""
fi

# Verify benchmarks directory exists
if [ ! -d "$BENCHMARKS_DIR" ]; then
    echo -e "${RED}Error: Benchmarks directory not found: $BENCHMARKS_DIR${NC}"
    exit 1
fi

# Print header
echo "================================================================================"
echo "BATCH BENCHMARK CONVERSION TO MPS"
echo "================================================================================"
echo "Benchmarks: $BENCHMARKS_DIR"
echo "Output:     $OUTPUT_BASE_DIR"
echo "================================================================================"
echo ""

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Global counters
total_benchmarks=0
success_benchmarks=0
failed_benchmarks=0
skipped_benchmarks=0

total_instances=0
success_instances=0
failed_instances=0
skipped_instances=0

# Master log file
MASTER_LOG="$OUTPUT_BASE_DIR/master_conversion.log"
MASTER_ERROR_LOG="$OUTPUT_BASE_DIR/master_errors.log"
> "$MASTER_LOG"
> "$MASTER_ERROR_LOG"

# Read skip list if exists
SKIP_FILE="$BENCHMARKS_DIR/.skip"
SKIP_LIST=""
if [ -f "$SKIP_FILE" ]; then
    echo "Found skip file: $SKIP_FILE"
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*) continue ;;
        esac
        line=$(echo "$line" | xargs)
        [ -z "$line" ] && continue
        SKIP_LIST="$SKIP_LIST|$line|"
        echo "  Will skip: $line"
    done < "$SKIP_FILE"
    echo ""
fi

# Process each benchmark directory
for benchmark_path in "$BENCHMARKS_DIR"/*; do
    [ ! -d "$benchmark_path" ] && continue

    benchmark_name=$(basename "$benchmark_path")

    # Check if in skip list
    if [ -n "$SKIP_LIST" ]; then
        case "$SKIP_LIST" in
            *"|$benchmark_name|"*)
                echo -e "${YELLOW}[SKIP]${NC} $benchmark_name - In skip list"
                skipped_benchmarks=$((skipped_benchmarks + 1))
                echo "SKIP_BENCHMARK,$benchmark_name,In skip list" >> "$MASTER_LOG"
                continue
                ;;
        esac
    fi

    instances_csv="$benchmark_path/instances.csv"

    # Skip if no instances.csv
    if [ ! -f "$instances_csv" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $benchmark_name - No instances.csv"
        skipped_benchmarks=$((skipped_benchmarks + 1))
        echo "SKIP_BENCHMARK,$benchmark_name,No instances.csv" >> "$MASTER_LOG"
        continue
    fi

    total_benchmarks=$((total_benchmarks + 1))

    # Create output directory for this benchmark
    output_dir="$OUTPUT_BASE_DIR/$benchmark_name"
    mkdir -p "$output_dir"

    log_file="$output_dir/conversion.log"
    error_log="$output_dir/errors.log"
    > "$log_file"
    > "$error_log"

    echo -e "${BLUE}[BENCH]${NC} $benchmark_name"

    benchmark_total=0
    benchmark_success=0
    benchmark_failed=0
    benchmark_skipped=0
    benchmark_has_failure=false

    # Read instances.csv
    while IFS=',' read -r onnx_path vnnlib_path timeout; do
        # Skip header and empty lines
        [[ "$onnx_path" == "onnx" ]] && continue
        [ -z "$onnx_path" ] && continue

        benchmark_total=$((benchmark_total + 1))
        total_instances=$((total_instances + 1))

        # Build full paths
        full_onnx="$benchmark_path/$onnx_path"
        full_vnnlib="$benchmark_path/$vnnlib_path"

        # Handle .gz extension
        if [ ! -f "$full_onnx" ] && [ -f "$full_onnx.gz" ]; then
            full_onnx="$full_onnx.gz"
        fi

        # Generate output filename
        onnx_basename=$(basename "$onnx_path" .onnx.gz)
        onnx_basename=$(basename "$onnx_basename" .onnx)
        vnnlib_basename=$(basename "$vnnlib_path" .vnnlib)

        onnx_subdir=$(dirname "$onnx_path" | sed 's|onnx/||' | sed 's|/|_|g')

        if [ -n "$onnx_subdir" ] && [ "$onnx_subdir" != "." ]; then
            output_mps="$output_dir/${onnx_subdir}_${onnx_basename}_${vnnlib_basename}.mps.bz2"
        else
            output_mps="$output_dir/${onnx_basename}_${vnnlib_basename}.mps.bz2"
        fi

        # Check if files exist
        if [ ! -f "$full_onnx" ]; then
            echo -e "  ${RED}[FAIL]${NC} ONNX not found: $onnx_path"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAIL_MISSING_ONNX,$benchmark_name,$onnx_path,$vnnlib_path" >> "$log_file"
            echo "FAIL_MISSING_ONNX,$benchmark_name,$onnx_path,$vnnlib_path" >> "$MASTER_LOG"
            echo "Missing ONNX: $full_onnx" >> "$error_log"
            continue
        fi

        if [ ! -f "$full_vnnlib" ]; then
            echo -e "  ${RED}[FAIL]${NC} VNN-LIB not found: $vnnlib_path"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAIL_MISSING_VNNLIB,$benchmark_name,$onnx_path,$vnnlib_path" >> "$log_file"
            echo "FAIL_MISSING_VNNLIB,$benchmark_name,$onnx_path,$vnnlib_path" >> "$MASTER_LOG"
            echo "Missing VNN-LIB: $full_vnnlib" >> "$error_log"
            continue
        fi

        # Skip if output already exists
        if [ -f "$output_mps" ]; then
            file_size=$(stat -f%z "$output_mps" 2>/dev/null || stat -c%s "$output_mps" 2>/dev/null || echo "?")
            file_size_kb=$((file_size / 1024))
            echo -e "  ${YELLOW}[EXIST]${NC} $(basename $onnx_path) + $(basename $vnnlib_path) (${file_size_kb} KB)"
            benchmark_skipped=$((benchmark_skipped + 1))
            skipped_instances=$((skipped_instances + 1))
            echo "EXISTS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$log_file"
            echo "EXISTS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$MASTER_LOG"
            continue
        fi

        # Convert using the module
        echo -n "  [$(printf '%3d' $benchmark_total)] $(basename $onnx_path) + $(basename $vnnlib_path) ... "

        if python -m gurobi_ml.vnnlib.cli convert "$full_onnx" "$full_vnnlib" "$output_mps" --quiet --no-verify 2>> "$error_log"; then
            file_size=$(stat -f%z "$output_mps" 2>/dev/null || stat -c%s "$output_mps" 2>/dev/null || echo "?")
            file_size_kb=$((file_size / 1024))
            echo -e "${GREEN}OK${NC} (${file_size_kb} KB)"
            benchmark_success=$((benchmark_success + 1))
            success_instances=$((success_instances + 1))
            echo "SUCCESS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$log_file"
            echo "SUCCESS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$MASTER_LOG"
        else
            echo -e "${RED}FAILED${NC}"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAILED,$benchmark_name,$onnx_path,$vnnlib_path,See error log" >> "$log_file"
            echo "FAILED,$benchmark_name,$onnx_path,$vnnlib_path,See error log" >> "$MASTER_LOG"
        fi

    done < "$instances_csv"

    # Benchmark summary
    echo ""
    echo "  Summary for $benchmark_name:"
    echo "    Total:   $benchmark_total"
    echo -e "    ${GREEN}Success: $benchmark_success${NC}"
    echo -e "    ${RED}Failed:  $benchmark_failed${NC}"
    echo -e "    ${YELLOW}Skipped: $benchmark_skipped${NC}"

    # Update benchmark counter
    if [ $benchmark_failed -gt 0 ]; then
        failed_benchmarks=$((failed_benchmarks + 1))
    elif [ $benchmark_success -gt 0 ]; then
        success_benchmarks=$((success_benchmarks + 1))
    else
        skipped_benchmarks=$((skipped_benchmarks + 1))
    fi

done

# Print final summary
echo ""
echo "================================================================================"
echo "FINAL SUMMARY"
echo "================================================================================"
echo ""
echo "Benchmarks:"
echo "  Total:   $total_benchmarks"
echo -e "  ${GREEN}Success: $success_benchmarks${NC}"
echo -e "  ${RED}Failed:  $failed_benchmarks${NC}"
echo -e "  ${YELLOW}Skipped: $skipped_benchmarks${NC}"
echo ""
echo "Instances:"
echo "  Total:   $total_instances"
echo -e "  ${GREEN}Success: $success_instances${NC}"
echo -e "  ${RED}Failed:  $failed_instances${NC}"
echo -e "  ${YELLOW}Skipped: $skipped_instances${NC}"
echo ""
echo "================================================================================"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Master log:       $MASTER_LOG"
if [ $failed_instances -gt 0 ]; then
    echo "Master errors:    $MASTER_ERROR_LOG"
fi
echo "================================================================================"

# Exit with error if any failed
if [ $failed_instances -gt 0 ]; then
    exit 1
fi

exit 0


# Print header
echo "================================================================================"
echo "BATCH BENCHMARK CONVERSION TO MPS"
echo "================================================================================"
echo "Benchmarks: $BENCHMARKS_DIR"
echo "Output:     $OUTPUT_BASE_DIR"
echo "================================================================================"
echo ""

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Global counters
total_benchmarks=0
success_benchmarks=0
failed_benchmarks=0
skipped_benchmarks=0

total_instances=0
success_instances=0
failed_instances=0
skipped_instances=0

# Master log file
MASTER_LOG="$OUTPUT_BASE_DIR/master_conversion.log"
MASTER_ERROR_LOG="$OUTPUT_BASE_DIR/master_errors.log"
> "$MASTER_LOG"
> "$MASTER_ERROR_LOG"

# Read skip list if exists (simple string list)
SKIP_FILE="$BENCHMARKS_DIR/.skip"
SKIP_LIST=""
if [ -f "$SKIP_FILE" ]; then
    echo "Found skip file: $SKIP_FILE"
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        case "$line" in
            ''|'#'*) continue ;;
        esac
        # Trim whitespace
        line=$(echo "$line" | xargs)
        [ -z "$line" ] && continue
        SKIP_LIST="$SKIP_LIST|$line|"
        echo "  Will skip: $line"
    done < "$SKIP_FILE"
    echo ""
fi

# Process each benchmark directory
for benchmark_path in "$BENCHMARKS_DIR"/*; do
    # Skip if not a directory
    if [ ! -d "$benchmark_path" ]; then
        continue
    fi

    benchmark_name=$(basename "$benchmark_path")

    # Check if in skip list (simple string matching)
    if [ -n "$SKIP_LIST" ]; then
        case "$SKIP_LIST" in
            *"|$benchmark_name|"*)
                echo -e "${YELLOW}[SKIP]${NC} $benchmark_name - In skip list"
                skipped_benchmarks=$((skipped_benchmarks + 1))
                echo "SKIP_BENCHMARK,$benchmark_name,In skip list" >> "$MASTER_LOG"
                continue
                ;;
        esac
    fi

    instances_csv="$benchmark_path/instances.csv"

    # Skip if no instances.csv
    if [ ! -f "$instances_csv" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $benchmark_name - No instances.csv"
        skipped_benchmarks=$((skipped_benchmarks + 1))
        echo "SKIP_BENCHMARK,$benchmark_name,No instances.csv" >> "$MASTER_LOG"
        continue
    fi

    total_benchmarks=$((total_benchmarks + 1))

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Benchmark: $benchmark_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Create output directory for this benchmark
    output_dir="$OUTPUT_BASE_DIR/$benchmark_name"
    mkdir -p "$output_dir"

    # Log files for this benchmark
    log_file="$output_dir/conversion.log"
    error_log="$output_dir/errors.log"
    > "$log_file"
    > "$error_log"

    # Counters for this benchmark
    benchmark_total=0
    benchmark_success=0
    benchmark_failed=0
    benchmark_skipped=0
    benchmark_has_failure=false

    # Read instances.csv and process each line
    while IFS=',' read -r onnx_path vnnlib_path timeout || [ -n "$onnx_path" ]; do
        # Skip empty lines
        if [ -z "$onnx_path" ]; then
            continue
        fi

        # Skip comments
        if [[ "$onnx_path" == "#"* ]]; then
            continue
        fi

        # If previous conversion failed, skip rest of this benchmark
        if [ "$benchmark_has_failure" = true ]; then
            benchmark_skipped=$((benchmark_skipped + 1))
            skipped_instances=$((skipped_instances + 1))
            echo "SKIP_AFTER_FAILURE,$benchmark_name,$onnx_path,$vnnlib_path" >> "$log_file"
            continue
        fi

        benchmark_total=$((benchmark_total + 1))
        total_instances=$((total_instances + 1))

        # Build full paths
        full_onnx="$benchmark_path/$onnx_path"
        full_vnnlib="$benchmark_path/$vnnlib_path"

        # Check for compressed ONNX files (.onnx.gz)
        if [ ! -f "$full_onnx" ] && [ -f "$full_onnx.gz" ]; then
            full_onnx="$full_onnx.gz"
        fi

        # Generate output filename (preserve subdirectory structure to avoid collisions)
        # Strip both .onnx and .onnx.gz extensions
        onnx_basename=$(basename "$onnx_path" .onnx.gz)
        onnx_basename=$(basename "$onnx_basename" .onnx)
        vnnlib_basename=$(basename "$vnnlib_path" .vnnlib)

        # Extract subdirectory from paths to handle cases like medical/ruarobot
        onnx_subdir=$(dirname "$onnx_path" | sed 's|onnx/||' | sed 's|/|_|g')
        vnnlib_subdir=$(dirname "$vnnlib_path" | sed 's|vnnlib/||' | sed 's|/|_|g')

        # Build unique filename with subdirectory prefix if present
        if [ -n "$onnx_subdir" ] && [ "$onnx_subdir" != "." ]; then
            output_mps="$output_dir/${onnx_subdir}_${onnx_basename}_${vnnlib_basename}.mps.bz2"
        else
            output_mps="$output_dir/${onnx_basename}_${vnnlib_basename}.mps.bz2"
        fi

        # Check if files exist
        if [ ! -f "$full_onnx" ]; then
            echo -e "  ${RED}[FAIL]${NC} ONNX not found: $onnx_path"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAIL_MISSING_ONNX,$benchmark_name,$onnx_path,$vnnlib_path" >> "$log_file"
            echo "FAIL_MISSING_ONNX,$benchmark_name,$onnx_path,$vnnlib_path" >> "$MASTER_LOG"
            echo "Missing ONNX: $full_onnx" >> "$error_log"
            continue
        fi

        if [ ! -f "$full_vnnlib" ]; then
            echo -e "  ${RED}[FAIL]${NC} VNN-LIB not found: $vnnlib_path"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAIL_MISSING_VNNLIB,$benchmark_name,$onnx_path,$vnnlib_path" >> "$log_file"
            echo "FAIL_MISSING_VNNLIB,$benchmark_name,$onnx_path,$vnnlib_path" >> "$MASTER_LOG"
            echo "Missing VNN-LIB: $full_vnnlib" >> "$error_log"
            continue
        fi

        # Skip if output already exists
        if [ -f "$output_mps" ]; then
            file_size=$(stat -f%z "$output_mps" 2>/dev/null || stat -c%s "$output_mps" 2>/dev/null || echo "?")
            file_size_kb=$((file_size / 1024))
            echo -e "  ${YELLOW}[EXIST]${NC} $(basename $onnx_path) + $(basename $vnnlib_path) (${file_size_kb} KB)"
            benchmark_skipped=$((benchmark_skipped + 1))
            skipped_instances=$((skipped_instances + 1))
            echo "EXISTS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$log_file"
            echo "EXISTS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$MASTER_LOG"
            continue
        fi

        # Convert
        echo -n "  [$(printf '%3d' $benchmark_total)] $(basename $onnx_path) + $(basename $vnnlib_path) ... "

        if python "$CONVERTER" "$full_onnx" "$full_vnnlib" "$output_mps" --quiet 2>> "$error_log"; then
            file_size=$(stat -f%z "$output_mps" 2>/dev/null || stat -c%s "$output_mps" 2>/dev/null || echo "?")
            file_size_kb=$((file_size / 1024))
            echo -e "${GREEN}OK${NC} (${file_size_kb} KB)"
            benchmark_success=$((benchmark_success + 1))
            success_instances=$((success_instances + 1))
            echo "SUCCESS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$log_file"
            echo "SUCCESS,$benchmark_name,$onnx_path,$vnnlib_path,$output_mps,$file_size" >> "$MASTER_LOG"
        else
            echo -e "${RED}FAILED${NC}"
            benchmark_failed=$((benchmark_failed + 1))
            failed_instances=$((failed_instances + 1))
            benchmark_has_failure=true
            echo "FAILED,$benchmark_name,$onnx_path,$vnnlib_path,See error log" >> "$log_file"
            echo "FAILED,$benchmark_name,$onnx_path,$vnnlib_path,See error log" >> "$MASTER_LOG"
        fi

    done < "$instances_csv"

    # Benchmark summary
    echo ""
    echo "  Summary for $benchmark_name:"
    echo "    Total:   $benchmark_total"
    echo -e "    ${GREEN}Success: $benchmark_success${NC}"
    echo -e "    ${RED}Failed:  $benchmark_failed${NC}"
    echo -e "    ${YELLOW}Skipped: $benchmark_skipped${NC}"

    # Update benchmark counter
    if [ $benchmark_failed -gt 0 ]; then
        failed_benchmarks=$((failed_benchmarks + 1))
    elif [ $benchmark_success -gt 0 ]; then
        success_benchmarks=$((success_benchmarks + 1))
    else
        skipped_benchmarks=$((skipped_benchmarks + 1))
    fi

done

# Print final summary
echo ""
echo "================================================================================"
echo "FINAL SUMMARY"
echo "================================================================================"
echo ""
echo "Benchmarks:"
echo "  Total:   $total_benchmarks"
echo -e "  ${GREEN}Success: $success_benchmarks${NC}"
echo -e "  ${RED}Failed:  $failed_benchmarks${NC}"
echo -e "  ${YELLOW}Skipped: $skipped_benchmarks${NC}"
echo ""
echo "Instances:"
echo "  Total:   $total_instances"
echo -e "  ${GREEN}Success: $success_instances${NC}"
echo -e "  ${RED}Failed:  $failed_instances${NC}"
echo -e "  ${YELLOW}Skipped: $skipped_instances${NC}"
echo ""
echo "================================================================================"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Master log:       $MASTER_LOG"
if [ $failed_instances -gt 0 ]; then
    echo "Master errors:    $MASTER_ERROR_LOG"
fi
echo "================================================================================"

# Exit with error if any failed
if [ $failed_instances -gt 0 ]; then
    exit 1
fi

exit 0
