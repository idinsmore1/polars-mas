#!/bin/bash
# Run benchmarks for both R PheWAS and Python polars-mas
# Usage: ./run_benchmarks.sh [--r-only | --python-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure results directory exists
mkdir -p results

run_r_benchmark() {
    echo "========================================"
    echo "Running R PheWAS Benchmark"
    echo "========================================"
    echo ""
    Rscript benchmark_phewas_r.R
    echo ""
}

run_python_benchmark() {
    echo "========================================"
    echo "Running Python polars-mas Benchmark"
    echo "========================================"
    echo ""
    cd "$SCRIPT_DIR/../.."
    uv run python src/tests/benchmark_phewas_python.py
    cd "$SCRIPT_DIR"
    echo ""
}

compare_results() {
    echo "========================================"
    echo "Benchmark Comparison"
    echo "========================================"
    echo ""

    R_RESULTS="results/benchmark_r_10_covariates_phewas_results.csv"
    PY_RESULTS="results/benchmark_python_10_covariates_polars_mas_results.csv"

    if [[ -f "$R_RESULTS" && -f "$PY_RESULTS" ]]; then
        echo "Results comparison (R vs Python):"
        echo ""
        echo "Samples     R (sec)    Python (sec)    Speedup"
        echo "-----------------------------------------------"

        # Skip headers and join on samples column
        tail -n +2 "$R_RESULTS" | while IFS=, read -r r_file r_samples r_phecodes r_time; do
            py_line=$(tail -n +2 "$PY_RESULTS" | grep ",$r_samples,")
            if [[ -n "$py_line" ]]; then
                py_time=$(echo "$py_line" | cut -d',' -f4)
                speedup=$(echo "scale=2; $r_time / $py_time" | bc 2>/dev/null || echo "N/A")
                printf "%-10s  %-10.2f  %-14.2f  %sx\n" "$r_samples" "$r_time" "$py_time" "$speedup"
            fi
        done
        echo ""
    else
        echo "Run both benchmarks first to generate comparison."
        [[ ! -f "$R_RESULTS" ]] && echo "  Missing: $R_RESULTS"
        [[ ! -f "$PY_RESULTS" ]] && echo "  Missing: $PY_RESULTS"
    fi
}

# Parse arguments
case "${1:-all}" in
    --r-only)
        run_r_benchmark
        ;;
    --python-only)
        run_python_benchmark
        ;;
    --compare)
        compare_results
        ;;
    all|*)
        run_r_benchmark
        run_python_benchmark
        compare_results
        ;;
esac
