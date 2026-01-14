#!/bin/bash
# Run covariate scaling benchmarks for both R PheWAS and Python polars-mas
#
# This benchmark demonstrates how polars-mas handles larger numbers of covariates
# more efficiently than the R PheWAS package.
#
# Usage: ./run_covariate_scaling_benchmarks.sh [--r-only | --python-only | --compare]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure results directory exists
mkdir -p results

run_r_benchmark() {
    echo "========================================"
    echo "Running R PheWAS Covariate Scaling Benchmark"
    echo "========================================"
    echo ""
    Rscript benchmark_covariate_scaling_r.R
    echo ""
}

run_python_benchmark() {
    echo "========================================"
    echo "Running Python polars-mas Covariate Scaling Benchmark"
    echo "========================================"
    echo ""
    cd "$SCRIPT_DIR/.."
    uv run python benchmarks/benchmark_covariate_scaling_python.py
    cd "$SCRIPT_DIR"
    echo ""
}

compare_results() {
    echo "========================================"
    echo "Covariate Scaling Benchmark Comparison"
    echo "========================================"
    echo ""

    R_RESULTS="results/benchmark_covariate_scaling_r_results.csv"
    PY_RESULTS="results/benchmark_covariate_scaling_python_results.csv"

    if [[ -f "$R_RESULTS" && -f "$PY_RESULTS" ]]; then
        echo "Results comparison (R vs Python polars-mas):"
        echo ""
        printf "%-12s %12s %12s %12s\n" "Covariates" "R (sec)" "Python (sec)" "Speedup"
        echo "----------------------------------------------------"

        # Skip headers and join on n_covariates column
        tail -n +2 "$R_RESULTS" | while IFS=, read -r n_covs r_time r_covariates; do
            py_line=$(tail -n +2 "$PY_RESULTS" | grep "^$n_covs,")
            if [[ -n "$py_line" ]]; then
                py_time=$(echo "$py_line" | cut -d',' -f2)
                speedup=$(echo "scale=2; $r_time / $py_time" | bc 2>/dev/null || echo "N/A")
                printf "%-12s %12.2f %12.2f %12sx\n" "$n_covs" "$r_time" "$py_time" "$speedup"
            fi
        done
        echo ""
        echo "Note: Speedup = R time / Python time (higher is better for Python)"
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
