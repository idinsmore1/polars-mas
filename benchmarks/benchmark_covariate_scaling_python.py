#!/usr/bin/env python3
"""Benchmark script for polars-mas covariate scaling.

Tests performance with increasing numbers of covariates (1, 3, 5, 10, 15, 20)
using the phewas_example_5000_samples_20_covariates.csv file.

This benchmark demonstrates how polars-mas handles larger numbers of covariates
more efficiently than the R PheWAS package.
"""

import time
import subprocess
import sys
from pathlib import Path

# Data file
DATA_FILE = Path(__file__).parent / "phewas_example_5000_samples_20_covariates.csv"

# Common parameters
PREDICTOR = "rsEXAMPLE"
NUM_WORKERS = 8
MODEL = "firth"

# Covariate sets for testing (progressively larger)
COVARIATE_SETS = {
    1: ["age"],
    3: ["age", "sex", "bmi"],
    5: ["age", "age2", "sex", "bmi", "smoking_status"],
    10: ["age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "hba1c"],
    15: ["age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides", "ldl"],
    20: ["age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides", "ldl", "hdl", "creatinine", "uric_acid", "glucose", "on_insulin"],
}


def count_file_info(file_path: Path) -> tuple[int, int]:
    """Count samples and phecodes in a file."""
    with open(file_path) as f:
        header = f.readline().strip().split(",")
        # Phecodes start at column index 22 (0-indexed), after rsEXAMPLE
        n_phecodes = len(header) - 22
        n_samples = sum(1 for _ in f)
    return n_samples, n_phecodes


def run_benchmark(file_path: Path, covariates: list[str], n_covariates: int, output_dir: Path) -> dict:
    """Run polars-mas with specified covariates and return timing info."""
    covariates_str = ",".join(covariates)
    output_file = output_dir / f"benchmark_covariate_scaling_{n_covariates}_covs_results"

    cmd = [
        "uv",
        "run",
        "polars-mas",
        "-i", str(file_path),
        "-o", str(output_file),
        "-p", PREDICTOR,
        "-d", "i:22-",  # All columns from index 22 onwards are phecodes
        "-c", covariates_str,
        "-m", MODEL,
        "-n", "27",  # Index of first outcome column (1-indexed for display)
        "-t", str(NUM_WORKERS),
        "--phewas",
        "-q",  # Quiet mode
    ]

    print(f"  Running with {n_covariates} covariate(s)...", end="", flush=True)
    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    if result.returncode != 0:
        print(f" FAILED")
        print(f"    Error: {result.stderr}")
        return None

    print(f" Done in {elapsed:.2f} seconds")

    return {
        "n_covariates": n_covariates,
        "covariates": covariates_str,
        "time_seconds": elapsed,
    }


def main():
    print("=" * 60)
    print("Python polars-mas Covariate Scaling Benchmark")
    print("=" * 60)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Model: {MODEL}")
    print(f"Data file: {DATA_FILE.name}")
    print()

    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        sys.exit(1)

    n_samples, n_phecodes = count_file_info(DATA_FILE)
    print(f"Samples: {n_samples}, PheCodes: {n_phecodes}")
    print()

    results = []
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Test each covariate set
    covariate_counts = [1, 3, 5, 10, 15, 20]

    for n_covs in covariate_counts:
        covariates = COVARIATE_SETS[n_covs]
        result = run_benchmark(DATA_FILE, covariates, n_covs, output_dir)
        if result:
            results.append(result)

    # Print summary
    print()
    print("=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    print(f"{'Covariates':>12} {'Time (s)':>12}")
    print("-" * 26)
    for r in results:
        print(f"{r['n_covariates']:>12} {r['time_seconds']:>12.2f}")

    # Save results to CSV
    output_file = output_dir / "benchmark_covariate_scaling_python_results.csv"
    with open(output_file, "w") as f:
        f.write("n_covariates,time_seconds,covariates\n")
        for r in results:
            f.write(f"{r['n_covariates']},{r['time_seconds']:.4f},\"{r['covariates']}\"\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
