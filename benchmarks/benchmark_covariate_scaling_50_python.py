#!/usr/bin/env python3
"""Benchmark script for polars-mas covariate scaling (up to 50 covariates).

Tests performance with increasing numbers of covariates (1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
using a 5000 sample subset from phewas_example_1e+05_samples_50_covariates.csv file.
"""

import time
import subprocess
import sys
import random
from pathlib import Path

import polars as pl

# Data file
DATA_FILE = Path(__file__).parent / "phewas_example_1e+05_samples_50_covariates.csv"

# Common parameters
PREDICTOR = "rsEXAMPLE"
NUM_WORKERS = 8
MODEL = "firth"
N_SAMPLES = 5000

# All available covariates in order (columns 1-50 in file, 0-indexed)
ALL_COVARIATES = [
    "sex", "age", "age2", "race_1", "race_2", "race_3", "bmi", "smoking_status",
    "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides",
    "ldl", "hdl", "creatinine", "uric_acid", "glucose", "on_insulin",
    "systolic_bp", "diastolic_bp", "heart_rate", "temperature", "respiratory_rate",
    "oxygen_saturation", "hemoglobin", "hematocrit", "platelet_count", "white_blood_cell_count",
    "albumin", "alt", "ast", "alkaline_phosphatase", "bilirubin",
    "sodium", "potassium", "chloride", "calcium", "magnesium",
    "phosphorus", "vitamin_d", "vitamin_b12", "folate", "iron",
    "ferritin", "tsh", "t4", "inr", "physical_activity",
]

# Covariate sets for testing (progressively larger)
COVARIATE_SETS = {
    1: ALL_COVARIATES[:1],
    3: ALL_COVARIATES[:3],
    5: ALL_COVARIATES[:5],
    10: ALL_COVARIATES[:10],
    15: ALL_COVARIATES[:15],
    20: ALL_COVARIATES[:20],
    25: ALL_COVARIATES[:25],
    30: ALL_COVARIATES[:30],
    35: ALL_COVARIATES[:35],
    40: ALL_COVARIATES[:40],
    45: ALL_COVARIATES[:45],
    50: ALL_COVARIATES[:50],
}


def ensure_sample_indices(n_total: int, max_samples: int, output_dir: Path) -> list[int]:
    """Ensure reproducible sample indices exist and return them."""
    indices_file = output_dir / "sample_indices.csv"

    if indices_file.exists():
        df = pl.read_csv(indices_file)
        return df["index"].to_list()

    # Generate new indices with fixed seed
    random.seed(42)
    indices = random.sample(range(1, n_total + 1), max_samples)  # 1-indexed for R compatibility

    # Save for reproducibility
    pl.DataFrame({"index": indices}).write_csv(indices_file)
    print(f"Generated sample indices saved to {indices_file}")

    return indices


def run_benchmark(subset_file: Path, covariates: list[str], n_covariates: int, output_dir: Path) -> dict:
    """Run polars-mas with specified covariates and return timing info."""
    covariates_str = ",".join(covariates)
    output_file = output_dir / f"benchmark_covariate_scaling_50_{n_covariates}_covs_results"

    cmd = [
        "uv",
        "run",
        "polars-mas",
        "-i", str(subset_file),
        "-o", str(output_file),
        "-p", PREDICTOR,
        "-d", "i:52-",  # All columns from index 52 onwards are phecodes (0-indexed)
        "-c", covariates_str,
        "-m", MODEL,
        "-n", "1",  # Index of first outcome column (1-indexed for display)
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
    print("Python polars-mas Covariate Scaling Benchmark (up to 50 covariates)")
    print("=" * 60)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Model: {MODEL}")
    print(f"Data file: {DATA_FILE.name}")
    print(f"Sample size: {N_SAMPLES}")
    print()

    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        sys.exit(1)

    # Load full data to get dimensions
    full_data = pl.read_csv(DATA_FILE)
    n_total = len(full_data)
    n_phecodes = len(full_data.columns) - 52  # Phecodes start at column 52 (0-indexed)

    print(f"Total samples available: {n_total}, PheCodes: {n_phecodes}")

    results = []
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Ensure reproducible sample indices
    max_samples = 50000
    sample_indices = ensure_sample_indices(n_total, max_samples, output_dir)

    # Get 5000 sample subset using pre-generated indices (convert to 0-indexed)
    current_indices = [i - 1 for i in sample_indices[:N_SAMPLES]]
    subset_data = full_data[current_indices]

    # Save subset to temp file
    subset_file = output_dir / f"temp_subset_{N_SAMPLES}_covariate_test.csv"
    subset_data.write_csv(subset_file)
    print(f"Using {len(subset_data)} samples for covariate scaling test\n")

    # Test each covariate set
    covariate_counts = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for n_covs in covariate_counts:
        covariates = COVARIATE_SETS[n_covs]
        result = run_benchmark(subset_file, covariates, n_covs, output_dir)
        if result:
            results.append(result)

    # Clean up temp file
    subset_file.unlink()

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
    output_file = output_dir / "benchmark_covariate_scaling_50_python_results.csv"
    with open(output_file, "w") as f:
        f.write("n_covariates,time_seconds,covariates\n")
        for r in results:
            f.write(f"{r['n_covariates']},{r['time_seconds']:.4f},\"{r['covariates']}\"\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
