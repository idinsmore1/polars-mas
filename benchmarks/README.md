# Benchmarks

This directory contains benchmarks comparing `polars-mas` against the original R [PheWAS](https://github.com/PheWAS/PheWAS) package.

## Performance

### Sample Size Scaling

Benchmark conditions: 3 covariates, ~1,800 phecodes, 8 threads

![Sample Size Scaling](sample_size_scaling.png)

| Sample Size | polars-mas (s) | PheWAS (s) | Speedup |
|-------------|----------------|------------|---------|
| 5,000       | 19.6           | 85.4       | 4.4x    |
| 10,000      | 27.7           | 115.6      | 4.2x    |
| 50,000      | 111.4          | 347.7      | 3.1x    |
| 100,000     | 222.1          | 681.7      | 3.1x    |

### Covariate Scaling

Benchmark conditions: 5,000 samples, ~1,800 phecodes, 8 threads

![Covariate Scaling](covariate_scaling.png)

| Covariates | polars-mas (s) | PheWAS (s) | Speedup |
|------------|----------------|------------|---------|
| 1          | 23.9           | 59.2       | 2.5x    |
| 3          | 25.9           | 83.7       | 3.2x    |
| 5          | 30.1           | 116.4      | 3.9x    |
| 10         | 35.6           | 244.6      | 6.9x    |
| 15         | 49.7           | 479.9      | 9.7x    |
| 20         | 64.7           | 888.9      | 13.7x   |

As the number of covariates increases, `polars-mas` shows increasingly better performance relative to PheWAS, achieving up to **13.7x speedup** with 20 covariates.

## Memory Usage

### Memory by Sample Size

Benchmark conditions: 5 covariates, ~1,800 phecodes, 8 threads

![Memory Sample Scaling](memory_sample_scaling.png)

### Memory by Covariate Count

Benchmark conditions: 5,000 samples, ~1,800 phecodes, 8 threads

![Memory Covariate Scaling](memory_covariate_scaling.png)

## Numerical Agreement

To validate that `polars-mas` produces identical results to the original PheWAS package, we compared output metrics using Bland-Altman plots. All differences were computed and rounded to 8 decimal places.

### P-value Agreement

![P-value Bland-Altman](pval_bland_altman.png)

### Beta Coefficient Agreement

![Beta Bland-Altman](beta_bland_altman.png)

Both plots show all 1,801 phenotype comparisons lying exactly on the zero line, demonstrating that `polars-mas` and PheWAS produce results that agree within a tolerance of **1e-8**.

## Test Environment

All benchmarks were run on the following system:

| Component | Specification |
|-----------|---------------|
| CPU       | Intel Xeon Gold 6132 @ 2.60GHz (14 cores, 28 threads) |
| Memory    | 64 GB |
| OS        | Ubuntu 22.04.5 LTS |
| Kernel    | 6.8.0-90-generic |

## Reproducing Benchmarks

To reproduce these benchmarks:

```bash
# Generate example datasets
Rscript generate_examples.R

# Run sample size scaling benchmarks
./run_benchmarks.sh

# Run covariate scaling benchmarks
./run_covariate_scaling_benchmarks.sh

# Generate plots
jupyter notebook benchmark_plotting.ipynb
```
