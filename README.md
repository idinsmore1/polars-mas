# polars-mas

A fast Python library for multiple association studies, built on [Polars](https://pola.rs/).

`polars-mas` is designed as a drop-in replacement for the R [PheWAS](https://github.com/PheWAS/PheWAS) package, providing **3-14x speedup** while producing numerically identical results (within 1e-8 tolerance).

## Features

- **Fast**: Leverages Polars for efficient data processing with multi-threaded computation
- **Accurate**: Produces results identical to the R PheWAS package
- **Flexible**: Supports multiple regression models and input/output formats
- **PheCode-aware**: Built-in PheCode definitions and sex-specific code handling

### Supported Models

| Model | Use Case | Flag |
|-------|----------|------|
| Firth logistic regression | Binary outcomes (default) | `-m firth` |
| Standard logistic regression | Binary outcomes | `-m logistic` |
| Linear regression | Quantitative outcomes | `-m linear` |

### Supported Formats

- **Input**: Parquet, CSV, TSV, TXT (tab-delimited)
- **Output**: Parquet, CSV, TSV, TXT

## Installation

```bash
pip install polars-mas
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add polars-mas
```

## Quick Start

### Basic Usage

```bash
polars-mas \
  -i data.csv \
  -o results \
  -p exposure \
  -d i:10- \
  -c age,sex,bmi
```

This runs Firth logistic regression with:
- `exposure` as the predictor
- Columns 10 onwards as dependent variables (phecodes)
- `age`, `sex`, `bmi` as covariates

### PheWAS Analysis

```bash
polars-mas \
  -i phewas_data.parquet \
  -o phewas_results \
  -p genetic_variant \
  -d i:20-1850 \
  -c age,sex,pc1,pc2,pc3 \
  --phewas \
  -n 4 \
  -t 8
```

The `--phewas` flag enables automatic PheCode annotation in results.

## CLI Reference

```
polars-mas [OPTIONS]

Input Options:
  -i, --input PATH              Input file (parquet, csv, tsv, txt)
  -o, --output PATH             Output file prefix
  -p, --predictors COLS         Predictor columns (comma-separated)
  -d, --dependents COLS         Dependent columns (comma-separated)
  -c, --covariates COLS         Covariate columns (comma-separated)
  -cc, --categorical-covariates Categorical covariate columns
  -nv, --null-values VALUES     Values to treat as null (comma-separated)
  -ot, --output-type TYPE       Output format: parquet, csv, tsv, txt

Association Parameters:
  -m, --model MODEL             firth (default), logistic, or linear
  -mc, --min-case-count N       Minimum cases/controls required (default: 20)
  -mcv, --missing-covariate-values
                                How to handle missing covariates:
                                fail, drop, forward, backward, min, max, mean, zero, one
  -qt, --quantitative           Dependent variables are quantitative

Performance:
  -n, --num-workers N           Number of worker processes (default: 1)
  -t, --threads N               Threads per worker (default: 2)

PheCode Options:
  --phewas                      PheCodes are dependent variables
  --flipwas                     PheCodes are predictor variables
  --sex-col COL                 Sex column name (default: sex)
  --female-code N               Code for female (default: 1)
  --male-only                   Include only male samples
  --female-only                 Include only female samples

Other:
  --dry-run                     Show configuration without running
  -v, --verbose                 Enable verbose logging
  -q, --quiet                   Suppress most output
```

### Column Selection

Columns can be specified by name or index:

```bash
# By name in comma-separated list
-p age,sex,bmi

# By index (0-based)
-d i:10          # Column 10
-d i:10-20       # Columns 10-19
-d i:10-         # Column 10 to end

# Can be used in conjuction as well!
-c age,sex,i:8-12
```

## Output

Results include:

| Column | Description |
|--------|-------------|
| `predictor` | Predictor variable name |
| `dependent` | Dependent variable name |
| `pval` | P-value (LRT for Firth) |
| `beta` | Coefficient estimate |
| `se` | Standard error |
| `OR` | Odds ratio (logistic models) |
| `ci_low`, `ci_high` | 95% confidence interval |
| `cases`, `controls` | Sample counts (binary outcomes) |
| `converged` | Model convergence status |
| `bonferroni_significant` | Bonferroni-corrected significance |
| `phenotype`, `category` | PheCode annotations (if `--phewas`) |

## Performance

See the [benchmarks](benchmarks/README.md) for detailed comparisons against the R PheWAS package.

**Summary**: `polars-mas` achieves 3-14x speedup depending on the number of covariates, with identical numerical results.

## Current Limitations

The following features from the R PheWAS package are not yet implemented:

- **Multiple testing correction**: Only Bonferroni correction is available (FDR/BH planned)
- **Variable transformations**: RINT and log transformations are not yet functional
- **Covariate scaling**: Standard and min-max scaling not yet implemented
- **Parallel workers**: Currently works best with one polars worker; `-n` flag reserved for future use. More threads with the `-t` option is fine.
- **Python API**: Only CLI interface is currently documented; programmatic API in development

## Roadmap

Planned features for future releases:

- [ ] FDR (Benjamini-Hochberg) multiple testing correction
- [ ] Rank-based inverse normal transformation (RINT)
- [ ] Log transformation for dependent variables
- [ ] Covariate standardization options
- [ ] True parallel processing with multiple workers
- [ ] Python API with DataFrame input/output
- [ ] Manhattan and QQ plot generation
- [ ] ICD-to-PheCode mapping utilities

## Requirements

- Python >= 3.11
- polars >= 1.9.0
- firthmodels >= 0.4.0
- statsmodels >= 0.14.4

## License

MIT

## Citation

If you use `polars-mas` in your research, please cite:

```
polars-mas: A fast Python library for multiple association studies
https://github.com/PheWAS/polars-mas
```

## Related Projects

- [PheWAS R Package](https://github.com/PheWAS/PheWAS) - The original R implementation
- [Polars](https://pola.rs/) - The underlying DataFrame library
- [firthmodels](https://github.com/jzluo/firthmodels) - Firth logistic regression implementation
