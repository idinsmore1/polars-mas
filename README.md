# Polars-MAS: Multiple Association Studies

`polars-mas` is a python library and CLI tool meant to perform large scale multiple association tests, primarily seen in academic research. Currently this tool only supports Firth's logistic regression. Will run as a stand in replacement for PheWAS R package analysis, especially for Phecodes. `polars-mas` is built to leverage the speed and memory efficiency of the `polars` dataframe library and it's interoperability with the `sklearn` and `statsmodels` libraries.  

## Installation
```python
pip install polars-mas
```

## Running the CLI
```text
polars-mas --help

Polars-MAS: A Python package for multiple association analysis.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path.
  -o OUTPUT, --output OUTPUT
                        Output file prefix. Will be suffixed with '{predictor}.csv'.
  -p PREDICTORS [PREDICTORS ...], --predictors PREDICTORS [PREDICTORS ...]
                        Predictor column names. These will be tested independently
  -s SEPARATOR, --separator SEPARATOR
                        Column separator. Default is ","
  -d DEPENDENTS [DEPENDENTS ...], --dependents DEPENDENTS [DEPENDENTS ...]
                        Dependent variable column names.
  -di DEPENDENTS_INDICES, --dependents-indices DEPENDENTS_INDICES
                        Dependent variable column indicies. Ignored if --dependents is used. Accepts comma separated list of
                        indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid. Range follows python
                        slicing conventions - includes start, excludes end.
  -c COVARIATES [COVARIATES ...], --covariates COVARIATES [COVARIATES ...]
                        Covariate column names.
  -ci COVARIATES_INDICIES, --covariates-indicies COVARIATES_INDICIES
                        Covariate column indicies. Ignored if --covariates is used. Accepts comma separated list of
                        indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid. Range follows python
                        slicing conventions - includes start, excludes end.
  -cc CATEGORICAL_COVARIATES [CATEGORICAL_COVARIATES ...], --categorical-covariates CATEGORICAL_COVARIATES [CATEGORICAL_COVARIATES ...]
                        Categorical covariate column names.
  -nv NULL_VALUES [NULL_VALUES ...], --null-values NULL_VALUES [NULL_VALUES ...]
                        List of values to be treated as missing values. Default is None (normal polars option).
  -qt, --quantitative   Dependent variables are quantitative traits.
  -mi {drop,forward,backward,min,max,mean,zero,one}, --missing {drop,forward,backward,min,max,mean,zero,one}
                        Method to handle missing values in covariates and predictor variables. If not specified, rows with
                        missing values in the predictor and covariate columns will be dropped.
  -t {standard,min-max}, --transform {standard,min-max}
                        Transform continuous covariates/predictor variables. Default is no transformation.
  -mc MIN_CASES, --min-cases MIN_CASES
                        Minimum number of cases for each dependent variable. Only applied when not --quantitative. Default is
                        20.
  -m {firth,linear}, --model {firth,linear}
                        Type of model to fit. Default is firth logistic regression.
  --phewas              Input data uses Phecodes for dependent variables.
  --phewas-sex-col PHEWAS_SEX_COL
                        Sex covariate column name for PheWAS analysis. Default = 'sex'. Must be coded as male = 0 and female
                        = 1.
  -th THREADS, --threads THREADS
                        Number of threads for numpy and sklearn to use within each worker.
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers for parallel processing and threads available to Polars. Default is number of CPUs.
  -v, --verbose         have more verbose logging
```
If you have an R environment with the `PheWAS` package installed, you can run the `src/tests/example_data/generate_examples.R` script to create dummy data for this repository. 

**NOTE ON THREADS AND WORKERS**: The total number of threads used by `polars-mas` is the number of workers (`-n`) multiplied by the number of threads (`-th`). So if you have 4 workers with 8 threads, `polars-mas` will use 32 threads on your machine. 
