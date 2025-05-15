import datetime
import time

import polars as pl
import polars_mas.mas_frame as pla  # This sets up the polars_mas namespace

from pathlib import Path
from loguru import logger


def run_mas(
    input: Path,
    output: Path,
    separator: str,
    predictors: list[str],
    dependents: list[str],
    covariates: list[str],
    categorical_covariates: list[str],
    null_values: list[str],
    missing: str,
    quantitative: bool,
    transform: str,
    min_cases: int,
    model: str,
    **kwargs,
) -> None:
    start = time.perf_counter()
    selected_columns = predictors + covariates + dependents
    independents = predictors + covariates
    # Preprocess the data, will update the lists of predictors, covariates, and dependents in place
    df = pl.scan_csv(input, separator=separator, null_values=null_values)
    preprocessed, independents, predictors, covariates, dependents = df.polars_mas.preprocess(
        selected_columns,
        independents,
        predictors,
        covariates,
        dependents,
        categorical_covariates,
        missing,
        quantitative,
        transform,
        min_cases,
        is_phewas=kwargs["phewas"],
        phewas_sex_col=kwargs["phewas_sex_col"],
    )
    assoc_kwargs = {
        "predictors": predictors,
        "covariates": covariates,
        "dependents": dependents,
        "model": model,
        "is_phewas": kwargs["phewas"],
        "sex_col": kwargs["phewas_sex_col"],
        "flipwas": kwargs["flipwas"],
    }
    output_df = preprocessed.polars_mas.run_associations(**assoc_kwargs)
    for predictor in predictors:
        if kwargs["flipwas"]:
            filter_col = "dependent"
        else:
            filter_col = "predictor"
        pred_df = output_df.filter(pl.col(filter_col) == predictor)
        pred_df.write_csv(f"{output}_{predictor}.csv")
    end = time.perf_counter()
    logger.success(
        f"Associations Complete! Runtime: {str(datetime.timedelta(seconds=(round(end - start))))}"
    )
