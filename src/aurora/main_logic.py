import polars as pl
import aurora.polars_aurora as pla

from pathlib import Path


def aurora(
        input: Path,
        output: Path, 
        separator: str, 
        predictor: str,
        dependents: list[str],
        covariates: list[str],
        categorical_covariates: list[str],
        null_values: list[str],
        frame_type: str,
        missing: str,
        quantitative: bool,
        standardize: bool,
        min_cases: int,
        **kwargs
    ) -> None:
    if frame_type == 'eager':
        reader = pl.read_csv
    elif frame_type == 'lazy':
        reader = pl.scan_csv
    df = reader(input, separator=separator, null_values=null_values)
    selected_columns = [predictor] + covariates + dependents
    predictors = [predictor] + covariates
    print(predictors, covariates, categorical_covariates)
    print(
        df
        .select(selected_columns)
        .aurora.check_for_constants()
        .aurora.handle_missing_values(missing, predictors)
        .aurora.category_to_dummy(
            categorical_covariates, 
            predictor,
            predictors, 
            covariates,
            dependents
        )
        .head().collect()
    )
    print(predictors, covariates, categorical_covariates)