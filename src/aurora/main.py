import polars as pl
import aurora.polars_aurora as pla

from pathlib import Path


def aurora(
    input: Path,
    output: Path,
    separator: str,
    predictors: list[str],
    dependents: list[str],
    covariates: list[str],
    categorical_covariates: list[str],
    null_values: list[str],
    frame_type: str,
    missing: str,
    quantitative: bool,
    transform: str,
    min_cases: int,
    linear_model: str,
    binary_model: str,
    **kwargs,
) -> None:
    if frame_type == "eager":
        reader = pl.read_csv
    elif frame_type == "lazy":
        reader = pl.scan_csv
    df = reader(input, separator=separator, null_values=null_values)
    selected_columns = predictors + covariates + dependents
    independents = predictors + covariates
    preprocessed = (
        df.select(selected_columns)
        # preprocessing methods
        .aurora.check_independents_for_constants(independents)
        .aurora.validate_dependents(dependents, quantitative)
        .aurora.handle_missing_values(missing, independents)
        .aurora.category_to_dummy(
            categorical_covariates, predictors, independents, covariates, dependents
        )
        .aurora.transform_continuous(transform, independents, categorical_covariates)
        # Make long format for dependent variables and remove missing values
        .aurora.melt(predictors, independents, dependents)
        .aurora.phewas_filter(kwargs["phewas"], kwargs["phewas_sex_col"], drop=True)
    )
    assoc_kwargs = {
        "output_file": output,
        "independents": independents,
        "quantitative": quantitative,
        "binary_model": binary_model,
        "linear_model": linear_model,
        "min_cases": min_cases,
        "is_phewas": kwargs["phewas"],
    }
    output = preprocessed.aurora.run_associations(**assoc_kwargs)
    print(output)
