import polars as pl
import polars_mas.mas_frame as pla

from pathlib import Path


def run_mas(
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
    df = pl.scan_csv(input, separator=separator, null_values=null_values)
    selected_columns = predictors + covariates + dependents
    independents = predictors + covariates
    preprocessed = (
        df.select(selected_columns)
        # preprocessing methods
        .polars_mas.check_independents_for_constants(independents)
        .polars_mas.validate_dependents(dependents, quantitative, min_cases)
        .polars_mas.handle_missing_values(missing, covariates)
        .polars_mas.category_to_dummy(
            categorical_covariates, predictors, independents, covariates, dependents
        )
        .polars_mas.transform_continuous(transform, independents, categorical_covariates)
        # Make long format for dependent variables and remove missing values
        # .polars_mas.melt(predictors, independents, dependents)
        # .polars_mas.phewas_filter(kwargs["phewas"], kwargs["phewas_sex_col"], drop=True)
        .collect()
        .lazy()
    )
    assoc_kwargs = {
        "output_path": output,
        "independents": independents,
        "predictors": predictors,
        "covariates": covariates,
        "dependents": dependents,
        "quantitative": quantitative,
        "binary_model": binary_model,
        "linear_model": linear_model,
        "is_phewas": kwargs["phewas"],
    }
    output_df = preprocessed.polars_mas.run_associations(**assoc_kwargs)
    print(output_df)
    # output_df = preprocessed.polars_mas.run_associations_serial(**assoc_kwargs)
    for predictor in predictors:
        pred_df = output_df.filter(pl.col("predictor") == predictor)
        pred_df.write_csv(f"{output}_{predictor}.csv")
    # print(pred_df.head())
