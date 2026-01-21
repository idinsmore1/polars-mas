import numpy as np
import polars as pl
from functools import partial
from loguru import logger
from threadpoolctl import threadpool_limits
from polars_mas.config import MASConfig
from polars_mas.preprocessing import drop_constant_covariates
from polars_mas.models import firth_regression, logistic_regression, linear_regression
import time


def run_associations(lf: pl.LazyFrame, config: MASConfig) -> pl.DataFrame:
    """Run association analyses based on the configuration"""
    num_predictors = len(config.predictor_columns)
    num_dependents = len(config.dependent_columns)
    num_groups = num_predictors * num_dependents
    logger.info(
        f"Starting association analyses for {num_groups} groups ({num_predictors} predictor{'s' if num_predictors != 1 else ''} x {num_dependents} dependent{'s' if num_dependents != 1 else ''})."
    )
    if config.model == "firth":
        logger.info("Using Firth logistic regression model for analysis.")
    elif config.model == "logistic":
        logger.info("Using standard logistic regression model for analysis.")
    elif config.model == "linear":
        logger.info("Using linear regression model for analysis.")
    result_lazyframes = []
    for predictor in config.predictor_columns:
        for dependent in config.dependent_columns:
            logger.trace(f"Analyzing predictor '{predictor}' with dependent '{dependent}'.")
            result_lazyframe = _perform_analysis(lf, predictor, dependent, config)
            result_lazyframes.append(result_lazyframe)
    if not result_lazyframes:
        logger.error("No valid analyses were performed. Please check your configuration and data.")
        return pl.DataFrame()
    # Collect in batches with progress
    batch_size = min(100, max(10, num_groups // 10))
    all_results = []
    for i in range(0, len(result_lazyframes), batch_size):
        batch = result_lazyframes[i : i + batch_size]
        results = pl.collect_all(batch)
        all_results.extend(results)
        completed = min(i + batch_size, len(result_lazyframes))
        logger.success(f"Progress: {completed}/{num_groups} ({100 * completed // num_groups}%)")

    result_combined = pl.concat(
        [result for result in all_results], how="diagonal_relaxed"
    ).sort("pval")
    logger.success("Association analyses completed successfully!")
    return result_combined


def _perform_analysis(
    lf: pl.LazyFrame, predictor: str, dependent: str, config: MASConfig
) -> pl.LazyFrame:
    """Perform the actual analysis for a given predictor and dependent variable using optimized function"""
    # Select only the relevant columns and drop missing values in the predictor and dependent
    columns = [predictor, dependent, *config.covariate_columns]
    analysis_lf = lf.select(columns)
    # analysis_lf = _drop_constant_covariates(analysis_lf, config)
    polars_output_schema: pl.Struct = _get_schema(config)
    association_schema: dict = _get_schema(config, for_polars=False)
    result_lf = (
        analysis_lf
        .map_batches(
            lambda df: _run_single_association(df, predictor, dependent, config, association_schema),
            schema=pl.Schema(polars_output_schema)
            # returns_scalar=True,
            # return_dtype=polars_output_schema,
        )
    )
    return result_lf
    

def _run_single_association(df: pl.DataFrame, predictor: str, dependent: str, config: MASConfig, output_schema: dict) -> pl.DataFrame:
    """Run the specified association model on the given data structure"""
    model_funcs = {
        "firth": firth_regression,
        "logistic": logistic_regression,
        "linear": linear_regression,
    }
    reg_func = model_funcs.get(config.model, None)
    if reg_func is None:
        raise ValueError(f"Model '{config.model}' is not supported.")
    output_schema: dict = _validate_data_structure(df, predictor, dependent, config, output_schema)
    if output_schema.get("failed_reason", "nan") != "nan":
        return pl.DataFrame([output_schema], schema=list(output_schema.keys()), orient='row')
    df = df.drop_nulls([predictor, dependent])
    df = _drop_constant_covariates(df, config)
    col_names = df.schema.names()
    predictor = col_names[0]
    dependent = col_names[1]
    covariates = [col for col in col_names if col not in [predictor, dependent]]
    equation = f"{dependent} ~ {predictor} + {' + '.join(covariates)}"
    X = df.select([predictor, *covariates])
    y = df.get_column(dependent).to_numpy()
    with threadpool_limits(config.num_threads):
        try:
            results = reg_func(X, y)
            output_schema.update(
                {"predictor": predictor, "dependent": dependent, "equation": equation, **results}
            )
            # return output_schema
        except Exception as e:
            logger.error(
                f"Error in {config.model} regression for predictor '{predictor}' and dependent '{dependent}': {e}"
            )
            output_schema.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "equation": equation,
                    "failed_reason": str(e),
                }
            )
            # return output_schema
    # logger.info()
    return pl.DataFrame([output_schema], schema=list(output_schema.keys()), orient='row')


def _validate_data_structure(data: pl.DataFrame, predictor: str, dependent: str, config: MASConfig, output_schema: dict) -> dict:
    if data.height == 0:
        logger.error(
            f"No data available after dropping nulls for predictor '{predictor}' and dependent '{dependent}'."
        )
        output_schema.update(
            {
                "predictor": predictor,
                "dependent": dependent,
                "failed_reason": "No data after dropping nulls.",
            }
        )
        return output_schema
    # Do check on case counts for non-quantitative outcomes
    if not config.quantitative:
        is_viable, message, case_count, controls_count, total_n = _check_case_counts(
            data, dependent, config.min_case_count
        )
        if not is_viable:
            logger.debug(
                f"Skipping analysis for predictor '{predictor}' and dependent '{dependent}': {message}"
            )
            output_schema.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "failed_reason": message,
                }
            )
            return output_schema
        else:
            output_schema.update(
                {
                    "cases": case_count,
                    "controls": controls_count,
                    "total_n": total_n,
                }
            )
            return output_schema
    else:
        if data.height < config.min_case_count:
            logger.debug(
                f"Skipping analysis for predictor '{predictor}' and dependent '{dependent}': Not enough observations ({data.height})."
            )
            output_schema.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "failed_reason": f"Not enough observations ({data.height}).",
                }
            )
            return output_schema
        else:
            output_schema.update({"n_observations": data.height})
            return output_schema

def _check_case_counts(
    struct_dataframe: pl.DataFrame, dependent: str, min_case_count: int
) -> tuple[bool, str, int, int, int]:
    """Check if the case and control counts meet the minimum requirements"""
    n_rows = struct_dataframe.height
    case_count = struct_dataframe.select(pl.col(dependent).sum()).item()
    controls_count = n_rows - case_count
    if case_count < min_case_count:
        return (
            False,
            f"Insufficient case count ({case_count} cases).",
            case_count,
            controls_count,
            n_rows,
        )
    elif controls_count < min_case_count:
        return (
            False,
            f"Insufficient control count ({controls_count} controls).",
            case_count,
            controls_count,
            n_rows,
        )
    elif case_count == n_rows:
        return False, "All observations are cases.", case_count, controls_count, n_rows
    return True, "", case_count, controls_count, n_rows


def _drop_constant_covariates(df: pl.DataFrame, config: MASConfig) -> pl.DataFrame:
    """Drop covariate columns that are constant (no variance)"""
    unique_counts = df.select(pl.col(config.covariate_columns).n_unique()).to_dicts()
    constant_covariates = []
    for col, count in unique_counts[0].items():
        if count <= 1:
            constant_covariates.append(col)
    if constant_covariates:
        logger.debug(f"Dropping constant covariate columns: {', '.join(constant_covariates)}")
        cleaned_df = df.drop(constant_covariates)
        return cleaned_df
    else:
        return df


def _get_schema(config: MASConfig, for_polars=True):
    if config.model == "firth" or config.model == "logistic":
        if for_polars:
            return pl.Struct(
                {
                    "predictor": pl.Utf8,
                    "dependent": pl.Utf8,
                    "pval": pl.Float64,
                    "beta": pl.Float64,
                    "se": pl.Float64,
                    "OR": pl.Float64,
                    "ci_low": pl.Float64,
                    "ci_high": pl.Float64,
                    "cases": pl.Int64,
                    "controls": pl.Int64,
                    "total_n": pl.Int64,
                    "converged": pl.Boolean,
                    "failed_reason": pl.Utf8,
                    "equation": pl.Utf8,
                }
            )
        else:
            return {
                "predictor": "nan",
                "dependent": "nan",
                "pval": float("nan"),
                "beta": float("nan"),
                "se": float("nan"),
                "OR": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "cases": -9,
                "controls": -9,
                "total_n": -9,
                "converged": False,
                "failed_reason": "nan",
                "equation": "nan",
            }
    if config.model == "linear":
        if for_polars:
            return pl.Struct(
                {
                    "predictor": pl.Utf8,
                    "dependent": pl.Utf8,
                    "pval": pl.Float64,
                    "beta": pl.Float64,
                    "se": pl.Float64,
                    "ci_low": pl.Float64,
                    "ci_high": pl.Float64,
                    "n_observations": pl.Int64,
                    "converged": pl.Boolean,
                    "failed_reason": pl.Utf8,
                    "equation": pl.Utf8,
                }
            )
        else:
            return {
                "predictor": "nan",
                "dependent": "nan",
                "pval": float("nan"),
                "beta": float("nan"),
                "se": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "cases": -9,
                "converged": False,
                "failed_reason": "nan",
                "equation": "nan",
            }
