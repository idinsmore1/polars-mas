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
            # Placeholder for actual analysis logic
            result_lazyframe = perform_analysis(lf, predictor, dependent, config)
            if result_lazyframe is not None:
                result_lazyframes.append(result_lazyframe)
            # Store or log results as needed
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
        [result.unnest("result") for result in all_results], how="diagonal_relaxed"
    ).sort("pval")
    logger.success("Association analyses completed successfully!")
    return result_combined


def perform_analysis(
    lf: pl.LazyFrame, predictor: str, dependent: str, config: MASConfig
) -> pl.LazyFrame:
    """Perform the actual analysis for a given predictor and dependent variable"""
    # Select only the relevant columns and drop missing values in the predictor and dependent
    columns = [predictor, dependent, *config.covariate_columns]
    analysis_lf = lf.select(columns)
    model_func = partial(_run_association, predictor=predictor, dependent=dependent, config=config)
    expected_schema = _get_schema(config)
    result_lf = analysis_lf.select(pl.struct(columns).alias("association_struct")).select(
        pl.col("association_struct")
        .map_batches(model_func, returns_scalar=True, return_dtype=expected_schema)
        .alias("result")
    )
    return result_lf


def _run_association(
    association_struct: pl.Struct, predictor: str, dependent: str, config: MASConfig
) -> dict:
    """Run the specified association model on the given data structure"""
    model_funcs = {
        "firth": firth_regression,
        "logistic": logistic_regression,
        "linear": linear_regression,
    }
    reg_func = model_funcs.get(config.model, None)
    if reg_func is None:
        raise ValueError(f"Model '{config.model}' is not supported.")

    output_struct = _get_schema(config, for_polars=False)
    # create a dataframe from the struct
    data = association_struct.struct.unnest()
    # drop null values in the predictor and dependent
    data = data.drop_nulls([predictor, dependent])
    # Check that there is enough data to run the model
    if data.height == 0:
        logger.error(
            f"No data available after dropping nulls for predictor '{predictor}' and dependent '{dependent}'."
        )
        output_struct.update(
            {
                "predictor": predictor,
                "dependent": dependent,
                "failed_reason": "No data after dropping nulls.",
            }
        )
        return output_struct
    # Do check on case counts for non-quantitative outcomes
    if not config.quantitative:
        is_viable, message, case_count, controls_count, total_n = _check_case_counts(
            data, dependent, config.min_case_count
        )
        if not is_viable:
            logger.debug(
                f"Skipping analysis for predictor '{predictor}' and dependent '{dependent}': {message}"
            )
            output_struct.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "failed_reason": message,
                }
            )
            return output_struct
        else:
            output_struct.update(
                {
                    "cases": case_count,
                    "controls": controls_count,
                    "total_n": total_n,
                }
            )
    else:
        if data.height < config.min_case_count:
            logger.debug(
                f"Skipping analysis for predictor '{predictor}' and dependent '{dependent}': Not enough observations ({data.height})."
            )
            output_struct.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "failed_reason": f"Not enough observations ({data.height}).",
                }
            )
            return output_struct
        else:
            output_struct.update({"n_observations": data.height})
    # Prepare the data for regression
    data = _drop_constant_covariates(data, config)
    col_names = data.collect_schema().names()
    # TODO Add transformations to variables
    predictor = col_names[0]
    dependent = col_names[1]
    covariates = [col for col in col_names if col not in [predictor, dependent]]
    equation = f"{dependent} ~ {predictor} + {' + '.join(covariates)}"
    X = data.select([predictor, *covariates])
    y = data.get_column(dependent).to_numpy()
    with threadpool_limits(config.num_threads):
        try:
            results = reg_func(X, y)
            output_struct.update(
                {"predictor": predictor, "dependent": dependent, "equation": equation, **results}
            )
            return output_struct
        except Exception as e:
            logger.error(
                f"Error in {config.model} regression for predictor '{predictor}' and dependent '{dependent}': {e}"
            )
            return output_struct.update(
                {
                    "predictor": predictor,
                    "dependent": dependent,
                    "equation": equation,
                    "failed_reason": str(e),
                }
            )


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


def _drop_constant_covariates(struct_dataframe: pl.DataFrame, config: MASConfig) -> pl.DataFrame:
    """Drop covariate columns that are constant (no variance)"""
    unique_counts = struct_dataframe.select(pl.col(config.covariate_columns).n_unique()).to_dicts()
    constant_covariates = []
    for col, count in unique_counts[0].items():
        if count <= 1:
            constant_covariates.append(col)
    if constant_covariates:
        logger.debug(f"Dropping constant covariate columns: {', '.join(constant_covariates)}")
        struct_dataframe = struct_dataframe.drop(constant_covariates)
    return struct_dataframe


def _get_schema(config: MASConfig, for_polars=True) -> pl.Struct | dict:
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
