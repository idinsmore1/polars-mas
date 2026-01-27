import os
import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from threadpoolctl import threadpool_limits
from polars_mas.config import MASConfig
from polars_mas.models import firth_regression, logistic_regression, linear_regression


def run_associations_ipc(config: MASConfig) -> pl.DataFrame:
    """Run all association analyses in parallel using IPC memory-mapped files."""
    targets = []
    for predictor in config.predictor_columns:
        for dependent in config.dependent_columns:
            targets.append((predictor, dependent))

    num_groups = len(targets)
    logger.info(
        f"Starting association analyses for {num_groups} groups "
        f"({len(config.predictor_columns)} predictor(s) x {len(config.dependent_columns)} dependent(s))."
    )

    # Set POLARS_MAX_THREADS for child processes (loky uses spawn)
    original_env = os.environ.get("POLARS_MAX_THREADS")
    os.environ["POLARS_MAX_THREADS"] = str(config.num_threads)
    try:
        results = Parallel(n_jobs=config.num_workers, verbose=0, backend="loky")(
            delayed(_perform_analysis_ipc)(predictor, dependent, config, i, num_groups)
            for i, (predictor, dependent) in enumerate(targets, 1)
        )
    finally:
        if original_env is None:
            os.environ.pop("POLARS_MAX_THREADS", None)
        else:
            os.environ["POLARS_MAX_THREADS"] = original_env

    result_combined = pl.concat(results, how="diagonal_relaxed").sort("pval")
    logger.success("All analyses complete!")
    return result_combined


def _perform_analysis_ipc(
    predictor: str, dependent: str, config: MASConfig, task_num: int, total_tasks: int
) -> pl.DataFrame:
    """
    Run analysis for a single predictor-dependent pair on the IPC file.
    All data operations happen inside this job.
    """
    config.setup_logger()
    with threadpool_limits(config.num_threads):
        schema = _get_schema(config)

        # Load only needed columns via memory-mapped IPC
        df = (
            pl.scan_ipc(config.ipc_file, memory_map=True)
            .select([predictor, dependent, *config.covariate_columns])
            .drop_nulls([predictor, dependent])
            .collect()
        )

        # Validate data structure
        schema = _validate_data_structure(df, predictor, dependent, config, schema)
        if schema.get("failed_reason", "nan") != "nan":
            return pl.DataFrame([schema], schema=list(schema.keys()), orient="row")

        # Drop constant covariates on this subset
        df = _drop_constant_covariates(df, config)

        # Build regression inputs
        col_names = df.schema.names()
        pred_col = col_names[0]
        dep_col = col_names[1]
        covariates = [c for c in col_names if c not in [pred_col, dep_col]]
        equation = f"{dep_col} ~ {pred_col} + {' + '.join(covariates)}"

        X = df.select([pred_col, *covariates])
        y = df.get_column(dep_col).to_numpy()

        model_funcs = {
            "firth": firth_regression,
            "logistic": logistic_regression,
            "linear": linear_regression,
        }
        reg_func = model_funcs[config.model]

        try:
            results = reg_func(X, y)
            schema.update(
                {"predictor": pred_col, "dependent": dep_col, "equation": equation, **results}
            )
        except Exception as e:
            logger.error(
                f"Error in {config.model} regression for predictor '{pred_col}' and dependent '{dep_col}': {e}"
            )
            schema.update(
                {
                    "predictor": pred_col,
                    "dependent": dep_col,
                    "equation": equation,
                    "failed_reason": str(e),
                }
            )
    
    log_interval = _get_log_interval(total_tasks)
    if task_num % log_interval == 0 or task_num == total_tasks:
        logger.info(f"Progress: {task_num}/{total_tasks} ({100 * task_num // total_tasks}%)")
    return pl.DataFrame([schema], schema=list(schema.keys()), orient="row")


def _validate_data_structure(
    data: pl.DataFrame, predictor: str, dependent: str, config: MASConfig, output_schema: dict
) -> dict:
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
    covariate_cols = [c for c in config.covariate_columns if c in df.columns]
    if not covariate_cols:
        return df
    unique_counts = df.select(pl.col(covariate_cols).n_unique()).to_dicts()
    constant_covariates = [col for col, count in unique_counts[0].items() if count <= 1]
    if constant_covariates:
        logger.debug(f"Dropping constant covariate columns: {', '.join(constant_covariates)}")
        return df.drop(constant_covariates)
    return df


def _get_log_interval(total: int) -> int:
    """Return how often to log progress based on total task count."""
    if total <= 10:
        return 1
    if total <= 50:
        return 5
    if total <= 100:
        return 10
    if total <= 200:
        return 20
    if total <= 300:
        return 30
    if total <= 400:
        return 40
    if total <= 500:
        return 50
    return 100


def _get_schema(config: MASConfig) -> dict:
    """Get the default output schema dict for the given model type."""
    if config.model in ("firth", "logistic"):
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
        return {
            "predictor": "nan",
            "dependent": "nan",
            "pval": float("nan"),
            "beta": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_observations": -9,
            "converged": False,
            "failed_reason": "nan",
            "equation": "nan",
        }
