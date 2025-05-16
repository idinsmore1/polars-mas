import time
import polars as pl
import polars_mas.mas
import numpy as np
import statsmodels.api as sm
from loguru import logger
from firthlogist import FirthLogisticRegression
from polars_mas.consts import sex_specific_codes

num_completed = 0
time_per_assoc = 0
time_per_block = 0
prev_time = None
NUM_GROUPS = 0


def _update_progress() -> None:
    global num_completed
    global time_per_assoc
    global time_per_block
    global prev_time
    global NUM_GROUPS
    block = 25

    num_completed += 1
    if prev_time is None:
        prev_time = time.perf_counter()
    now = time.perf_counter()
    elapsed_time = now - prev_time
    # print(elapsed_time)
    time_per_assoc += elapsed_time
    time_per_block += elapsed_time
    prev_time = now
    if num_completed % block == 0 or num_completed == NUM_GROUPS:
        cpu_time_per_block = time_per_block
        time_per_block = 0
        logger.log("PROGRESS", f"Completed: [{num_completed}/{NUM_GROUPS}] - {cpu_time_per_block:.2f}s")


def run_association_test(
    model_struct: pl.Struct, model_type: str, num_groups: int, is_phewas: bool, is_flipwas: bool, sex_col: str
) -> dict:
    global NUM_GROUPS
    NUM_GROUPS = num_groups
    output_struct = {
        "predictor": "nan",
        "dependent": "nan",
        "pval": float("nan"),
        "beta": float("nan"),
        "se": float("nan"),
        "OR": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "cases": float("nan"),
        "controls": float("nan"),
        "total_n": float("nan"),
        "failed_reason": "nan",
        "equation": "nan",
    }
    if model_type == "linear":
        for key in ["OR", "ci_low", "ci_high", "cases", "controls"]:
            del output_struct[key]
        reg_func = linear_regression
    elif model_type == "logistic":
        # reg_func = logistic_regression
        raise ValueError("Logistic regression not yet implemented.")
    elif model_type == "firth":
        reg_func = firth_regression
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    # These happen every time
    df = model_struct.struct.unnest()
    col_names = df.collect_schema().names()
    predictor = col_names[0]
    dependent = col_names[-1]
    covariates = [col for col in col_names if col not in [predictor, dependent]]
    equation = f"{dependent} ~ {predictor} + {' + '.join(covariates)}"
    output_struct.update({"predictor": predictor, "dependent": dependent, "equation": equation})
    # df = df.drop_nulls([predictor, dependent])
    x = df.select([predictor, *covariates])
    y = df.get_column(dependent).to_numpy()
    # If this is a PheWAS analysis, sex is a covariate, and the dependent is sex-specific, remove it from the analysis
    if is_phewas:
        if sex_col in covariates and dependent in sex_specific_codes:
            # Remove sex column from covariates and x
            covariates = [col for col in covariates if col != sex_col]
            x = x.select(pl.col([predictor, *covariates]))
    if is_flipwas:
        if sex_col in covariates and predictor in sex_specific_codes:
            # Remove
            covariates = [col for col in covariates if col != sex_col]
            x = x.select(pl.col([predictor, *covariates]))
    non_consts = x.mas.check_grouped_independents_for_constants(
        [predictor, *covariates], dependent
    )
    x = x.select(non_consts)
    if predictor not in x.collect_schema().names():
        logger.warning(f"Predictor {predictor} was removed due to constant values. Skipping analysis.")
        output_struct.update(
            {
                "failed_reason": "Predictor removed due to constant values",
            }
        )
        _update_progress()
        return output_struct
    try:
        results = reg_func(x, y)
        output_struct.update(results)
    except Exception as e:
        logger.error(f"Error in {model_type} regression for {dependent}: {e}")
        output_struct.update({"failed_reason": str(e)})
    _update_progress()
    return output_struct


def firth_regression(x: pl.DataFrame, y: np.ndarray) -> dict:
    """Run Firth regression on the given data.
    Parameters
    ----------
    x : polars.DataFrame
        The data to use for the regression.
    y : np.ndarray
        The dependent variable.
    Returns
    -------
    dict
        The results of the regression.
    """
    cases, controls, total_counts = _get_counts(y)
    fl = FirthLogisticRegression(max_iter=1000, test_vars=0)
    fl.fit(x, y)
    return {
        "cases": cases,
        "controls": controls,
        "total_n": total_counts,
        "pval": fl.pvals_[0],
        "beta": fl.coef_[0],
        "se": fl.bse_[0],
        "OR": np.e ** fl.coef_[0],
        "ci_low": fl.ci_[0][0],
        "ci_high": fl.ci_[0][1],
    }


def logistic_regression(x: pl.DataFrame, y: np.ndarray) -> dict:
    """
    Run logistic regression on the given data.

    Parameters
    ----------
    x : polars.DataFrame
        The independent variables.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    dict
        The results of the regression, including total number of observations,
        p-value, beta coefficient, and standard error.
    """
    cases, controls, total_counts = _get_counts(y)
    x = sm.add_constant(x, prepend=False)
    model = sm.Logit(y, x).fit(maxiter=1000)
    return {
        "cases": cases,
        "controls": controls,
        "total_n": total_counts,
        "pval": model.pvalues[0],
        "beta": model.params[0],
        "se": model.bse[0],
    }


def linear_regression(x: pl.DataFrame, y: np.ndarray) -> dict:
    """
    Run linear regression on the given data.

    Parameters
    ----------
    x : polars.DataFrame
        The independent variables.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    dict
        The results of the regression, including total number of observations,
        p-value, beta coefficient, and standard error.
    """
    total_counts = y.shape[0]
    x = sm.add_constant(x, prepend=False)
    # print(x)
    model = sm.OLS(y, x).fit(maxiter=1000)
    return {
        "total_n": total_counts,
        "pval": model.pvalues[0],
        "beta": model.params[0],
        "se": model.bse[0],
    }


def _get_counts(y):
    """
    Calculate the number of cases and controls from the dependent variable.

    Parameters
    ----------
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    tuple
        A tuple containing the number of cases, controls, and total counts.
    """
    cases = y.sum().astype(int)
    total_counts = y.shape[0]
    controls = total_counts - cases
    return cases, controls, total_counts
