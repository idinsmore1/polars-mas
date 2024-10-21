import polars as pl
import polars_mas.mas_frame as pla
import numpy as np
from loguru import logger
from firthlogist import FirthLogisticRegression
from polars_mas.consts import sex_specific_codes


def polars_firth_regression(
    struct_col: pl.Struct, independents: list[str], dependent_values: str
) -> dict:
    """
    Perform Firth logistic regression on a given dataset.

    Parameters:
    struct_col (pl.Struct): A Polars Struct column containing the data.
    independents (list[str]): List of independent variable names.
    dependent_values (str): Name of the dependent variable.
    min_cases (int): Minimum number of cases required to perform the regression.

    Returns:
    dict: A dictionary containing the results of the regression, including p-value,
          beta coefficient, standard error, odds ratio, confidence intervals,
          number of cases, controls, total number of observations, and failure reason if any.
    """
    # Need to have the full struct to allow polars to output properly
    output_struct = {
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
    }
    regframe = struct_col.struct.unnest()
    dependent = regframe.select("dependent").unique().item()
    predictor = regframe.select("predictor").unique().item()
    if dependent in sex_specific_codes and 'sex' in independents:
        reg_independents = [col for col in independents if col != 'sex']
    else:
        reg_independents = independents.copy()
    X = regframe.select(reg_independents)
    non_consts = X.polars_mas.check_grouped_independents_for_constants(reg_independents, dependent)
    X = X.select(non_consts)
    if reg_independents[0] not in X.collect_schema().names():
        logger.warning(f"Predictor {predictor} was removed due to constant values. Skipping analysis.")
        output_struct.update(
            {
                "failed_reason": "Predictor removed due to constant values",
            }
        )
        return output_struct
    y = regframe.select(dependent_values).to_numpy().ravel()
    cases = y.sum().astype(int)
    total_counts = y.shape[0]
    controls = total_counts - cases
    output_struct.update(
        {
            "cases": cases,
            "controls": controls,
            "total_n": total_counts,
        }
    )
    try:
        # We are only interested in the first predictor for the association test
        fl = FirthLogisticRegression(max_iter=1000, test_vars=0)
        fl.fit(X, y)
        # input_vars = X.collect_schema().names()
        output_struct.update(
            {
                "pval": fl.pvals_[0],
                "beta": fl.coef_[0],
                "se": fl.bse_[0],
                "OR": np.e ** fl.coef_[0],
                "ci_low": fl.ci_[0][0],
                "ci_high": fl.ci_[0][1],
                # "input_vars": ",".join(input_vars),
            }
        )
        return output_struct
    except Exception as e:
        logger.error(f"Error in Firth regression for {dependent}: {e}")
        output_struct.update({"failed_reason": str(e)})
        return output_struct
