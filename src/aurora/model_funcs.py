import polars as pl
import aurora.polars_aurora as pla
import numpy as np
from loguru import logger
from firthlogist import FirthLogisticRegression


def polars_firth_regression(
    struct_col: pl.Struct, independents: list[str], dependent_values: str, min_cases: int
) -> dict:
    output_struct = {
        "pval": float('nan'),
        "beta": float('nan'),
        "se": float('nan'),
        "OR": float('nan'),
        "ci_low": float('nan'),
        "ci_high": float('nan'),
        "cases": float('nan'),
        "controls": float('nan'),
        "total_n": float('nan'),
        "failed_reason": 'nan',
    }
    regframe = struct_col.struct.unnest()
    phenotype = regframe.select("dependent").unique().item()
    X = regframe.select(independents).aurora.check_independents_for_constants(independents, drop=True)
    x_cols = X.collect_schema().names()
    if independents[0] not in x_cols:
        logger.warning(
            f"Predictor {independents[0]} was removed due to constant values. Skipping analysis."
        )
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
    if cases < min_cases or controls < min_cases:
        logger.warning(
            f"Too few cases or controls for {phenotype}: {cases} cases - {controls} controls. Skipping analysis."
        )
        output_struct.update(
            {
                "failed_reason": "Too few cases or controls",
            }
        )
        return output_struct
    try:
        # We are only interested in the first predictor for the association test
        fl = FirthLogisticRegression(max_iter=1000, test_vars=0)
        fl.fit(X, y)
        output_struct.update(
            {
                "pval": fl.pvals_[0],
                "beta": fl.coef_[0],
                "se": fl.bse_[0],
                "OR": np.e ** fl.coef_[0],
                "ci_low": fl.ci_[0][0],
                "ci_high": fl.ci_[0][1],
            }
        )
        return output_struct
    except Exception as e:
        logger.error(f"Error in Firth regression for {phenotype}: {e}")
        output_struct.update({"failed_reason": str(e)})
        return output_struct
