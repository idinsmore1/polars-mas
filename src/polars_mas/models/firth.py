import polars as pl
import numpy as np
from firthmodels import FirthLogisticRegression

def firth_regression(X: pl.DataFrame, y: np.ndarray) -> dict:
    """Run Firth regression on the given data.

    Parameters
    ----------
    X : polars.DataFrame
        The data to use for the regression.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    dict
        The results of the regression.
    """
    fl = FirthLogisticRegression()
    fl.fit(X, y)
    return {
        "pval": fl.pvalues_[0],
        "beta": fl.coef_[0],
        "se": fl.bse_[0],
        "OR": np.e ** fl.coef_[0],
        "ci_low": fl.conf_int()[0][0],
        "ci_high": fl.conf_int()[0][1],
    }