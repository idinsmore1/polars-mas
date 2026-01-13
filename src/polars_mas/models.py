import polars as pl
import numpy as np
import statsmodels.api as sm
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


def logistic_regression(X: pl.DataFrame, y: np.ndarray) -> dict:
    """Run standard logistic regression on the given data using statsmodels"""
    X = sm.add_constant(X.to_numpy(), prepend=False)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    return {
        "pval": result.pvalues[0],
        "beta": result.params[0],
        "se": result.bse[0],
        "OR": np.e ** result.params[0],
        "ci_low": result.conf_int()[0][0],
        "ci_high": result.conf_int()[0][1],
    }


def linear_regression(X: pl.DataFrame, y: np.ndarray) -> dict:
    X = sm.add_constant(X.to_numpy(), prepend=False)
    model = sm.OLS(y, X)
    result = model.fit()
    return {
        "pval": result.pvalues[0],
        "beta": result.params[0],
        "se": result.bse[0],
        "ci_low": result.conf_int()[0][0],
        "ci_high": result.conf_int()[0][1],
    }