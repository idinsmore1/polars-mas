import polars as pl

from functools import partial
from loguru import logger


@pl.api.register_dataframe_namespace('mas')
@pl.api.register_lazyframe_namespace('mas')
class MASFrame:
    """
    This class is a namespace for the polars_mas library. It allows us to register
    functions as methods of the DataFrame and LazyFrame classes.
    """

    def __init__(self, df):
        self._df = df

    def check_independents_for_constants(self, args) -> pl.LazyFrame:
        pass

    def validate_dependents(self, args) -> pl.LazyFrame:
        pass

