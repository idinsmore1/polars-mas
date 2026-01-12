import polars as pl
from loguru import logger
from polars_mas.config import MASConfig
from polars_mas.preprocessing import (
    handle_missing_covariates,
    limit_sex_specific,
    drop_constant_covariates
)

def run_pipeline(config: MASConfig):
    data = config.read_data()
    print(data.head().collect())
    # Preprocessing steps
    logger.info("Starting preprocessing...")
    data = limit_sex_specific(data, config)
    data = handle_missing_covariates(data, config)
    data = drop_constant_covariates(data, config)
    print(data.head().collect())