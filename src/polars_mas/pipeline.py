import polars as pl
from loguru import logger
from polars_mas.config import MASConfig
from polars_mas.preprocessing import (
    handle_missing_covariates,
    limit_sex_specific,
    drop_constant_covariates
)
from polars_mas.analysis import run_associations

def run_pipeline(config: MASConfig):
    # Read in the data
    data = config.read_data()
    # Preprocessing steps
    logger.info("Starting preprocessing...")
    data = limit_sex_specific(data, config)
    data = handle_missing_covariates(data, config)
    data = drop_constant_covariates(data, config)
    logger.success("Preprocessing completed.")
    # Analysis steps
    results = run_associations(data, config)
    print(results)