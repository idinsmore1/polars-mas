import polars as pl
from loguru import logger
from polars_mas.config import MASConfig

def run_pipeline(config: MASConfig):
    data = config.read_data()
    print(data.head().collect())