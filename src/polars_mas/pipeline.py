import os
import atexit
import polars as pl
from loguru import logger
from polars_mas.config import MASConfig
from polars_mas.preprocessing import (
    handle_missing_covariates,
    limit_sex_specific,
    drop_constant_covariates,
    create_dummy_covariates,
    write_temp_ipc,
)
from polars_mas.postprocessing import postprocess
from polars_mas.analysis import run_associations_ipc


def run_pipeline(config: MASConfig):
    config.setup_logger()
    data = config.read_data()

    # Preprocessing uses full Polars thread pool
    logger.info("Starting preprocessing...")
    data = limit_sex_specific(data, config)
    data = handle_missing_covariates(data, config)
    data = drop_constant_covariates(data, config)
    data = create_dummy_covariates(data, config)
    write_temp_ipc(data, config)
    atexit.register(_cleanup_ipc, config)
    logger.success("Preprocessing completed.")

    # Analysis with IPC + guaranteed cleanup
    try:
        results = run_associations_ipc(config)
        results = postprocess(results, config)
        print(results)
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user.")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        _cleanup_ipc(config)


def _cleanup_ipc(config: MASConfig) -> None:
    """Remove the temporary IPC file if it exists."""
    if config.ipc_file and os.path.exists(config.ipc_file):
        try:
            os.unlink(config.ipc_file)
            logger.debug(f"Cleaned up IPC file: {config.ipc_file}")
        except OSError as e:
            logger.warning(f"Failed to clean up IPC file {config.ipc_file}: {e}")
