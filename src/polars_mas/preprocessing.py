import tempfile
import polars as pl
from loguru import logger
from polars_mas.config import MASConfig


def handle_missing_covariates(lf: pl.LazyFrame, config: MASConfig) -> pl.LazyFrame:
    """Handle missing covariate values based on the configuration"""
    if config.missing_covariate_values == "fail":
        missing_counts = lf.select(pl.col(config.covariate_columns).null_count()).collect().to_dicts()
        for col, count in missing_counts[0].items():
            if count == config.included_row_count:
                raise ValueError(f"All values are missing in covariate column '{col}'.")
            if count > 0:
                raise ValueError(
                    f"Missing values found in covariate column '{col}': {count} missing values."
                )
        return lf
    elif config.missing_covariate_values == "drop":
        original_row_count = config.included_row_count
        lf = lf.drop_nulls(subset=config.covariate_columns)
        config.update_row_and_column_counts(lf)
        if config.included_row_count < original_row_count:
            logger.warning(
                f"Dropped {original_row_count - config.included_row_count} rows due to missing covariate values."
            )
        return lf
    else:
        fill_value = {
            "forward": pl.col(config.covariate_columns).fill_null(strategy="forward"),
            "backward": pl.col(config.covariate_columns).fill_null(strategy="backward"),
            "min": pl.col(config.covariate_columns).fill_null(pl.col(config.covariate_columns).min()),
            "max": pl.col(config.covariate_columns).fill_null(pl.col(config.covariate_columns).max()),
            "mean": pl.col(config.covariate_columns).fill_null(pl.col(config.covariate_columns).mean()),
            "zero": 0,
            "one": 1,
        }.get(config.missing_covariate_values, None)
        if fill_value is not None:
            return lf.fill_null(fill_value)
    return lf


def limit_sex_specific(lf: pl.LazyFrame, config: MASConfig) -> pl.LazyFrame:
    """Limit data to a specific sex if specified in the configuration"""
    if not config.male_only and not config.female_only:
        return lf
    original_row_count = config.included_row_count
    if config.sex_col not in config.included_columns:
        raise ValueError(f"Sex column '{config.sex_col}' not found in included columns.")
    if config.male_only:
        logger.info("Limiting analysis to male samples only.")
        sex_specific = lf.filter(pl.col(config.sex_col) != config.female_code)
        config.update_row_and_column_counts(sex_specific)
    else:
        logger.info("Limiting analysis to female samples only.")
        sex_specific = lf.filter(pl.col(config.sex_col) == config.female_code)
        config.update_row_and_column_counts(sex_specific)
    if config.included_row_count <= original_row_count:
        logger.success(
            f"Dropped {original_row_count - config.included_row_count} rows for {'male' if config.male_only else 'female'} specific analysis."
        )
    return sex_specific


def drop_constant_covariates(lf: pl.LazyFrame, config: MASConfig) -> pl.LazyFrame:
    """Drop covariate columns that are constant (no variance)"""
    constant_covariates = []
    unique_counts = lf.select(pl.col(config.covariate_columns).n_unique()).collect().to_dicts()
    for col, count in unique_counts[0].items():
        if count <= 1:
            constant_covariates.append(col)
    if constant_covariates:
        logger.warning(f"Dropping constant covariate columns: {', '.join(constant_covariates)}")
        lf = lf.drop(constant_covariates)
        config.covariate_columns = [
            col for col in config.covariate_columns if col not in constant_covariates
        ]
        config.update_row_and_column_counts(lf)
    return lf


def create_dummy_covariates(lf: pl.LazyFrame, config: MASConfig) -> pl.LazyFrame:
    """Create dummy variables for categorical covariates"""
    # If there are no categorical covariates, return the original LazyFrame
    if not config.categorical_covariate_columns:
        return lf
    unique_counts = (
        lf.select(pl.col(config.categorical_covariate_columns).implode().list.unique())
        .unique()
        .collect()
        .to_dicts()
    )
    new_cols = []
    for col, values in unique_counts[0].items():
        if len(values) > 2:
            # Skip the first value to avoid multicollinearity
            new_cols.extend([f"{col}_{value}" for value in values[1:]])
            lf = lf.with_columns(
                [
                    pl.when(pl.col(col) == value).then(1).otherwise(0).alias(f"{col}_{value}")
                    for value in values[1:]
                ]
            ).drop(col)
    config.covariate_columns = [
        col for col in config.covariate_columns if col not in config.categorical_covariate_columns
    ] + new_cols
    config.update_row_and_column_counts(lf)
    return lf


def write_temp_ipc(lf: pl.LazyFrame, config: MASConfig) -> None:
    """Write preprocessed data to a temporary IPC file, preferring /dev/shm on Linux."""
    import sys
    import shutil
    import os

    ipc_dir = None
    if sys.platform == "linux" and os.path.exists("/dev/shm"):
        shm_usage = shutil.disk_usage("/dev/shm")
        # Use /dev/shm if at least 2GB free
        if shm_usage.free > 2 * 1024**3:
            ipc_dir = "/dev/shm"
            logger.info("Using /dev/shm for IPC file (shared memory).")
        else:
            logger.info("/dev/shm has insufficient space, falling back to temp directory.")

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".ipc", dir=ipc_dir)
    tf_name = tf.name
    tf.close()
    lf.sink_ipc(tf_name)
    config.ipc_file = tf_name
    logger.info(f"Temporary IPC file written to {config.ipc_file}")