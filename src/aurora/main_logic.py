import polars as pl
from pathlib import Path

def aurora(
        input: Path,
        output: Path, 
        separator: str, 
        predictor: str,
        dependents: list[str],
        covariates: list[str],
        categorical_covariates: list[str],
        frame_type: str,
        missing: str,
        quantitative: bool,
        standardize: bool,
        min_cases: int,
        **kwargs
    ) -> None:
    if frame_type == 'eager':
        df = pl.read_csv(input, separator=separator)
    elif frame_type == 'lazy':
        df = pl.scan_csv(input, separator=separator)
    selected_columns = [predictor] + covariates + dependents
    predictor_columns = [predictor] + covariates
    print(df.head().collect())