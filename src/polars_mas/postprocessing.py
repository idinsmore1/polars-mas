import polars as pl
from polars_mas.config import MASConfig
from polars_mas.constants import phecode_defs


def postprocess(df: pl.DataFrame, config: MASConfig) -> pl.DataFrame:
    """Wrapper function for all postprocessing steps"""
    df = _calculate_corrected_pvalues(df)
    df = _add_phecode_definitions(df, config)
    _write_to_output(df, config)
    return df


def _calculate_corrected_pvalues(df: pl.DataFrame, method='bonferroni', baseline_pval=0.05) -> pl.DataFrame:
    if method == 'bonferroni':
        num_tests = df.filter(pl.col('pval').is_not_null()).select(pl.len()).item()
        threshold = baseline_pval / num_tests
        return df.with_columns(
            pl.col('pval').lt(threshold).alias('bonferroni_significant')
        )
    

def _add_phecode_definitions(df: pl.DataFrame, config: MASConfig) -> pl.DataFrame:
    if config.is_phewas:
        return df.join(phecode_defs, left_on='dependent', right_on='phecode', how='left')
    elif config.is_flipwas:
        return df.join(phecode_defs, left_on='predictor', right_on='phecode', how='left')
    else:
        return df
    

def _write_to_output(df: pl.DataFrame, config: MASConfig) -> None:
    if config.output_type == "parquet":
        df.write_parquet(f'{config.output}_polars_mas_results.parquet')
    elif config.output_type == "csv":
        df.write_csv(f'{config.output}_polars_mas_results.csv')
    elif config.output_type == "tsv":
        df.write_csv(f'{config.output}_polars_mas_results.tsv', sep='\t')
    elif config.output_type == "txt":
        df.write_csv(f'{config.output}_polars_mas_results.txt', sep='\t')