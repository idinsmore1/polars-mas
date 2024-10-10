import polars as pl
from pathlib import Path

source_path = Path(__file__).parent.parent

phecode_defs = pl.scan_csv(source_path / 'data/phecode_definitions1.2.csv', schema_overrides={'phecode': pl.String})
male_specific_codes = phecode_defs.filter(pl.col('sex') == 'Male').collect()['phecode'].to_list()
female_specific_codes = phecode_defs.filter(pl.col('sex') == "Female").collect()['phecode'].to_list()