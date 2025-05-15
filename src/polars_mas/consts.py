import polars as pl
from pathlib import Path

lib_path = Path(__file__).parent

phecode_defs = (
    pl.scan_csv(
        lib_path / "data/phecode_definitions1.2.csv",
        schema_overrides={"phecode": pl.String},
    )
    .select(["phecode", "phenotype", "sex", "category", "category_number"])
    .collect()
)
male_specific_codes = set(phecode_defs.filter(pl.col("sex") == "Male")["phecode"].to_list())
female_specific_codes = set(phecode_defs.filter(pl.col("sex") == "Female")["phecode"].to_list())
sex_specific_codes = male_specific_codes.union(female_specific_codes)
