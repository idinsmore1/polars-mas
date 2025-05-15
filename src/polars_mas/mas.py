import time
import polars as pl

from functools import partial
from loguru import logger
from polars_mas.consts import male_specific_codes, female_specific_codes


@pl.api.register_dataframe_namespace('mas')
@pl.api.register_lazyframe_namespace('mas')
class MASFrame:
    """
    This class is a namespace for the polars_mas library. It allows us to register
    functions as methods of the DataFrame and LazyFrame classes.
    """

    def __init__(self, df):
        self._df = df

    def phewas_check(self, args) -> pl.LazyFrame:
        """
        Check if the data is suitable for PheWAS analysis.
        """
        if not args.phewas and not args.flipwas:
            # Not a phewas analysis.
            return self._df

        if args.sex_col not in args.col_names:
            logger.log("IMPORTANT", f"sex column {args.sex_col} not found in input file. Sex-specific PheWAS filtering will not be performed.")
            return self._df
        male_codes_in_df = [col for col in args.col_names if col in male_specific_codes]
        female_codes_in_df = [col for col in args.col_names if col in female_specific_codes]
        sex_codes = male_codes_in_df + female_codes_in_df
        if not male_codes_in_df and not female_codes_in_df:
            logger.log("IMPORTANT", "No sex-specific PheCodes found in input file. Returning all phecodes.")
            return self._df
        # Counts for each sex-specific code pre-filtering
        pre_counts = (
            self._df
            .select(sex_codes)
            .count()
            .collect()
            .transpose(
                include_header=True,
                header_name='phecode',
                column_names=['count']
            )
        )
        code_matched = (
            self._df
            .with_columns([
                pl.when(pl.col(args.sex_col).eq(args.female_code))
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
                for column in male_codes_in_df
            ])
            .with_columns([
                pl.when(pl.col(args.sex_col).eq(args.male_code))
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
                for column in female_codes_in_df
            ])
        )
        # Counts post-filtering
        post_counts = (
            code_matched
            .select(sex_codes)
            .count()
            .collect()
            .transpose(
                include_header=True,
                header_name='phecode',
                column_names=['count']
            )
        )
        changed = (
            pre_counts.join(post_counts, on="phecode", how="inner", suffix="_post")
            .filter(pl.col("count") != pl.col("count_post"))
            .get_column("phecode")
            .to_list()
        )
        if changed:
            logger.log(
                "IMPORTANT",
                f"{len(changed)} PheWAS sex-specific codes have mismatched sex values. See log file for details.",
            )
            logger.warning(
                f"Female specific codes with mismatched sex values: {[col for col in changed if col in female_codes_in_df]}"
            )
            logger.warning(
                f"Male specific codes with mismatched sex values: {[col for col in changed if col in male_codes_in_df]}"
            )
        
        return code_matched


    def check_independents_for_constants(self, args) -> pl.LazyFrame:
        pass

    def validate_dependents(self, args) -> pl.LazyFrame:
        pass


def run_multiple_association_study(args) -> pl.LazyFrame:
    """
    Run the multiple association study.
    """
    # Check if the data is suitable for PheWAS analysis.
    start = time.perf_counter()
    df = pl.scan_csv(args.input, separator=args.separator, null_values=args.null_values)
    df.mas.phewas_check(args)