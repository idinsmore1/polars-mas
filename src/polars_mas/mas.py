import time
import polars as pl

from functools import partial
from loguru import logger
from polars_mas.consts import male_specific_codes, female_specific_codes, phecode_defs
from polars_mas.model_funcs import run_association_test


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
        male_codes_in_df = args.col_names.intersection(male_specific_codes)
        female_codes_in_df = args.col_names.intersection(female_specific_codes)
        sex_codes = male_codes_in_df.union(female_codes_in_df)
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
            .collect()
        )
        # Counts post-filtering
        post_counts = (
            code_matched
            .select(sex_codes)
            .count()
            .transpose(
                include_header=True,
                header_name='phecode',
                column_names=['count']
            )
        )
        changed = (
            pre_counts
            .join(post_counts, on="phecode", how="inner", suffix="_post")
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
        
        return code_matched.lazy()

    def filter_to_sex_specific(self, args) -> pl.LazyFrame:
        if not args.male_only and not args.female_only:
            return self._df
        if args.male_only:
            logger.log("IMPORTANT", "Filtering to male-only data.")
            df = self._df.filter(pl.col(args.sex_col).eq(args.male_code)).drop(args.sex_col)
            args.predictors = args.predictors.difference({args.sex_col})
            args.covariates = args.covariates.difference({args.sex_col})
            args.dependents = args.dependents.difference({args.sex_col})
            args.categorical_covariates = args.categorical_covariates.difference({args.sex_col})
            args.independents = args.predictors.union(args.covariates)
            args.selected_columns = args.predictors.union(args.covariates).union(args.dependents)
        elif args.female_only:
            logger.log("IMPORTANT", "Filtering to male-only data.")
            df = self._df.filter(pl.col(args.sex_col).eq(args.female_code)).drop(args.sex_col)
            args.predictors = args.predictors.difference({args.sex_col})
            args.covariates = args.covariates.difference({args.sex_col})
            args.dependents = args.dependents.difference({args.sex_col})
            args.categorical_covariates = args.categorical_covariates.difference({args.sex_col})
            args.independents = args.predictors.union(args.covariates)
            args.selected_columns = args.predictors.union(args.covariates).union(args.dependents)
        return df


    def check_for_constants(self, args) -> pl.LazyFrame:
        """Check/remove constant columns from the DataFrame."""
        df = self._df
        const_cols = set(
            df
            .select(pl.col(args.selected_columns).drop_nulls().unique().len())
            .collect()
            .transpose(
                include_header=True,
                header_name='column',
                column_names=['unique_count']
            )
            .filter(pl.col("unique_count") == 1)
            .get_column("column")
            .to_list()
        )
        if const_cols:
            if not args.drop_constants:
                # logger.error(f"Columns {','.join(const_cols)} are constant. Please remove them from selection or use --drop-constants.")
                raise ValueError(f"Columns {','.join(const_cols)} are constant. Please remove them from selection or use --drop-constants.")
            logger.log("IMPORTANT", f"Columns {','.join(const_cols)} are constant. They will be dropped.")
            args.predictors = args.predictors.difference(const_cols)
            args.covariates = args.covariates.difference(const_cols)
            args.dependents = args.dependents.difference(const_cols)
            args.categorical_covariates = args.categorical_covariates.difference(const_cols)
            if any([x == set() for x in [args.predictors, args.dependents]]):
                raise ValueError("No predictors or dependents left after dropping constant columns.")
            args.independents = args.predictors.union(args.covariates)
            args.selected_columns = args.predictors.union(args.covariates).union(args.dependents)
            df = df.drop(pl.col(const_cols))
        return df

    def check_grouped_independents_for_constants(
        self, independents: list[str], dependent: str = None
    ) -> list[str]:
        const_cols = (
            self._df.select(pl.col(independents).drop_nulls().unique().len())
            # .collect()
            .transpose(include_header=True)
            .filter(pl.col("column_0") == 1)
            .select(pl.col("column"))
        )["column"].to_list()
        if const_cols:
            if len(const_cols) > 1:
                predicate = "are"
                plural = "s"
            else:
                predicate = "is"
                plural = ""
            logger.warning(
                f"Column{plural} {','.join(const_cols)} {predicate} constant{plural}. Dropping from {dependent} analysis."
            )
            non_consts = [col for col in independents if col not in const_cols]
            return non_consts
        return independents

    def handle_missing_covariates(self, args) -> pl.LazyFrame:
        """Handle missing covariates in the DataFrame."""
        # If method is not drop, just fill the missing values with the specified method
        if args.missing != "drop":
            logger.info(
                f"Filling missing values in columns {','.join(args.covariates)} with {args.missing} method."
            )
            return self._df.with_columns(pl.col(args.covariates).fill_null(strategy=args.missing))
        # If method is drop, drop rows with missing values in the specified independents
        new_df = self._df.drop_nulls(subset=args.covariates)
        new_height = new_df.select(pl.len()).collect().item()
        old_height = self._df.select(pl.len()).collect().item()
        if new_height == 0:
            raise ValueError("All rows have missing values in one of the specified covariates. Please check your data.")
        if new_height != old_height:
            logger.info(f"Dropped {old_height - new_height} rows with missing values.")
        return new_df
    
    def validate_dependents(self, args) -> pl.LazyFrame:
        """Validates the dependent variables."""
        if args.quantitative:
            valid_dependents = set(
                self._df
                .select(args.dependents)
                .count()
                .collect()
                .transpose(
                    include_header=True,
                    header_name='dependent',
                    column_names=['count']
                )
                .filter(pl.col("count") >= args.min_cases)
                .get_column("dependent")
                .to_list()
            )
            if valid_dependents != args.dependents:
                logger.warning(f"Dropping {len(args.dependents) - len(valid_dependents)} dependents with fewer than {args.min_cases} cases.")
                args.dependents = valid_dependents
            return self._df.with_columns(pl.col(args.dependents).cast(pl.Float64))
        
        not_binary = set(
            self._df
            .select(pl.col(args.dependents).unique().drop_nulls().n_unique())
            .collect()
            .transpose(
                include_header=True,
                header_name='dependent',
                column_names=['unique_count']
            )
            .filter(pl.col('unique_count').ne(2))
            .get_column('dependent')
            .to_list()
        )
        if not_binary:
            raise ValueError(f"Dependent variables {','.join(not_binary)} are not binary. Please check your data.")
        invalid_dependents = set(
            self._df
            .select(pl.col(args.dependents))
            .sum()
            .collect()
            .transpose(
                include_header=True,
                header_name='dependent',
                column_names=['cases']
            )
            .filter(pl.col("cases").lt(args.min_cases))
            .get_column("dependent")
            .to_list()
        )
        if invalid_dependents:
            logger.log("IMPORTANT", f"Dropping {len(invalid_dependents)} dependents with fewer than {args.min_cases} cases.")
            args.dependents = args.dependents.difference(invalid_dependents)
        logger.debug(f"Dependents after filtering: {args.dependents}")
        return self._df.with_columns(pl.col(args.dependents).cast(pl.UInt8))

    def validate_flipwas(self, args) -> pl.LazyFrame:
        """Validates the flipped PheWAS codes."""
        if not args.flipwas:
            return self._df
        not_binary = set(
            self._df
            .select(pl.col(args.predictors).unique().drop_nulls().n_unique())
            .collect()
            .transpose(
                include_header=True,
                header_name='predictor',
                column_names=['unique_count']
            )
            .filter(pl.col('unique_count').gt(2))
            .get_column('predictor')
            .to_list()
        )
        if not_binary:
            raise ValueError(f"Predictors variables {','.join(not_binary)} are not binary. Please check your data.")
        invalid_predictors = set(
            self._df
            .select(pl.col(args.predictors))
            .sum()
            .collect()
            .transpose(
                include_header=True,
                header_name='predictor',
                column_names=['cases']
            )
            .filter(pl.col("cases").lt(args.min_cases))
            .get_column("predictor")
            .to_list()
        )
        if invalid_predictors:
            logger.log("IMPORTANT", f"Dropping {len(invalid_predictors)} predictors with fewer than {args.min_cases} cases.")
            args.predictors = args.predictors.difference(invalid_predictors)
            logger.debug(f"Dropped predictors: {','.join(invalid_predictors)}")
        logger.debug(f"predictors after filtering: {args.predictors}")
        return self._df.with_columns(pl.col(args.predictors).cast(pl.UInt8))
    
    def category_to_dummy(self, args) -> pl.LazyFrame:
        """Converts categorical variables to dummy variables."""
        if not args.categorical_covariates:
            return self._df
        not_binary = set(
            self._df
            .select(pl.col(args.categorical_covariates).unique().drop_nulls().n_unique())
            .collect()
            .transpose(
                include_header=True,
                header_name='categorical_covariate',
                column_names=['unique_count']
            )
            .filter(pl.col('unique_count').gt(2))
            .get_column('categorical_covariate')
            .to_list()
        )
        if not_binary:
            logger.log("IMPORTANT", f"Categorical covariates {','.join(not_binary)} are not binary. Converting to dummy variables.")
            dummy = self._df.collect().to_dummies(not_binary, separator='_dum_', drop_first=True).lazy()
            dummy_cols = set(dummy.collect_schema().names())
            new_cols = {x for x in dummy_cols.difference(args.selected_columns) if '_dum_' in x}
            args.covariates = args.covariates.difference(not_binary).union(new_cols)
            args.categorical_covariates = args.categorical_covariates.difference(not_binary).union(new_cols)
            args.independents = args.predictors.union(args.covariates)
            args.selected_columns = args.predictors.union(args.covariates).union(args.dependents)
            return dummy
        return self._df

    def run_associations(self, args) -> pl.LazyFrame:
        num_groups = len(args.predictors) * len(args.dependents)
        logger.info(f'Running {num_groups} associations for {len(args.predictors)} predictors and {len(args.dependents)} dependents.')
        regression_function = partial(
            run_association_test,
            model_type=args.model,
            num_groups=num_groups,
            is_phewas=args.phewas,
            is_flipwas=args.flipwas,
            sex_col=args.sex_col
        )
        regression_df = self._df
        result_df = pl.DataFrame()
        if not args.flipwas:
            for predictor in args.predictors:
                res_list = []
                for dependent in args.dependents:
                    order = [predictor, *list(args.covariates), dependent]
                    lazy_df = (
                        regression_df
                        .select(
                            pl.col(order),
                            pl.struct(order).alias('model_struct')
                        )
                        .drop_nulls([predictor, dependent])
                        .select(
                            pl.col('model_struct')
                            .map_batches(regression_function, returns_scalar=True, return_dtype=pl.Struct)
                            .alias('result')
                        )
                    )
                    res_list.append(lazy_df)
                results = pl.collect_all(res_list)
                output = pl.concat([result.unnest('result') for result in results]).sort('pval')
                result_df = result_df.vstack(output)
        elif args.flipwas:
            for dependent in args.dependents:
                res_list = []
                for predictor in args.predictors:
                    order = [predictor, *list(args.covariates), dependent]
                    lazy_df = (
                        regression_df
                        .select(
                            pl.col(order),
                            pl.struct(order).alias('model_struct')
                        )
                        .drop_nulls([predictor, dependent])
                        .select(
                            pl.col('model_struct')
                            .map_batches(regression_function, returns_scalar=True, return_dtype=pl.Struct)
                            .alias('result')
                        )
                    )
                    res_list.append(lazy_df)
                results = pl.collect_all(res_list)
                output = pl.concat([result.unnest('result') for result in results]).sort('pval')
                result_df = result_df.vstack(output)
        # Add annotations to the results if it's a phewas.
        if args.phewas or args.flipwas:
            if args.flipwas:
                left_col = "predictor"
            else:
                left_col = "dependent"
            result_df = (
                result_df
                .join(phecode_defs, left_on=left_col, right_on='phecode')
                .sort(['predictor', 'pval'])
            )
        return result_df


def run_multiple_association_study(args) -> pl.LazyFrame:
    """
    Run the multiple association study.
    """
    # Check if the data is suitable for PheWAS analysis.
    start = time.perf_counter()
    df = pl.scan_csv(args.input, separator=args.separator, null_values=args.null_values)
    result = (
        df
        .mas.handle_missing_covariates(args)
        .mas.phewas_check(args)
        .mas.filter_to_sex_specific(args)
        .mas.check_for_constants(args)
        .mas.validate_dependents(args)
        .mas.validate_flipwas(args)
        .mas.category_to_dummy(args)
        .collect()
        .lazy()
        .mas.run_associations(args)
    )
    print(result)