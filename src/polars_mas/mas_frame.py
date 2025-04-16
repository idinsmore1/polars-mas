import polars as pl

from functools import partial

from loguru import logger
from polars_mas.consts import male_specific_codes, female_specific_codes, phecode_defs
from polars_mas.model_funcs import run_association_test


@pl.api.register_dataframe_namespace("polars_mas")
@pl.api.register_lazyframe_namespace("polars_mas")
class MASFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df

    def preprocess(
        self,
        selected_columns: list[str],
        independents: list[str],
        predictors: list[str],
        covariates: list[str],
        dependents: list[str],
        categorical_covariates: list[str],
        missing: str,
        quantitative: bool,
        transform: str,
        min_cases: int,
        is_phewas: bool,
        phewas_sex_col: str,
    ) -> tuple[pl.LazyFrame, list[str], list[str], list[str], list[str]]:
        preprocessed = (
            self._df.polars_mas.phewas_check(is_phewas, phewas_sex_col)
            .select(selected_columns)
            .polars_mas.check_independents_for_constants(independents)
            .polars_mas.validate_dependents(dependents, quantitative, min_cases)
            .polars_mas.handle_missing_values(missing, covariates)
            .polars_mas.category_to_dummy(
                categorical_covariates, predictors, independents, covariates, dependents
            )
            .polars_mas.transform_continuous(transform, independents, categorical_covariates)
            .collect()
            .lazy()
        )
        return preprocessed, independents, predictors, covariates, dependents

    def check_independents_for_constants(self, independents: list[str], drop=False) -> pl.LazyFrame:
        """
        Check for constant columns in the given independents and optionally drop them.

        This method checks if any of the specified predictor columns in the DataFrame
        have constant values (i.e., all values in the column are the same). If such
        columns are found, it either raises an error or drops them based on the `drop`
        parameter.

        Args:
            independents (list[str]): List of predictor column names to check for constants.
            drop (bool, optional): If True, constant columns will be dropped from the DataFrame.
                                   If False, an error will be raised if constant columns are found.
                                   Defaults to False.
            dependent str: dependent variable being tested (useful when running regression for logging).
                           adds a log if columns are dropped.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame with constant columns dropped if `drop` is True,
                                         otherwise the original DataFrame.

        Raises:
            ValueError: If constant columns are found and `drop` is False.
        """
        df = self._df
        const_cols = (
            df.select(pl.col(independents).drop_nulls().unique().len())
            .collect()
            .transpose(include_header=True)
            .filter(pl.col("column_0") == 1)
            .select(pl.col("column"))
        )["column"].to_list()
        if const_cols:
            if not drop:
                logger.error(
                    f'Columns {",".join(const_cols)} are constants. Please remove from analysis or set drop=True.'
                )
                raise ValueError
            logger.log("IMPORTANT", f'Dropping constant columns {",".join(const_cols)}')
            new_independents = [col for col in independents if col not in const_cols]
            independents.clear()
            independents.extend(new_independents)
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
                f'Column{plural} {",".join(const_cols)} {predicate} constant{plural}. Dropping from {dependent} analysis.'
            )
            non_consts = [col for col in independents if col not in const_cols]
            return non_consts
        return independents

    def validate_dependents(
        self, dependents: list[str], quantitative: bool, min_cases: int
    ) -> pl.LazyFrame:
        """
        Validates and casts the dependent variables in the DataFrame.

        Parameters:
        dependents (list[str]): List of dependent variable column names.
        quantitative (bool): Flag indicating if the dependent variables are quantitative.

        Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame with the dependent variables cast to the appropriate type.

        Raises:
        ValueError: If any of the dependent variables are not binary when quantitative is False.
        """
        # Handle quantitative variables
        if quantitative:
            valid_dependents = (
                self._df.select(dependents)
                .count()
                .collect()
                .transpose(include_header=True)
                .filter(pl.col("column_0") >= min_cases)
            )["column"].to_list()
            if set(valid_dependents) != set(dependents):
                logger.warning(
                    f"Dropping {len(dependents) - len(valid_dependents)} dependent variables from analysis due to having less than {min_cases} measurements."
                )
                dependents.clear()
                dependents.extend(valid_dependents)
            return self._df.with_columns(pl.col(dependents).cast(pl.Float64))

        # Handle binary variables
        not_binary = (
            self._df.select(pl.col(dependents).unique().drop_nulls().n_unique())
            .collect()
            .transpose(include_header=True)
            .filter(pl.col("column_0") > 2)
        )["column"].to_list()
        if not_binary:
            logger.error(
                f"Dependent variables {not_binary} are not binary. Please remove from analysis."
            )
            raise ValueError
        invalid_dependents = (
            self._df.select(dependents)
            .sum()
            .collect()
            .transpose(include_header=True)
            .filter(pl.col("column_0") < min_cases)
        )["column"].to_list()
        if invalid_dependents:
            logger.log(
                "IMPORTANT",
                f"Dropping {len(invalid_dependents)} dependent variables from analysis due to having less than {min_cases} cases.",
            )
            valid_dependents = [col for col in dependents if col not in invalid_dependents]
            dependents.clear()
            dependents.extend(valid_dependents)
            new_df = self._df.drop(pl.col(invalid_dependents))
        else:
            new_df = self._df
        return new_df.with_columns(pl.col(dependents).cast(pl.UInt8))

    def handle_missing_values(self, method: str, independents: list[str]) -> pl.LazyFrame:
        """
        Handle missing values in the DataFrame using the specified method.

        Parameters:
        -----------
        method : str
            The method to handle missing values. If 'drop', rows with missing values
            in the specified independents will be dropped. Otherwise, the missing values
            will be filled using the specified method (e.g., 'mean', 'median', 'mode').
        independents : list[str]
            List of column names to apply the missing value handling method.

        Returns:
        --------
        pl.DataFrame
            A new DataFrame with missing values handled according to the specified method.

        Notes:
        ------
        - If the method is 'drop', rows with missing values in the specified independents
          will be dropped, and a log message will indicate the number of rows dropped.
        - If the method is not 'drop', missing values in the specified independents will be
          filled using the specified method, and a log message will indicate the columns
          and method used.
        """
        # If method is not drop, just fill the missing values with the specified method
        if method != "drop":
            logger.info(
                f'Filling missing values in columns {",".join(independents)} with {method} method.'
            )
            return self._df.with_columns(pl.col(independents).fill_null(strategy=method))
        # If method is drop, drop rows with missing values in the specified independents
        new_df = self._df.drop_nulls(subset=independents)
        new_height = new_df.select(pl.len()).collect().item()
        old_height = self._df.select(pl.len()).collect().item()
        if new_height != old_height:
            logger.info(f"Dropped {old_height - new_height} rows with missing values.")
        return new_df

    def category_to_dummy(
        self,
        categorical_covariates: list[str],
        predictors: str,
        independents: list[str],
        covariates: list[str],
        dependents: list[str],
    ) -> pl.LazyFrame:
        """
        Converts categorical columns to dummy/one-hot encoded variables.

        This method identifies categorical columns with more than two unique values
        and converts them into dummy variables. It updates the provided lists of
        independents and covariates to reflect these changes.

        Parameters:
        -----------
        categorical_covariates : list[str]
            List of categorical covariate column names.
        predictor : str
            The name of the predictor column.
        independents : list[str]
            List of predictor column names.
        covariates : list[str]
            List of covariate column names.
        dependents : list[str]
            List of dependent column names.

        Returns:
        --------
        pl.DataFrame | pl.LazyFrame
            The modified DataFrame or LazyFrame with dummy variables.
        """
        not_binary = (
            self._df.select(pl.col(categorical_covariates).drop_nulls().n_unique())
            .collect()
            .transpose(include_header=True)
            .filter(pl.col("column_0") > 2)
        )["column"].to_list()
        if not_binary:
            plural = len(not_binary) > 1
            logger.log(
                "IMPORTANT",
                f'Categorical column{"s" if plural else ""} {",".join(not_binary)} {"are" if plural else "is"} not binary. LazyFrame will be loaded to create dummy variables.',
            )
            dummy = self._df.collect().to_dummies(not_binary, drop_first=True).lazy()
            dummy_cols = dummy.collect_schema().names()
            # Update the lists in place to keep track of the independents and covariates
            independents.clear()
            independents.extend([col for col in dummy_cols if col not in dependents])
            original_covariates = [col for col in covariates]  # Make a copy for categorical knowledge
            covariates.clear()
            covariates.extend([col for col in independents if col not in predictors])
            binary_covariates = [col for col in categorical_covariates if col not in not_binary]
            new_binary_covariates = [col for col in covariates if col not in original_covariates]
            categorical_covariates.clear()
            categorical_covariates.extend(binary_covariates + new_binary_covariates)
            return dummy.lazy()
        return self._df

    def transform_continuous(
        self, transform: str, independents: list[str], categorical_covariates: list[str]
    ) -> pl.LazyFrame:
        """
        Transforms continuous independents in the DataFrame based on the specified transformation method.

        Parameters:
        -----------
        transform : str
            The transformation method to apply. Supported methods are 'standard' for standardization
            and 'min-max' for min-max scaling.
        independents : list[str]
            A list of all predictor column names in the DataFrame.
        categorical_covariates : list[str]
            A list of categorical covariate column names to exclude from transformation.

        Returns:
        --------
        pl.DataFrame | pl.LazyFrame
            The DataFrame with transformed continuous independents.

        Notes:
        ------
        - If the specified transformation method is not recognized, the original DataFrame is returned.
        - The method logs the transformation process for continuous independents.
        """
        continuous_independents = [col for col in independents if col not in categorical_covariates]
        if transform == "standard":
            logger.info(f"Standardizing continuous independents {continuous_independents}.")
            return self._df.with_columns(pl.col(continuous_independents).transforms.standardize())
        elif transform == "min-max":
            logger.info(f"Min-max scaling continuous independents {continuous_independents}.")
            return self._df.with_columns(pl.col(continuous_independents).transforms.min_max())
        return self._df

    def phewas_check(self, is_phewas: bool, sex_col: str) -> pl.LazyFrame:
        if not is_phewas:
            return self._df
        col_names = self._df.collect_schema().names()
        if sex_col not in col_names:
            logger.error(
                f"Column {sex_col} not found in PheWAS dataframe. Please provide the correct sex column name."
            )
            raise ValueError(
                f"Column {sex_col} not found in PheWAS dataframe. Please provide the correct sex column name."
            )
        male_codes_in_df = [col for col in col_names if col in male_specific_codes]
        female_codes_in_df = [col for col in col_names if col in female_specific_codes]
        pre_counts = (
            self._df.select([*male_codes_in_df, *female_codes_in_df])
            .count()
            .collect()
            .transpose(
                include_header=True,
                header_name="phecode",
                column_names=["count"],
                # column_names=["phecode", "count"]
            )
        )
        code_matched = self._df.with_columns(
            [
                pl.when(pl.col(sex_col) == 1)  # Females should always be coded as 1
                .then(None)  # We want these to be not included in analysis
                .otherwise(pl.col(male_code))
                .alias(male_code)
                for male_code in male_codes_in_df
            ]
        ).with_columns(
            [
                pl.when(pl.col(sex_col) == 0)  # Males should always be coded as 0
                .then(None)  # We want these to be not included in analysis
                .otherwise(pl.col(female_code))
                .alias(female_code)
                for female_code in female_codes_in_df
            ]
        )
        post_counts = (
            code_matched.select([*male_codes_in_df, *female_codes_in_df])
            .count()
            .collect()
            .transpose(include_header=True, header_name="phecode", column_names=["count"])
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

    def run_associations(
        self,
        predictors: list[str],
        covariates: list[str],
        dependents: list[str],
        model: str,
        is_phewas: bool,
        sex_col: str,
        flipwas: bool
    ) -> pl.DataFrame:
        num_groups = len(predictors) * len(dependents)
        s_p = "s" if len(predictors) > 1 else ""
        s_d = "s" if len(dependents) > 1 else ""
        logger.info(
            f"Running associations for {len(predictors)} predictor{s_p} over {len(dependents)} dependent{s_d}."
        )
        reg_func = partial(
            run_association_test,
            model_type=model,
            num_groups=num_groups,
            is_phewas=is_phewas,
            sex_col=sex_col,
        )
        reg_frame = self._df.collect().lazy()
        result_frame = pl.DataFrame()
        for predictor in predictors:
            res_list = []
            for dependent in dependents:
                if flipwas:
                    order = [dependent, *covariates, predictor]
                else:
                    order = [predictor, *covariates, dependent]
                lazy_df = (
                    reg_frame
                    .select(
                        pl.col(order),
                        pl.struct(order).alias("model_struct"),
                    )
                    .drop_nulls([predictor, dependent])
                    .select(
                        pl.col("model_struct")
                        .map_batches(reg_func, returns_scalar=True, return_dtype=pl.Struct)
                        .alias("result")
                    )
                )
                res_list.append(lazy_df)
            results = pl.collect_all(res_list)
            output = pl.concat([result.unnest("result") for result in results]).sort("pval")
            result_frame = pl.concat([result_frame, output])
            print(result_frame)
        # output.write_csv(f'{output_path}_{predictor}.csv')
        if is_phewas:
            if flipwas:
                left_col = 'predictor'
            else:
                left_col = 'dependent'
            result_frame = result_frame.join(phecode_defs, left_on=left_col, right_on="phecode").sort(
                ["predictor", "pval"]
            )
            print(result_frame)
        return result_frame


@pl.api.register_expr_namespace("transforms")
class Transforms:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def standardize(self) -> pl.Expr:
        return (self._expr - self._expr.mean()) / self._expr.std()

    def min_max(self) -> pl.Expr:
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())
