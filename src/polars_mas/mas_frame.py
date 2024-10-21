import polars as pl
import time
import datetime

from functools import partial
from pathlib import Path

from loguru import logger
from polars_mas.consts import male_specific_codes, female_specific_codes, phecode_defs
from polars_mas.model_funcs import polars_firth_regression


@pl.api.register_dataframe_namespace("polars_mas")
@pl.api.register_lazyframe_namespace("polars_mas")
class MASFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df

    def check_independents_for_constants(self, independents, drop=False) -> pl.DataFrame | pl.LazyFrame:
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
                           adds an additional log if columns are dropped.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame with constant columns dropped if `drop` is True,
                                         otherwise the original DataFrame.

        Raises:
            ValueError: If constant columns are found and `drop` is False.

        Notes:
            - This method works with both `pl.DataFrame` and `pl.LazyFrame`.
            - The method logs an error message if constant columns are found and `drop` is False.
            - The method logs an info message if constant columns are dropped or if no constant columns are found.
        """
        if isinstance(self._df, pl.DataFrame):
            const_cols = (
                self._df.select(pl.col(independents).drop_nulls().unique().len())
                .transpose(include_header=True)
                .filter(pl.col("column_0") == 1)
                .select(pl.col("column"))
            )["column"].to_list()
        else:
            const_cols = (
                self._df.select(pl.col(independents).drop_nulls().unique().len())
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
            logger.warning(f'Dropping constant columns {",".join(const_cols)}')
            new_independents = [col for col in independents if col not in const_cols]
            independents.clear()
            independents.extend(new_independents)
            return self._df.drop(pl.col(const_cols))
        return self._df

    def check_grouped_independents_for_constants(
        self, independents: list[str], dependent: str = None
    ) -> list[str]:
        if isinstance(self._df, pl.DataFrame):
            const_cols = (
                self._df.select(pl.col(independents).drop_nulls().unique().len())
                .transpose(include_header=True)
                .filter(pl.col("column_0") == 1)
                .select(pl.col("column"))
            )["column"].to_list()
        else:
            const_cols = (
                self._df.select(pl.col(independents).drop_nulls().unique().len())
                .collect()
                .transpose(include_header=True)
                .filter(pl.col("column_0") == 1)
                .select(pl.col("column"))
            )["column"].to_list()
        if const_cols:
            logger.warning(
                f'Columns {",".join(const_cols)} are constants. Dropping from {dependent} analysis.'
            )
            non_consts = [col for col in independents if col not in const_cols]
            return non_consts
        return independents

    def validate_dependents(self, dependents: list[str], quantitative: bool, min_cases: int) -> pl.DataFrame | pl.LazyFrame:
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
            if isinstance(self._df, pl.LazyFrame):
                valid_dependents = (
                    self._df
                    .select(dependents)
                    .count()
                    .collect()
                    .transpose(include_header=True)
                    .filter(pl.col('column_0') >= min_cases)
                )['column'].to_list()
            else:
                valid_dependents = (
                    self._df
                    .select(dependents)
                    .count()
                    .transpose(include_header=True)
                    .filter(pl.col('column_0') >= min_cases)
                )['column'].to_list()
            if set(valid_dependents) != set(dependents):
                logger.warning(f'Dropping {len(dependents) - len(valid_dependents)} dependent variables from analysis due to having less than {min_cases} measurements.')
                dependents.clear()
                dependents.extend(valid_dependents)
            return self._df.with_columns(pl.col(dependents).cast(pl.Float64))
        
        # Handle binary variables
        if isinstance(self._df, pl.DataFrame):
            not_binary = (
                self._df.select(pl.col(dependents).unique().drop_nulls().n_unique())
                .transpose(include_header=True)
                .filter(pl.col("column_0") > 2)
            )["column"].to_list()
        else:
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
        if isinstance(self._df, pl.LazyFrame):
                invalid_dependents = (
                    self._df
                    .select(dependents)
                    .sum()
                    .collect()
                    .transpose(include_header=True)
                    .filter(pl.col('column_0') < min_cases)
                )['column'].to_list()
        else:
            invalid_dependents = (
                self._df
                .select(dependents)
                .sum()
                .transpose(include_header=True)
                .filter(pl.col('column_0') < min_cases)
            )['column'].to_list()
        if invalid_dependents:
            logger.warning(f'Dropping {len(invalid_dependents)} dependent variables from analysis due to having less than {min_cases} cases.')
            valid_dependents = [col for col in dependents if col not in invalid_dependents]
            dependents.clear()
            dependents.extend(valid_dependents)
            new_df = self._df.drop(pl.col(invalid_dependents))
        else:
            new_df = self._df
        return new_df.with_columns(pl.col(dependents).cast(pl.UInt8))

    def handle_missing_values(self, method: str, independents: list[str]):
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
        if isinstance(new_df, pl.DataFrame):
            if new_df.height != self._df.height:
                logger.info(f"Dropped {self._df.height - new_df.height} rows with missing values.")
        else:
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
    ) -> pl.DataFrame | pl.LazyFrame:
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
        if isinstance(self._df, pl.DataFrame):
            not_binary = (
                self._df.select(pl.col(categorical_covariates).drop_nulls().n_unique())
                .transpose(include_header=True)
                .filter(pl.col("column_0") > 2)
            )["column"].to_list()
        else:
            not_binary = (
                self._df.select(pl.col(categorical_covariates).drop_nulls().n_unique())
                .collect()
                .transpose(include_header=True)
                .filter(pl.col("column_0") > 2)
            )["column"].to_list()
        if not_binary:
            plural = len(not_binary) > 1
            if isinstance(self._df, pl.LazyFrame):
                logger.warning(
                    f'Categorical column{"s" if plural else ""} {",".join(not_binary)} {"are" if plural else "is"} not binary. LazyFrame will be loaded to create dummy variables.'
                )
                cats = self._df.collect()
            else:
                logger.info(
                    f'Categorical column{"s" if plural else ""} {",".join(not_binary)} {"are" if plural else "is"} not binary. Creating dummy variables.'
                )
                cats = self._df
            dummy = cats.to_dummies(not_binary, drop_first=True)
            dummy_cols = dummy.collect_schema().names()
            # Update the lists in place to keep track of the independents and covariates
            independents.clear()
            independents.extend([col for col in dummy_cols if col not in dependents])
            original_covars = [col for col in covariates]  # Make a copy for categorical knowledge
            covariates.clear()
            covariates.extend([col for col in independents if col != predictors])
            binary_covars = [col for col in categorical_covariates if col not in not_binary]
            new_binary_covars = [col for col in covariates if col not in original_covars]
            categorical_covariates.clear()
            categorical_covariates.extend(binary_covars + new_binary_covars)

            if isinstance(self._df, pl.LazyFrame):
                # Convert the dummy dataframe back to a LazyFrame for faster operations
                dummy = pl.LazyFrame(dummy)
            return dummy
        return self._df

    def transform_continuous(
        self, transform: str, independents: list[str], categorical_covariates: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
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

    def melt(
        self, predictors: list[str], independents: list[str], dependents: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Transforms the DataFrame by melting specified columns into a long format suitable for modeling.

        Args:
            predictors (list[str]): List of predictor column names to be melted.
            independents (list[str]): List of independent column names to be retained.
            dependents (list[str]): List of dependent column names to be melted.

        Returns:
            pl.DataFrame | pl.LazyFrame: A DataFrame or LazyFrame with the melted structure, including a new column 'model_struct'
            that contains a struct of predictor, predictor_value, covariates, dependent, and dependent_value.

        Notes:
            - The method first unpivots the DataFrame on the dependent columns, then drops any rows with null values in the
              'dependent_value' column.
            - It then unpivots the DataFrame again on the predictor columns.
            - A new column 'model_struct' is created, which is a struct containing the predictor, predictor_value, covariates,
              dependent, and dependent_value.
            - The 'independents' list is modified in place to include 'predictor_value' and covariates.
        """
        covars = [col for col in independents if col not in predictors]
        melted_df = (
            self._df.unpivot(
                index=independents,
                on=dependents,
                variable_name="dependent",
                value_name="dependent_value",
            )
            .drop_nulls(subset=["dependent_value"])
            .unpivot(
                index=covars + ["dependent", "dependent_value"],
                on=predictors,
                variable_name="predictor",
                value_name="predictor_value",
            )
            .drop_nulls(subset=["predictor_value"])
            .with_columns(
                pl.struct("predictor", "predictor_value", *covars, "dependent", "dependent_value").alias(
                    "model_struct"
                )
            )
        )
        independents.clear()
        independents.extend(["predictor_value", *covars])
        return melted_df

    def phewas_filter(self, is_phewas: bool, sex_col: str, drop: True) -> pl.DataFrame | pl.LazyFrame:
        if not is_phewas:
            return self._df
        sex_specific_codes = male_specific_codes + female_specific_codes
        if sex_col not in self._df.collect_schema().names():
            start_phrase = f"Column {sex_col} not found in PheWAS dataframe."
            if not drop:
                logger.error(f"{start_phrase} Please provide the correct column name.")
                raise ValueError
            logger.warning(f"{start_phrase} Sex specific phecodes will be dropped.")
            return self._df.filter(~pl.col("dependent").is_in(sex_specific_codes))
        # Otherwise, filter
        condition = (
            # Keep rows where sex is not male (1) OR phecode is not in female_specific_phecodes
            ((pl.col(sex_col) != 0) | ~pl.col("dependent").is_in(female_specific_codes))
            &
            # AND keep rows where sex is not female (0) OR phecode is not in male_specific_phecodes
            ((pl.col(sex_col) != 1) | ~pl.col("dependent").is_in(male_specific_codes))
        )
        return self._df.filter(condition)

    def run_associations(
        self,
        independents: list[str],
        quantitative: bool,
        binary_model: str,
        linear_model: str,
        is_phewas: bool,
    ) -> pl.DataFrame | pl.LazyFrame:
        if not quantitative:
            if binary_model == "firth":
                reg_function = partial(
                    polars_firth_regression,
                    independents=independents,
                    dependent_values="dependent_value",
                )
                start_time = time.perf_counter()
                output = (
                    self._df.group_by("dependent", "predictor")
                    .agg(
                        pl.col("model_struct")
                        .map_batches(reg_function, return_dtype=pl.Struct, returns_scalar=True)
                        .alias("result")
                    )
                    .unnest("result")
                )
                if is_phewas:
                    # Add on the phecode definitions
                    if isinstance(output, pl.LazyFrame):
                        output = output.join(phecode_defs, left_on="dependent", right_on="phecode")
                    else:
                        output = output.join(
                            phecode_defs.collect(), left_on="dependent", right_on="phecode"
                        )
            elif binary_model != "firth":
                logger.warning(
                    "Other implementations have not be made yet. Please use 'firth' for binary models."
                )
        else:
            logger.error("Quantitative models have not been implemented yet.")
            raise NotImplementedError
        # All outputs will be named output
        if isinstance(output, pl.LazyFrame):
            output = output.collect()
        output = (
            output.fill_nan(None)
            .select(
                [pl.col("dependent"), pl.col("predictor"), pl.all().exclude(["dependent", "predictor"])]
            )
            .sort(["predictor", "pval"], nulls_last=True)
        )
        end_time = time.perf_counter()
        logger.success(f"Associations Complete! Runtime: {str(datetime.timedelta(seconds=(round(end_time - start_time))))}")
        return output


@pl.api.register_expr_namespace("transforms")
class Transforms:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def standardize(self) -> pl.Expr:
        return (self._expr - self._expr.mean()) / self._expr.std()

    def min_max(self) -> pl.Expr:
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())
