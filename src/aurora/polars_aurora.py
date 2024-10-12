import polars as pl

from pathlib import Path
from loguru import logger
from aurora.consts import male_specific_codes, female_specific_codes


@pl.api.register_dataframe_namespace("aurora")
@pl.api.register_lazyframe_namespace("aurora")
class AuroraFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df

    def check_predictors_for_constants(self, predictors, drop=False) -> pl.DataFrame | pl.LazyFrame:
        """
        Check for constant columns in the given predictors and optionally drop them.

        This method checks if any of the specified predictor columns in the DataFrame
        have constant values (i.e., all values in the column are the same). If such
        columns are found, it either raises an error or drops them based on the `drop`
        parameter.

        Args:
            predictors (list[str]): List of predictor column names to check for constants.
            drop (bool, optional): If True, constant columns will be dropped from the DataFrame.
                                   If False, an error will be raised if constant columns are found.
                                   Defaults to False.

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
                self._df.select(pl.col(predictors).drop_nulls().unique().len())
                .transpose(include_header=True)
                .filter(pl.col("column_0") == 1)
                .select(pl.col("column"))
            )["column"].to_list()
        else:
            const_cols = (
                self._df.select(pl.col(predictors).drop_nulls().unique().len())
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
            logger.warning(f'Dropping constant columns {",".join(const_cols)}.')
            new_predictors = [col for col in predictors if col not in const_cols]
            predictors.clear()
            predictors.extend(new_predictors)
            return self._df.drop(pl.col(const_cols))
        logger.info("No constant columns found.")
        return self._df

    def validate_dependents(
        self, dependents: list[str], quantitative: bool
    ) -> pl.DataFrame | pl.LazyFrame:
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
        if quantitative:
            return self._df.with_columns(pl.col(dependents).cast(pl.Float64))
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
        return self._df.with_columns(pl.col(dependents).cast(pl.UInt8))

    def handle_missing_values(self, method: str, predictors: list[str]):
        """
        Handle missing values in the DataFrame using the specified method.

        Parameters:
        -----------
        method : str
            The method to handle missing values. If 'drop', rows with missing values
            in the specified predictors will be dropped. Otherwise, the missing values
            will be filled using the specified method (e.g., 'mean', 'median', 'mode').
        predictors : list[str]
            List of column names to apply the missing value handling method.

        Returns:
        --------
        pl.DataFrame
            A new DataFrame with missing values handled according to the specified method.

        Notes:
        ------
        - If the method is 'drop', rows with missing values in the specified predictors
          will be dropped, and a log message will indicate the number of rows dropped.
        - If the method is not 'drop', missing values in the specified predictors will be
          filled using the specified method, and a log message will indicate the columns
          and method used.
        """
        # If method is not drop, just fill the missing values with the specified method
        if method != "drop":
            logger.info(
                f'Filling missing values in columns {",".join(predictors)} with {method} method.'
            )
            return self._df.with_columns(pl.col(predictors).fill_null(strategy=method))
        # If method is drop, drop rows with missing values in the specified predictors
        new_df = self._df.drop_nulls(subset=predictors)
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
        predictor: str,
        predictors: list[str],
        covariates: list[str],
        dependents: list[str],
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Converts categorical columns to dummy/one-hot encoded variables.

        This method identifies categorical columns with more than two unique values
        and converts them into dummy variables. It updates the provided lists of
        predictors and covariates to reflect these changes.

        Parameters:
        -----------
        categorical_covariates : list[str]
            List of categorical covariate column names.
        predictor : str
            The name of the predictor column.
        predictors : list[str]
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
            # Update the lists in place to keep track of the predictors and covariates
            predictors.clear()
            predictors.extend(
                [predictor] + [col for col in dummy_cols if col not in dependents and col != predictor]
            )
            original_covars = [col for col in covariates]  # Make a copy for categorical knowledge
            covariates.clear()
            covariates.extend([col for col in predictors if col != predictor])
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
        self, transform: str, predictors: list[str], categorical_covariates: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Transforms continuous predictors in the DataFrame based on the specified transformation method.

        Parameters:
        -----------
        transform : str
            The transformation method to apply. Supported methods are 'standard' for standardization
            and 'min-max' for min-max scaling.
        predictors : list[str]
            A list of all predictor column names in the DataFrame.
        categorical_covariates : list[str]
            A list of categorical covariate column names to exclude from transformation.

        Returns:
        --------
        pl.DataFrame | pl.LazyFrame
            The DataFrame with transformed continuous predictors.

        Notes:
        ------
        - If the specified transformation method is not recognized, the original DataFrame is returned.
        - The method logs the transformation process for continuous predictors.
        """
        continuous_predictors = [col for col in predictors if col not in categorical_covariates]
        if transform == "standard":
            logger.info(f"Standardizing continuous predictors {continuous_predictors}.")
            return self._df.with_columns(pl.col(continuous_predictors).transforms.standardize())
        elif transform == "min-max":
            logger.info(f"Min-max scaling continuous predictors {continuous_predictors}.")
            return self._df.with_columns(pl.col(continuous_predictors).transforms.min_max())
        return self._df

    def melt(self, predictors: list[str], dependents: list[str]) -> pl.DataFrame | pl.LazyFrame:
        """
        Transforms the DataFrame by unpivoting specified columns and creating a structured column.
        Args:
            predictors (list[str]): List of column names to be used as predictors.
            dependents (list[str]): List of column names to be used as dependents.
        Returns:
            pl.DataFrame | pl.LazyFrame: A DataFrame or LazyFrame with the transformed structure.
        The method performs the following steps:
        1. Unpivots the DataFrame using the specified predictors and dependents.
        2. Drops rows with null values in the 'dependent_value' column.
        3. Creates a new column 'model_struct' containing a struct of the predictors, 'dependent', and 'dependent_value'.
        """
        return (
            self._df.unpivot(
                index=predictors,
                on=dependents,
                variable_name="dependent",
                value_name="dependent_value",
            )
            .drop_nulls(subset=["dependent_value"])
            .with_columns(pl.struct(*predictors, "dependent", "dependent_value").alias("model_struct"))
        )

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
            ((pl.col(sex_col) != 0) | ~pl.col("phecode").is_in(female_specific_codes))
            &
            # AND keep rows where sex is not female (0) OR phecode is not in male_specific_phecodes
            ((pl.col(sex_col) != 1) | ~pl.col("phecode").is_in(male_specific_codes))
        )
        return self._df.filter(condition)
    
    def run_associations(self, output_file: Path, quantitative: bool, is_phewas: bool) -> pl.DataFrame | pl.LazyFrame:
        pass

@pl.api.register_expr_namespace("transforms")
class Transforms:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def standardize(self) -> pl.Expr:
        return (self._expr - self._expr.mean()) / self._expr.std()

    def min_max(self) -> pl.Expr:
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())
