import polars as pl
import polars.selectors as cs
import numpy as np

from loguru import logger
from aurora.consts import male_specific_codes, female_specific_codes


@pl.api.register_dataframe_namespace("aurora")
@pl.api.register_lazyframe_namespace("aurora")
class AuroraFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df

    def check_for_constants(self) -> pl.DataFrame | pl.LazyFrame:
        """
        Check for columns in the dataframe are not constant.
        """
        if isinstance(self._df, pl.DataFrame):
            const_cols = (
                self._df
                .select(pl.all().unique().len())
                .transpose(include_header=True)
                .filter(pl.col('column_0') == 1)
                .select(pl.col('column'))
            )['column'].to_list()
        else:
            const_cols = (
                self._df
                .select(pl.all().unique().len())
                .collect()
                .transpose(include_header=True)
                .filter(pl.col('column_0') == 1)
                .select(pl.col('column'))
            )['column'].to_list()
        if const_cols:
            logger.error(f'Columns {const_cols} are constants. Please remove from analysis.')
            raise ValueError
        logger.info('No constant columns found.')
        return self._df

    def handle_missing_values(self, method: str, predictors: list[str]):
        # If method is not drop, just fill the missing values with the specified method
        if method != 'drop':
            logger.info(f'Filling missing values in columns {predictors} with {method} method.')
            return self._df.with_columns(pl.col(predictors).fill_null(strategy=method))
        # If method is drop, drop rows with missing values in the specified predictors
        new_df = self._df.drop_nulls(subset=predictors)
        if isinstance(new_df, pl.DataFrame):
            if new_df.height != self._df.height:
                logger.info(f'Dropped {self._df.height - new_df.height} rows with missing values.')
        else:
            new_height = new_df.select(pl.len()).collect().item()
            old_height = self._df.select(pl.len()).collect().item()
            if new_height != old_height:
                logger.info(f'Dropped {old_height - new_height} rows with missing values.')
        return new_df
    
    def category_to_dummy(self, categorical_covariates: list[str], predictor: str, predictors: list[str], covariates: list[str], dependents: list[str]) -> pl.DataFrame | pl.LazyFrame:
        if isinstance(self._df, pl.DataFrame):
            not_binary = (
                self._df
                .select(pl.col(categorical_covariates).n_unique())
                .transpose(include_header=True)
                .filter(pl.col('column_0') > 2)
            )['column'].to_list()
        else:
            not_binary = (
                self._df
                .select(pl.col(categorical_covariates).n_unique())
                .collect()
                .transpose(include_header=True)
                .filter(pl.col('column_0') > 2)
            )['column'].to_list()
        if not_binary:
            if isinstance(self._df, pl.LazyFrame):
                logger.warning(f'Columns {not_binary} are not binary. LazyFrame will be loaded to create dummy variables.')
                cats = self._df.collect()
            else:
                logger.info(f'Columns {not_binary} are not binary. Creating dummy variables.')
                cats = self._df
            dummy = cats.to_dummies(not_binary, drop_first=True)
            dummy_cols = dummy.collect_schema().names()
            # Update the lists in place to keep track of the predictors and covariates
            predictors.clear()
            predictors.extend([predictor] + [col for col in dummy_cols if col not in dependents and col != predictor])
            original_covars = [col for col in covariates] # Make a copy for categorical knowledge
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
    
    def transform_continuous(self, transform: str, predictors: list[str], categorical_covariates: list[str]) -> pl.DataFrame | pl.LazyFrame:
        continuous_predictors = [col for col in predictors if col not in categorical_covariates]
        if transform == 'standard':
            logger.info(f'Standardizing continuous predictors {continuous_predictors}.')
            return self._df.with_columns(pl.col(continuous_predictors).transforms.standardize())
        elif transform == 'min-max':
            logger.info(f'Min-max scaling continuous predictors {continuous_predictors}.')
            return self._df.with_columns(pl.col(continuous_predictors).transforms.min_max())
        return self._df
    

@pl.api.register_expr_namespace('transforms')
class Transforms:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def standardize(self) -> pl.Expr:
        return (self._expr - self._expr.mean()) / self._expr.std()
    
    def min_max(self) -> pl.Expr:
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())

# @pl.api.register_dataframe_namespace("aurora")
# @pl.api.register_lazyframe_namespace("aurora")
class PhewasFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df

    def handle_missing(self, predictors: list[str], method=None) -> pl.DataFrame:
        """
        Handle missing values in the specified predictors.
        Parameters:
        -----------
        predictors : list[str]
            List of predictor column names to handle missing values for.
        method : str, optional
            Method to handle missing values. If None, rows with missing values
            in the specified predictors will be dropped.
            Can be one of 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'.
            Default is None.
        Returns:
        --------
        pl.DataFrame
            A DataFrame with missing values handled according to the specified method.
        Raises:
        -------
        ValueError
            If an unknown method is specified.
        """
        valid_strategies = [
            None,
            "forward",
            "backward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]
        if method not in valid_strategies:
            raise ValueError(
                f"Unknown method '{method}'. Valid strategies are: {valid_strategies}"
            )
        if method is None:
            new_df = self._df.drop_nulls(subset=predictors)
            if isinstance(new_df, pl.DataFrame):
                if new_df.height != self._df.height:
                    print(
                        f"Dropped {self._df.height - new_df.height} rows with missing values."
                    )
            else:
                new_height = new_df.select(pl.len()).collect().item()
                old_height = self._df.select(pl.len()).collect().item()
                if new_height != old_height:
                    print(
                        f"Dropped {old_height - new_height} rows with missing values."
                    )
        else:
            new_df = self._df.fill_null(subset=predictors, strategy=method)
        return new_df

    def filter_by_count(
        self, phenotypes: list[str], min_cases: int = 20
    ) -> pl.DataFrame:
        pheno_data = self._df.select(phenotypes)
        if isinstance(pheno_data, pl.LazyFrame):
            case_counts = pheno_data.sum().collect().transpose()
            total_counts = pheno_data.count().collect().transpose()
        else:
            case_counts = pheno_data.sum().transpose()
            total_counts = pheno_data.count().transpose()
        control_counts = total_counts - case_counts
        target_df = pl.DataFrame(
            {
                "phenotype": phenotypes,
                "cases": case_counts,
                "controls": control_counts,
                "total": total_counts,
            }
        )
        invalid_phenos = target_df.filter(
            (pl.col("cases") < min_cases) | (pl.col("controls") < min_cases)
        )
        if invalid_phenos.height > 0:
            print(
                f"{invalid_phenos.height} phenotypes dropped due to having less than {min_cases} cases/controls."
            )
        valid_phenos = target_df.filter(
            (pl.col("cases") >= min_cases) & (pl.col("controls") >= min_cases)
        )
        valid_cols = sorted(
            list(set(phenotypes).intersection(set(valid_phenos["phenotype"].to_list())))
        )
        return self._df.select(pl.all().exclude(phenotypes), *valid_cols)

    def filter_sex_specific_codes(self, drop=False) -> pl.DataFrame:
        """
        filter sex-specific codes in the DataFrame.

        This method processes the DataFrame to filter sex-specific codes based on the specified method.

        Parameters:
        drop: bool -> Drop the sex specific code names from the analysis. Default = False.

        Returns:
        pl.DataFrame: The processed DataFrame with sex-specific codes handled.
        """

        if "sex" not in [col for col in self.col_names] or drop:
            if not drop:
                print(
                    "Sex is not listed in the dataframe. Sex-specific phecodes will be dropped."
                )
            else:
                print("Dropping sex-specific phecodes")
            return self._df.filter(
                ~pl.col("phecode").is_in(male_specific_codes + female_specific_codes)
            )
        condition = (
            # Keep rows where sex is not male (1) OR phecode is not in female_specific_phecodes
            ((pl.col("sex") != 0) | ~pl.col("phecode").is_in(female_specific_codes))
            &
            # AND keep rows where sex is not female (0) OR phecode is not in male_specific_phecodes
            ((pl.col("sex") != 1) | ~pl.col("phecode").is_in(male_specific_codes))
        )
        return self._df.filter(condition)
