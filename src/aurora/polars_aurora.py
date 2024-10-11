import polars as pl
import polars.selectors as cs
import numpy as np

from aurora.consts import male_specific_codes, female_specific_codes


@pl.api.register_dataframe_namespace("aurora")
@pl.api.register_lazyframe_namespace("aurora")
class PhewasFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        self._df = df
        self.col_names = df.collect_schema().names()

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
