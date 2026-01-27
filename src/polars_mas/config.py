from __future__ import annotations

import os
import sys
import argparse
import polars as pl

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from types import FunctionType
from functools import partial
from loguru import logger


@dataclass
class MASConfig:
    """
    Config class to hold all configuration parameters for the MAS analysis
    """

    # analysis_type: Literal["phewas", "flipwas"]
    input: Path
    output: Path
    predictors: str
    dependents: str
    covariates: str
    categorical_covariates: str
    null_values: str | None
    num_workers: int
    num_threads: int
    model: Literal["firth", "logistic", "linear"]
    min_case_count: int
    missing_covariate_values: Literal[
        "fail", "drop", "forward", "backward", "min", "max", "mean", "zero", "one"
    ]
    quantitative: bool
    rint: bool
    logt: bool
    is_phewas: bool
    is_flipwas: bool
    sex_col: str
    female_code: int
    male_only: bool
    female_only: bool
    verbose: bool
    quiet: bool
    output_type: Literal["parquet", "csv", "tsv", "txt"]


    # Derived attributes post-init
    reader: FunctionType | partial[pl.LazyFrame] | None = field(default=None, init=False)
    column_names: list[str] = field(default_factory=list, init=False)
    total_column_count: int = field(default_factory=int, init=False)
    included_column_count: int = field(default_factory=int, init=False)
    included_row_count: int = field(default_factory=int, init=False)
    ipc_file: str | None = field(default=None, init=False)
    # Column lists
    predictor_columns: list[str] = field(default_factory=list, init=False)
    dependent_columns: list[str] = field(default_factory=list, init=False)
    covariate_columns: list[str] = field(default_factory=list, init=False)
    categorical_covariate_columns: list[str] = field(default_factory=list, init=False)
    included_columns: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate and process the inputs after initialization"""
        self._validate_io()
        self._parse_column_lists()
        self._assert_unique_column_sets()

    def setup_logger(self):
        logger.remove()
        if self.quiet:
            logger.add(
                sys.stdout, level="SUCCESS", filter=lambda record: record["level"].no <= 25, enqueue=True
            )
            logger.add(sys.stderr, level="ERROR", enqueue=True)
        elif self.verbose:
            logger.add(
                sys.stdout, level="DEBUG", filter=lambda record: record["level"].no <= 25, enqueue=True
            )
            logger.add(sys.stderr, level="WARNING", enqueue=True)
        else:
            # Show everything above INFO
            logger.add(
                sys.stdout,
                level="INFO",
                filter=lambda record: record["level"].no <= 25,
                enqueue=True,
            )
            logger.add(
                sys.stderr,
                level="WARNING",
                enqueue=True,
            )

    def _validate_io(self):
        "Validate input and output paths"
        if not self.input.exists():
            raise FileNotFoundError(f"Input file does not exist: {self.input}")
        if not self.output.parent.exists():
            raise ValueError(f"Output directory does not exist: {self.output.parent}")

        # Parse the input columns
        null_values = None if self.null_values is None else self.null_values.split(",")
        if self.input.suffix == ".parquet":
            self.reader = pl.scan_parquet  # Parquet has nulls defined in schema, don't need a partial
        elif self.input.suffix == ".csv":
            self.reader = partial(pl.scan_csv, null_values=null_values)
        elif self.input.suffix == ".tsv":
            self.reader = partial(pl.scan_csv, separator="\t", null_values=null_values)
        elif self.input.suffix == ".txt":
            self.reader = partial(pl.scan_csv, separator="\t", null_values=null_values)
        else:
            raise ValueError(f"Unsupported input file format: {self.input.suffix}")

        self.column_names = self.reader(self.input).collect_schema().names()
        self.total_column_count = len(self.column_names)

    def _parse_column_lists(self) -> None:
        "Parse the column list arguments into lists of column names"
        self.predictor_columns = self._parse_column_list(self.predictors)
        self.dependent_columns = self._parse_column_list(self.dependents)
        self.covariate_columns = self._parse_column_list(self.covariates)
        self.categorical_covariate_columns = self._parse_column_list(self.categorical_covariates)

    def _parse_column_list(self, column_str: str | None) -> list[str]:
        "Parse a single column list argument into a list of column names"
        if column_str is None:
            return []
        col_splits = column_str.split(",")
        column_list = []
        for col in col_splits:
            # Indexed columns start with the 'i:' identifier
            if col[:2] == "i:":
                column_list.extend(self._extract_indexed_columns(col))
            else:
                if col not in self.column_names:
                    raise ValueError(f"Column {col} does not exist in the input file.")
                column_list.append(col)
        return column_list

    def _extract_indexed_columns(self, index_str: str) -> list[str]:
        "Extract the column indicies from an index column string"
        indicies = index_str.split(":")[-1]
        # Only one column index passed
        if indicies.isnumeric():
            index = int(indicies)
            if index >= self.total_column_count:
                raise ValueError(
                    f"Index {index} is out of range for input file with {self.total_column_count} columns"
                )
            return [self.column_names[index]]
        # Multiple column indices passed
        elif "-" in indicies:
            start, end = indicies.split("-")
            start = int(start)
            # End is either specified or should be all remaining columns
            end = int(end) if end != "" else self.total_column_count
            if start >= self.total_column_count:
                raise ValueError(
                    f"Start index {start} is out of range for input file with {self.total_column_count} columns"
                )
            if end > self.total_column_count:
                raise ValueError(
                    f"End index {end} out of range for {self.total_column_count} columns. If you want to use all remaining columns, use {start}-."
                )
            return self.column_names[start:end]
        else:
            raise ValueError(
                "Invalid index format. Please use i:<index>, i:<start>-<end>, or i:<start>-."
            )

    def _assert_unique_column_sets(self):
        "Ensure that the predictor, dependent, and covariate columns are unique"
        predictor_set = set(self.predictor_columns)
        dependent_set = set(self.dependent_columns)
        covariate_set = set(self.covariate_columns)
        cat_covariate_set = set(self.categorical_covariate_columns)

        if predictor_set & dependent_set:
            raise ValueError("Predictor and dependent columns must be unique")
        if predictor_set & covariate_set:
            raise ValueError("Predictor and covariate columns must be unique")
        if dependent_set & covariate_set:
            raise ValueError("Dependent and covariate columns must be unique")
        if not cat_covariate_set:
            pass
        elif not cat_covariate_set & covariate_set:
            raise ValueError("Categorical covariate columns must be a subset of covariate columns")
        included_columns = list(predictor_set | dependent_set | covariate_set)
        # We do this step so that they are ordered in the same order as they appear in the file
        self.included_columns = [col for col in self.column_names if col in included_columns]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MASConfig:
        """Create a MASConfig from parsed CLI arguments."""
        return cls(
            # analysis_type=args.analysis_type,
            input=args.input,
            output=args.output,
            predictors=args.predictors,
            dependents=args.dependents,
            covariates=args.covariates,
            categorical_covariates=args.categorical_covariates,
            null_values=args.null_values,
            num_workers=args.num_workers,
            num_threads=args.threads,
            model=args.model,
            min_case_count=args.min_case_count,
            quantitative=args.quantitative,
            rint=args.rint,
            logt=args.logt,
            missing_covariate_values=args.missing_covariate_values,
            is_phewas=args.phewas,
            is_flipwas=args.flipwas,
            sex_col=args.sex_col,
            female_code=args.female_code,
            male_only=args.male_only,
            female_only=args.female_only,
            verbose=args.verbose,
            quiet=args.quiet,
            output_type=args.output_type,
        )

    def summary(self):
        logger.info(
            "\nConfiguration summary:\n"
            # f"  Analysis type: {self.analysis_type}\n"
            f"  Input file: {self.input}\n"
            f"  Output prefix: {self.output}\n"
            f"  Max Polars Threads: {pl.thread_pool_size()}\n"
            f"  Max Computation Threads: {self.num_threads}\n"
            f"  Predictors:  {self._format_column_list(self.predictor_columns)}\n"
            f"  Dependents:  {self._format_column_list(self.dependent_columns)}\n"
            f"  Covariates:  {self._format_column_list(self.covariate_columns)}"
        )

    @staticmethod
    def _format_column_list(columns: list[str], max_display: int = 5) -> str:
        """Format column list for display, truncating if too long."""
        n = len(columns)
        if n == 0:
            return "(none)"
        if n <= max_display:
            return f"{n} column{'s' if n != 1 else ''}: {', '.join(columns)}"
        # Show first 2 and last 2 with count
        preview = f"{columns[0]}, {columns[1]}, ... {columns[-2]}, {columns[-1]}"
        return f"{n} columns: {preview}"

    def read_data(self) -> pl.LazyFrame:
        if self.reader is None:
            raise ValueError("Reader function is not set.")
        lf = self.reader(self.input).select(self.included_columns)
        self.update_row_and_column_counts(lf)
        logger.info(
            f"Successfully read {self.input.name} and selected {self.included_row_count} rows and {self.included_column_count} columns"
        )
        return lf

    def update_row_and_column_counts(self, lf: pl.LazyFrame) -> None:
        """Update the included row and column counts based on a LazyFrame"""
        self.included_column_count = lf.collect_schema().len()
        self.included_row_count = lf.select(pl.len()).collect().item()
