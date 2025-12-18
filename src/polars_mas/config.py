from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from functools import partial
from loguru import logger

import polars as pl


@dataclass
class MASConfig:
    """
    Config class to hold all configuration parameters for the MAS analysis
    """

    analysis_type: Literal["phewas", "flipwas"]
    input: Path
    output: Path
    predictors: str
    dependents: str
    covariates: str

    # Derived attributes post-init
    column_names: list[str] = field(default_factory=list, init=False)
    n_columns: int = field(default_factory=int, init=False)
    predictor_columns: list[str] = field(default_factory=list, init=False)
    dependent_columns: list[str] = field(default_factory=list, init=False)
    covariate_columns: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate and process the inputs after initialization"""
        self._validate_io()
        self._parse_column_lists()
        self._assert_unique_column_sets()

    def _validate_io(self):
        "Validate input and output paths"
        if not self.input.exists():
            raise ValueError(f"Input file does not exist: {self.input}")
        if not self.output.parent.exists():
            raise ValueError(f"Output directory does not exist: {self.output.parent}")
        
        # Parse the input columns
        if self.input.suffix == ".parquet":
            reader = pl.scan_parquet
        elif self.input.suffix == ".csv":
            reader = pl.scan_csv
        elif self.input.suffix == ".tsv":
            reader = partial(pl.scan_csv, separator="\t")
        elif self.input.suffix == ".txt":
            reader = partial(pl.scan_csv, separator="\t")
        else:
            raise ValueError(f'Unsupported input file format: {self.input.suffix}')
        self.column_names = reader(self.input).collect_schema().names()
        self.n_columns = len(self.column_names)

    def _parse_column_lists(self) -> None:
        "Parse the column list arguments into lists of column names"
        self.predictor_columns = self._parse_column_list(self.predictors)
        self.dependent_columns = self._parse_column_list(self.dependents)
        self.covariate_columns = self._parse_column_list(self.covariates)

    def _parse_column_list(self, column_str: str | None) -> list[str]:
        "Parse a single column list argument into a list of column names"
        if column_str is None:
            return []
        col_splits = column_str.split(',')
        column_list = []
        for col in col_splits:
            # Indexed columns start with the 'i:' identifier
            if col[:2] == 'i:':
                column_list.extend(self._extract_indexed_columns(col))
            else:
                if col not in self.column_names:
                    raise ValueError(f"Column {col} does not exist in the input file.")
                column_list.append(col)
        return column_list

    def _extract_indexed_columns(self, index_str: str) -> list[str]:
        "Extract the column indicies from an index column string"
        indicies = index_str.split(':')[-1]
        # Only one column index passed
        if indicies.isnumeric():
            index = int(indicies)
            if index >= self.n_columns:
                raise ValueError(f"Index {index} is out of range for input file with {self.n_columns} columns")
            return [self.column_names[index]]
        # Multiple column indices passed
        elif '-' in indicies:
            start, end = indicies.split('-')
            start = int(start)
            # End is either specified or should be all remaining columns
            end = int(end) if end != "" else self.n_columns
            if start >= self.n_columns:
                raise ValueError(f"Start index {start} is out of range for input file with {self.n_columns} columns")
            if end > self.n_columns:
                raise ValueError(
                    f"End index {end} out of range for {self.n_columns} columns. If you want to use all remaining columns, use {start}-."
                )
            return self.column_names[start:end]
        else:
            raise ValueError('Invalid index format. Please use i:<index>, i:<start>-<end>, or i:<start>-.')
        
    def _assert_unique_column_sets(self):
        "Ensure that the predictor, dependent, and covariate columns are unique"
        predictor_set = set(self.predictor_columns)
        dependent_set = set(self.dependent_columns)
        covariate_set = set(self.covariate_columns)

        if predictor_set & dependent_set:
            raise ValueError("Predictor and dependent columns must be unique")
        if predictor_set & covariate_set:
            raise ValueError("Predictor and covariate columns must be unique")
        if dependent_set & covariate_set:
            raise ValueError("Dependent and covariate columns must be unique")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MASConfig:
        """Create a MASConfig from parsed CLI arguments."""
        return cls(
            analysis_type=args.analysis_type,
            input=args.input,
            output=args.output,
            predictors=args.predictors,
            dependents=args.dependents,
            covariates=args.covariates,
        )

    def summary(self):
        logger.info(
            "\nConfiguration summary:\n"
            f"  Analysis type: {self.analysis_type}\n"
            f"  Input file: {self.input}\n"
            f"  Output prefix: {self.output}\n"
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