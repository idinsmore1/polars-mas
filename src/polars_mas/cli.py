import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polars Multiple Association Study (MAS) CLI")
    # Specify what kind of analysis the user wants to perform. This will be useful.
    parser.add_argument(
        "analysis_type", help="The type of analysis to perform", choices=["phewas", "flipwas"], type=str
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the analysis. Will show a summary of the input and output configuration."
    )
    # Input options
    input_group = parser.add_argument_group("Input Options", "Options for specifying input data")
    input_group.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input file path. Can be a .parquet, .csv, .tsv, or .txt file. File suffix must match the file format. If using .txt, ensure it is tab-delimited.",
    )
    input_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file prefix. Will be appended with the appropriate suffix based on analysis.",
    )
    input_group.add_argument(
        "-p",
        "--predictors",
        type=str,
        help="Predictor columns (comma separated list, names or 'i:INDEX for indices)",
    )
    input_group.add_argument(
        "-d",
        "--dependents",
        type=str,
        help="Dependent columns (comma separated list, names or 'i:INDEX for indices)",
    )
    input_group.add_argument(
        "-c",
        "--covariates",
        type=str,
        help="Covariate columns (comma separated list, names or 'i:INDEX for indices)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(argv)
