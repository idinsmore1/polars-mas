import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polars Multiple Association Study (MAS) CLI")
    # Specify what kind of analysis the user wants to perform. This will be useful.
    # parser.add_argument(
    #     "analysis_type", help="The type of analysis to perform", choices=["phewas", "flipwas"], type=str
    # )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the analysis. Will show a summary of the input and output configuration.",
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
    input_group.add_argument(
        "-cc",
        "--categorical-covariates",
        type=str,
        help="Categorical covariate columns (comma separated list, names or 'i:INDEX for indices)",
    )
    input_group.add_argument(
        "-nv",
        "--null-values",
        type=str,
        help="Specify the values to be considered as null/missing in the input data (comma separated list). Default is None (polars default)",
        default=None,
    )
    # Association Parameters
    param_group = parser.add_argument_group(
        "Association Parameters", "Options for specifying association parameters"
    )
    param_group.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to use for the analysis. Default is 1.",
    )
    param_group.add_argument(
        "-t",
        "--threads",
        type=int,
        default=2,
        help="Number of threads available to each worker for analysis. Default is 2.",
    )
    param_group.add_argument(
        "--quantitative",
        "-qt",
        action="store_true",
        help="Specify if the dependent variables are quantitative instead of binary. Default is False.",
    )
    rint_logt_group = param_group.add_mutually_exclusive_group()
    rint_logt_group.add_argument(
        "--rint",
        action="store_true",
        help="Apply rank-based inverse normal transformation to the dependent variables. Default is False.",
    )
    rint_logt_group.add_argument(
        "--logt",
        action="store_true",
        help="Apply log transformation to the dependent variables. Default is False.",
    )
    param_group.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["firth", "logistic", "linear"],
        default="firth",
        help="Model to use for association analysis. Default is Firth's logistic regression.",
    )
    param_group.add_argument(
        "-mc",
        "--min-case-count",
        type=int,
        default=20,
        help="Minimum count of observations required for an association to be considered. Default is 20. Also works as the minimum observation count for linear regression.",
    )
    param_group.add_argument(
        "-mcv",
        "--missing-covariate-values",
        type=str,
        choices=["fail", "drop", "forward", "backward", "min", "max", "mean", "zero", "one"],
        help="Specify how to handle missing values in covariates. Default is raise an error if missing values are present.",
        default="fail",
    )
    phecode_group = parser.add_argument_group("PheCode Options", "PheCode Analysis-specific parameters")
    phewas_group = phecode_group.add_mutually_exclusive_group()
    phewas_group.add_argument(
        "--phewas",
        action="store_true",
        help="This is a PheWAS analysis where PheCodes are the dependent variables.",
    )
    phewas_group.add_argument(
        "--flipwas",
        action="store_true",
        help="This is a PheWAS analysis where the PheCodes are the predictor variables.",
    )
    phecode_group.add_argument(
        "--sex-col",
        type=str,
        help="Specify the column name for sex in the input data. Default='sex'",
        default="sex",
    )
    phecode_group.add_argument(
        "--female-code",
        type=int,
        default=1,
        help="Specify the code used for female in the sex column. Default is 1.",
    )
    sex_specific_group = phecode_group.add_mutually_exclusive_group()
    sex_specific_group.add_argument(
        "--male-only", action="store_true", help="Include only male samples in the analysis."
    )
    sex_specific_group.add_argument(
        "--female-only", action="store_true", help="Include only female samples in the analysis."
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress most logging output."
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(argv)
