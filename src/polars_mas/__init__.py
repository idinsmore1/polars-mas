import os
import sys
import argparse

# import polars as pl
from loguru import logger
from pathlib import Path
from threadpoolctl import threadpool_limits
from typing import Callable
from importlib import import_module
from pprint import pprint


def multiple_association_study() -> None:
    parser = argparse.ArgumentParser(
        description="Polars-MAS: A Python package for multiple association analysis."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input file path.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output file prefix. Will be suffixed with '{predictor}.csv'.")
    parser.add_argument(
        "-p",
        "--predictors",
        required=True,
        type=str,
        nargs="+",
        help="Predictor column names. These will be tested independently",
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        help='Column separator. Default is ","',
        default=",",
    )
    # Column selection arguments
    parser.add_argument(
        "-d",
        "--dependents",
        type=str,
        nargs="+",
        help="Dependent variable column names.",
        default=None,
    )
    parser.add_argument(
        "-di",
        "--dependents-indices",
        type=str,
        help="""Dependent variable column indicies. Ignored if --dependents is used.
        Accepts comma separated list of indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid.
        Range follows python slicing conventions - includes start, excludes end.""",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--covariates",
        type=str,
        nargs="+",
        help="Covariate column names.",
        default=None,
    )
    parser.add_argument(
        "-ci",
        "--covariates-indicies",
        type=str,
        help="""Covariate column indicies. Ignored if --covariates is used.
        Accepts comma separated list of indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid.
        Range follows python slicing conventions - includes start, excludes end.""",
        default=None,
    )
    parser.add_argument(
        "-cc",
        "--categorical-covariates",
        type=str,
        nargs="+",
        help="Categorical covariate column names.",
        default=None,
    )
    parser.add_argument(
        "-nv",
        "--null-values",
        type=str,
        nargs="+",
        help="List of values to be treated as missing values. Default is None (normal polars option).",
        default=None,
    )
    # Test parameter arguments
    parser.add_argument(
        "-qt",
        "--quantitative",
        action="store_true",
        help="Dependent variables are quantitative traits.",
    )
    parser.add_argument(
        "-m",
        "--missing",
        type=str,
        choices=["drop", "forward", "backward", "min", "max", "mean", "zero", "one"],
        help="Method to handle missing values in covariates and predictor variables. If not specified, rows with missing values in the predictor and covariate columns will be dropped.",
        default="drop",
    )
    parser.add_argument(
        "-t",
        "--transform",
        type=str,
        choices=["standard", "min-max"],
        help="Transform continuous covariates/predictor variables. Default is no transformation.",
        default=None,
    )
    parser.add_argument(
        "-mc",
        "--min-cases",
        type=int,
        help="Minimum number of cases for each dependent variable. Only applied when not --quantitative. Default is 20.",
        default=20,
    )
    parser.add_argument(
        "-lm",
        "--linear-model",
        type=str,
        choices=["lm", "glm"],
        help="Type of linear model to fit. Default is lm.",
        default="lm",
    )
    parser.add_argument(
        "-bm",
        "--binary-model",
        type=str,
        choices=["firth", "logistic"],
        help="Type of binary model to fit. Default is firth logistic regression.",
        default="firth",
    )
    parser.add_argument(
        "--phewas",
        action="store_true",
        help="Input data uses Phecodes for dependent variables.",
    )
    parser.add_argument(
        "--phewas-sex-col",
        type=str,
        help="Sex covariate column name for PheWAS analysis. Default = 'sex'. Must be coded as male = 0 and female = 1.",
        default="sex",
    )
    # Stuff for polars and numpy
    parser.add_argument(
        "-fr",
        "--frame-type",
        type=str,
        choices=["lazy", "eager"],
        help="Type of Polars Frame to use. Defaults to lazy.",
        default="lazy",
    )
    parser.add_argument(
        "-th",
        "--threads",
        type=int,
        help="Number of threads for numpy and sklearn to use.",
        default=1,
    )
    parser.add_argument(
        "-pt",
        "--polars-threads",
        type=int,
        help="Number of threads for polars to use. Defaults to all threads on machine.",
        default=os.cpu_count(),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="have more verbose logging")
    args = parser.parse_args()
    setup_logger(args.output, args.verbose)
    _validate_args(args)
    run_mas = _load_and_limit(args.threads, args.polars_threads)
    _log_args(args)
    # Run Aurora
    run_mas(**vars(args))


def _log_args(args):
    log = "Input arguments:\n"
    for key, value in vars(args).items():
        print_val = value
        if key in ['dependents', 'covariates', 'predictors', 'categorical_covariates']:
            if len(value) > 5:
                print_val = f"{','.join(value[:2])}...{','.join(value[-2:])} - ({len(value)} total)"
            else:
                print_val = ','.join(value)
        log += f"\t{key}: {print_val}\n"
    logger.info(log)


def _load_and_limit(threads: int, polars_threads: int) -> Callable:
    """
    Configures the environment and imports necessary modules for Aurora.

    This function sets the maximum number of threads for the Polars library,
    imports the Polars module, Aurora's Polars extension, and the main logic
    module. It also limits the number of threads used by the thread pool.
    We do this in a function as polars threads have to be set before importing.

    Args:
        threads (int): The maximum number of threads to be used by the thread pool.
        polars_threads (int): The maximum number of threads to be used by the Polars library.

    Returns:
        the aurora function from the main logic module.
    """
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)
    pl = import_module("polars")
    pla = import_module("polars_mas.mas_frame")
    polars_mas = import_module("polars_mas.main")
    threadpool_limits(limits=threads)
    return polars_mas.run_mas


def _validate_args(args):
    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")
    if not args.output.parent.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output.parent}")
    # Load in the header of the input file to check passed columns
    file_col_names = _load_input_header(args.input, args.separator)
    logger.info(f"{len(file_col_names)} columns found in input file.")
    # Check predictor
    if any([predictor not in file_col_names for predictor in args.predictors]):
        raise ValueError(f"Predictor column '{args.predictors}' not found in input columns.")

    # Check dependents
    if args.dependents:
        for dep in args.dependents:
            if dep not in file_col_names:
                raise ValueError(f"Dependent column '{dep}' not found in input file.")
    elif args.dependents_indices:
        args.dependents = _match_columns_to_indices(args.dependents_indices, file_col_names)
    else:
        raise ValueError("No dependent variables specified.")

    # Check covariates
    if args.covariates:
        for cov in args.covariates:
            if cov not in file_col_names:
                raise ValueError(f"Covariate column '{cov}' not found in input file.")
    elif args.covariates_indicies:
        args.covariates = _match_columns_to_indices(args.covariates_indicies, file_col_names)
    else:
        args.covariates = []

    ## Check categorical covariates
    if args.categorical_covariates and not args.covariates:
        raise ValueError("Categorical covariates specified without specifying covariates")
    elif args.categorical_covariates:
        for cov in args.categorical_covariates:
            if cov not in args.covariates:
                raise ValueError(
                    f"Categorical covariate column '{cov}' not found in given covariates: {args.covariates}."
                )
    else:
        args.categorical_covariates = []

    # Check that threads < polars_threads and that polars_threads <= os.cpu_count()
    if args.polars_threads > os.cpu_count():
        logger.warning(
            f"Number of Polars threads ({args.polars_threads}) exceeds number of available CPUs ({os.cpu_count()}). Setting Polars threads to {os.cpu_count()}."
        )
        args.polars_threads = os.cpu_count()
    if args.threads >= args.polars_threads:
        logger.warning(
            f"Number of computation threads ({args.threads}) exceeds number of Polars threads ({args.polars_threads}). Setting threads to {args.polars_threads}."
        )
        args.threads = args.polars_threads


########### Validation functions ############
def _load_input_header(input_file: Path, separator: str) -> list[str]:
    with input_file.open() as f:
        return f.readline().strip().split(separator)


def _match_columns_to_indices(indices: str, col_names: list[str]) -> list[str]:
    if "," in indices:
        splits = indices.split(",")
        output_columns = []
        for split in splits:
            if "" == split:
                continue
            # Recursively call this function to handle separated indices
            output_columns.extend(_match_columns_to_indices(split, col_names))
        # output_columns now is a flat list of all columns as text
        return output_columns

    if indices.isnumeric():
        index = int(indices)
        if index >= len(col_names):
            raise ValueError(f"Index {index} out of range for {len(col_names)} columns in input file.")
        return [col_names[int(indices)]]

    elif "-" in indices:
        start, end = indices.split("-")
        start_idx = int(start)
        if start_idx >= len(col_names):
            raise ValueError(
                f"Start index {start_idx} out of range for input file column indices. {len(col_names)} columns found."
            )
        if end != "" and int(end) >= len(col_names):
            raise ValueError(
                f"End index {end} out of range for {len(col_names)} columns. If you want to use all remaining columns, use {start_idx}-."
            )
        if end != "":
            end_idx = int(end)
            return col_names[start_idx:end_idx]
        return col_names[start_idx:]
    else:
        raise ValueError(f"Invalid index format, must use '-' for a range: {indices}")


def setup_logger(output: Path, verbose: bool):
    logger.remove()

    log_file_path = output.with_suffix(".log")
    if log_file_path.exists():
        log_file_path.unlink()
    logger.add(
        log_file_path,
        format="{time: DD-MM-YYYY -> HH:mm} | {level} | {message}",
        level="INFO",
        enqueue=True,
    )
    if verbose:
        stdout_level = "DEBUG"
        stderr_level = "WARNING"
    else:
        stdout_level = "INFO"
        stderr_level = "ERROR"

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time: DD-MM-YYYY -> HH:mm:ss}</green> <level>{message}</level>",
        level=stdout_level,
        filter=lambda record: record["level"].name not in ["WARNING", "ERROR"],
        enqueue=True,
    )
    logger.add(
        sys.stderr,
        colorize=True,
        format="<red>{time: DD-MM-YYYY -> HH:mm:ss}</red> <level>{message}</level>",
        level=stderr_level,
        filter=lambda record: record["level"].name not in ["DEBUG", "INFO", "SUCCESS"],
        enqueue=True,
    )


if __name__ == "__main__":
    multiple_association_study()
