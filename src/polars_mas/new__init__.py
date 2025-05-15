import os
import sys
import argparse

from copy import deepcopy
from importlib import import_module
from itertools import cycle
from pathlib import Path
from typing import Callable
from loguru import logger
from threadpoolctl import threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="polars-mas", description="A python package for multiple association studies."
    )
    # Add a group for the input arguments
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging. Default is False.",
    )
    ingroup = parser.add_argument_group("Inputs", "Input file arguments.")
    ingroup.add_argument("-i", "--input", required=True, type=Path, help="Input file path.")
    ingroup.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output file prefix. Will be suffixed with '{suffix}.csv'",
    )
    ingroup.add_argument(
        '-f',
        '--suffix',
        type=str,
        choices=['predictors', 'dependents'],
        default='predictors',
        help="Use either the name of the predictors or the dependents as the suffix for the output files. Default is predictor.",
    )
    ingroup.add_argument(
        "-s", "--separator", type=str, default=",", help="Separator for the input file. Default is ','"
    )
    # predictors
    predictors = ingroup.add_mutually_exclusive_group(required=True)
    predictors.add_argument(
        "-p",
        "--predictors",
        type=str,
        help="Comma-separated list of predictor variables. Cannot be used with --predictors-indices.",
    )
    predictors.add_argument(
        "-pi",
        "--predictors-indices",
        type=str,
        help="""Predictor variable column indicies. Cannot be used with --predictors.
        Accepts comma separated list of indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid.
        Range follows python slicing conventions - includes start, excludes end.""",
    )
    # dependents
    dependents = ingroup.add_mutually_exclusive_group(required=True)
    dependents.add_argument(
        "-d",
        "--dependents",
        type=str,
        help="Comma-separated list of dependent variables. Cannot be used with --dependents-indices.",
    )
    dependents.add_argument(
        "-di",
        "--dependents-indices",
        type=str,
        help="""Dependent variable column indicies. Cannot be used with --dependents.
        Accepts comma separated list of indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid.
        Range follows python slicing conventions - includes start, excludes end.""",
    )
    # covariates
    covariates = ingroup.add_mutually_exclusive_group(required=False)
    covariates.add_argument(
        "-c",
        "--covariates",
        type=str,
        help="Comma-separated list of covariate variables. Cannot be used with --covariates-indices.",
    )
    covariates.add_argument(
        "-ci",
        "--covariates-indices",
        type=str,
        help="""Covariate variable column indicies. Cannot be used with --covariates.
        Accepts comma separated list of indices/indicies ranges. E.g. 2, 2-5, 2-, 2,3 , 2,5-8, 2,8- are all valid.
        Range follows python slicing conventions - includes start, excludes end.""",
    )
    ingroup.add_argument(
        "-cc",
        "--categorical-covariates",
        type=str,
        help="Comma-separated list of categorical covariate variables. Default is None.",
        default=None,
    )
    ingroup.add_argument(
        "-nv",
        "--null-values",
        type=str,
        help="Comma-separated list of null values. Default is None (polars default).",
        default=None,
    )
    # Add group for association settings
    assoc_group = parser.add_argument_group(
        "Association Settings", "Settings for the association analysis."
    )
    assoc_group.add_argument(
        "-qt",
        "--quantitative",
        action="store_true",
        help="Dependent variables are quantitative, not binary. Default is False.",
    )
    assoc_group.add_argument(
        "-n",
        "--num-workers",
        type=int,
        help="Number of workers for parallel processing used by polars. Default is the number of available CPU cores - 1",
        default=os.cpu_count() - 1,
    )
    assoc_group.add_argument(
        "-t",
        "--threads-per-worker",
        type=int,
        help="Number of threads each worker has available. Default is 1.",
        default=1,
    )
    assoc_group.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["firth", "linear"],
        default="firth",
        help="Model to use for association analysis. Default is 'firth'.",
    )
    assoc_group.add_argument(
        "-mi",
        "--missing",
        type=str,
        choices=["drop", "forward", "backward", "min", "max", "mean", "zero", "one"],
        help="Method to handle missing values in covariates and predictor variables. If not specified, rows with missing values in the predictor and covariate columns will be dropped.",
        default="drop",
    )
    phewas_group = assoc_group.add_mutually_exclusive_group(required=False)
    phewas_group.add_argument(
        "--phewas",
        action="store_true",
        help="This is a PheWAS analysis with phecodes as the dependent variables.",
    )
    phewas_group.add_argument(
        "--flipwas",
        action="store_true",
        help="This is a PheWAS analysis with phecodes as the predictor variables",
    )
    sex_group = assoc_group.add_mutually_exclusive_group(required=False)
    sex_group.add_argument(
        '-mo',
        '--male-only',
        action='store_true',
        help="Use only males in the analysis. Default is False.",
    )
    sex_group.add_argument(
        '-fo',
        '--female-only',
        action='store_true',
        help="Use only females in the analysis. Default is False.",
    )
    assoc_group.add_argument(
        "--sex-col",
        type=str,
        help="Column name for sex-based analysis. Default is sex. Assumes females = 1 and males = 0.",
        default='sex',
    )
    args = parser.parse_args()
    setup_logger(args.output, args.verbose)
    validate_args(args)
    log_args(args)
    load_polars_and_limit_threads(args)


def load_polars_and_limit_threads(args):
    """Polars has to be limited before importing it"""
    os.environ['POLARS_MAX_THREADS'] = str(args.num_workers)
    import_module("polars")
    import_module("polars_mas.mas_frame")


def setup_logger(output: Path, verbose: bool):
    logger.remove()
    log_file_path = output.with_suffix(".log")
    if log_file_path.exists():
        log_file_path.unlink()
    logger.add(
        log_file_path,
        format="{time: DD-MM-YYYY -> HH:mm} | {level} | {message}",
        level="DEBUG",
        enqueue=True,
    )
    if verbose:
        stdout_level = "INFO"
        stderr_level = "WARNING"
    else:
        stdout_level = "INFO"
        stderr_level = "ERROR"
    logger.level("PROGRESS", no=23, color="<cyan>", icon="🕐")
    logger.level("IMPORTANT", no=25, color="<yellow>", icon="⚠️")
    logger.add(
        sys.stdout,
        # colorize=True,
        format="<green>{time: DD-MM-YYYY -> HH:mm:ss}</green> <level>{message}</level>",
        level=stdout_level,
        filter=lambda record: record["level"].name not in ["WARNING", "ERROR"],
        enqueue=True,
    )
    logger.add(
        sys.stderr,
        # colorize=True,
        format="<red>{time: DD-MM-YYYY -> HH:mm:ss}</red> <level>{message}</level>",
        level=stderr_level,
        filter=lambda record: record["level"].name not in ["DEBUG", "INFO", "SUCCESS"],
        enqueue=True,
    )


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate the arguments passed to the script and return a SimpleNamespace object with the validated arguments."""
    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    if not args.output.parent.exists():
        raise FileNotFoundError(f"Output directory {args.output.parent} does not exist.")
    # Load the column headers
    file_col_names = _load_input_header(args.input, args.separator)

    # Load the predictors
    if args.predictors:
        predictors = args.predictors.split(",")
    elif args.predictors_indices:
        predictors = _match_columns_to_indices(args.predictors_indices, file_col_names)
    if not predictors:
        raise ValueError("No predictors found. Please provide a valid list of predictors or indices.")
    for predictor in predictors:
        if predictor not in file_col_names:
            raise ValueError(f"Predictor {predictor} not found in input file.")
    args.predictors = predictors
    delattr(args, "predictors_indices")

    # Load the dependents
    if args.dependents:
        dependents = args.dependents.split(",")
    elif args.dependents_indices:
        dependents = _match_columns_to_indices(args.dependents_indices, file_col_names)
    if not dependents:
        raise ValueError("No dependents found. Please provide a list of dependents or indices.")
    for dependent in dependents:
        if dependent not in file_col_names:
            raise ValueError(f"Dependent {dependent} not found in input file.")
    args.dependents = dependents
    delattr(args, "dependents_indices")

    # load the covariates
    if args.covariates:
        covariates = args.covariates.split(",")
    elif args.covariates_indices:
        covariates = _match_columns_to_indices(args.covariates_indices, file_col_names)
    else:
        covariates = []
    for covariate in covariates:
        if covariate not in file_col_names:
            raise ValueError(f"Covariate {covariate} not found in input file.")
    args.covariates = covariates
    delattr(args, "covariates_indices")
    args.independents = predictors + covariates
    
    if args.categorical_covariates:
        if not covariates:
            raise ValueError("Categorical covariates cannot be used without passing them to covariates flag ('-c'/'-ci').")
        categorical_covariates = args.categorical_covariates.split(",")
        for covariate in categorical_covariates:
            if covariate not in covariates:
                raise ValueError(f"Categorical covariate {covariate} not found in covariates list.")
    else:
        categorical_covariates = []
    args.categorical_covariates = categorical_covariates

    if args.quantitative and args.model in ["firth"]:
        raise ValueError("Quantitative traits must be used with linear based models.")
    if not args.quantitative and args.model in ["linear"]:
        raise ValueError("Binary traits must be used with logistic based models.")
    
    # Check that threads < polars_threads and that polars_threads <= os.cpu_count()
    if args.num_workers > os.cpu_count():
        logger.warning(
            f"Number of worker threads ({args.num_workers}) exceeds number of available CPUs ({os.cpu_count()}). Setting worker threads to {os.cpu_count()}."
        )
        args.num_workers = os.cpu_count()
    if args.threads_per_worker > os.cpu_count():
        logger.warning(
            f"Number of computation threads ({args.threads_per_worker}) exceeds number of available CPUs ({os.cpu_count()}). Setting threads to 1."
        )
        args.threads_per_worker = 1
    if args.flipwas and args.suffix == "predictors":
        logger.warning('This is a flipped PheWAS analysis. All output files will be merged into {dependent}.csv to reduce the number of files.')
        args.suffix = "dependents"

    if args.male_only or args.female_only:
        if args.sex_col not in file_col_names:
            raise ValueError(f"Column {args.sex_col} not found in input file, but specified a sex-specific analysis. Please set the correct sex column name with --sex-col.")
    return args


def log_args(args):
    log = "Input arguments:\n"
    skip_keys = [
        "null_values",
    ]
    val_dict = {k: v for k, v in vars(args).items() if k not in skip_keys}
    for key, value in val_dict.items():
        if key in skip_keys:
            continue
        print_val = value
        if key in ["dependents", "covariates", "predictors", "categorical_covariates"]:
            if len(value) > 5:
                print_val = f"{','.join(value[:2])}...{','.join(value[-2:])} - ({len(value)} total)"
            else:
                print_val = ",".join(value)
        log += f"\t{key}: {print_val}\n"
    logger.opt(ansi=True).info(log)


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