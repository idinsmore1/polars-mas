import os
from polars_mas.cli import parse_args


def main():
    args = parse_args()
    # Set polars max threads before importing polars
    os.environ["POLARS_MAX_THREADS"] = str(args.num_workers)
    # Now import the other packages to limit polars threads
    from polars_mas.config import MASConfig
    from polars_mas.pipeline import run_pipeline
    config = MASConfig.from_args(args)
    config.summary()
    # If this a dry-run, then stop here
    if args.dry_run:
        return
    run_pipeline(config)
