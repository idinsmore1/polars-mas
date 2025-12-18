from polars_mas.cli import parse_args
from polars_mas.config import MASConfig


def main():
    args = parse_args()
    config = MASConfig.from_args(args)
    config.summary()
    # If this a dry-run, then stop here
    if args.dry_run:
        return

