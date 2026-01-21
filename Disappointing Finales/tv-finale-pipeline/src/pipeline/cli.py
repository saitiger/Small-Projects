from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.config import get_config
from pipeline.features import build_features
from pipeline.imdb_download import download_imdb_datasets
from pipeline.imdb_parse import build_imdb_tables
from pipeline.normalize import normalize
from pipeline.report import build_report
from pipeline.seed_top100 import seed_top100
from pipeline.tvmaze_client import tvmaze_enrich


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    _setup_logging()
    config = get_config()

    parser = argparse.ArgumentParser(description="TV finale data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed-top100", help="Seed IMDb top 100 list")
    seed_parser.add_argument("--input-csv", type=Path, default=None)
    seed_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("imdb-download", help="Download IMDb datasets")

    build_parser = subparsers.add_parser("build-imdb", help="Build IMDb show + episode tables")
    build_parser.add_argument("--overwrite", action="store_true")

    tvmaze_parser = subparsers.add_parser("tvmaze-enrich", help="Enrich with TVMaze")
    tvmaze_parser.add_argument("--overwrite", action="store_true")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize to canonical tables")
    normalize_parser.add_argument("--overwrite", action="store_true")

    features_parser = subparsers.add_parser("features", help="Build show-level features")
    features_parser.add_argument("--overwrite", action="store_true")

    report_parser = subparsers.add_parser("report", help="Generate quality report")
    report_parser.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("run-all", help="Run the full pipeline")

    args = parser.parse_args()

    if args.command == "seed-top100":
        seed_top100(config, input_csv=args.input_csv, force=args.force)
        return

    if args.command == "imdb-download":
        download_imdb_datasets(config)
        return

    if args.command == "build-imdb":
        build_imdb_tables(config, overwrite=args.overwrite)
        return

    if args.command == "tvmaze-enrich":
        tvmaze_enrich(config, overwrite=args.overwrite)
        return

    if args.command == "normalize":
        normalize(config, overwrite=args.overwrite)
        return

    if args.command == "features":
        build_features(config, overwrite=args.overwrite)
        return

    if args.command == "report":
        build_report(config, overwrite=args.overwrite)
        return

    if args.command == "run-all":
        try:
            seed_top100(config)
        except FileNotFoundError:
            download_imdb_datasets(config)
            seed_top100(config, force=True)
        download_imdb_datasets(config)
        shows_df, episodes_df = build_imdb_tables(config)
        mapping_df, tvmaze_episodes_df = tvmaze_enrich(config, shows_df=shows_df)
        shows_df, episodes_df = normalize(
            config,
            shows_df=shows_df,
            episodes_df=episodes_df,
            mapping_df=mapping_df,
            tvmaze_episodes_df=tvmaze_episodes_df,
        )
        build_features(config, shows_df=shows_df, episodes_df=episodes_df)
        build_report(config, shows_df=shows_df, episodes_df=episodes_df, mapping_df=mapping_df)
        return


if __name__ == "__main__":
    main()
