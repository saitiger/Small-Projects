from __future__ import annotations

import argparse
import logging

try:
    from google.cloud import bigquery
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery"
    ) from exc

from bq_migrate.checks import run_checks
from bq_migrate.config import get_config
from bq_migrate.load import load_all
from bq_migrate.views import create_views


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="BigQuery migration for TV finale dataset")
    parser.add_argument("--project", dest="project_id", default=None)
    parser.add_argument("--dataset", dest="dataset_id", default=None)
    parser.add_argument("--location", dest="location", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("load-all", help="Load all CSVs into BigQuery")
    subparsers.add_parser("create-views", help="Create analysis views")
    subparsers.add_parser("run-checks", help="Run post-load checks")
    subparsers.add_parser("all", help="Run load + views + checks")

    args = parser.parse_args()
    config = get_config(project_id=args.project_id, dataset_id=args.dataset_id, location=args.location)

    client = bigquery.Client(project=config.project_id, location=config.location)

    if args.command == "load-all":
        load_all(client, config)
        return

    if args.command == "create-views":
        create_views(client, config)
        return

    if args.command == "run-checks":
        run_checks(client, config)
        return

    if args.command == "all":
        load_all(client, config)
        create_views(client, config)
        run_checks(client, config)
        return


if __name__ == "__main__":
    main()
