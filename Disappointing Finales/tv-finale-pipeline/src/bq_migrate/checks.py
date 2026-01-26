from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
try:
    from google.cloud import bigquery
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery"
    ) from exc

from bq_migrate.config import BQConfig

logger = logging.getLogger(__name__)


def run_checks(client: bigquery.Client, config: BQConfig) -> None:
    dataset = f"{config.project_id}.{config.dataset_id}"
    _check_row_counts(client, config, dataset)
    _check_nulls(client, dataset)
    _check_views(client, dataset)


def _check_row_counts(client: bigquery.Client, config: BQConfig, dataset: str) -> None:
    for table, filename in config.tables.items():
        csv_path = config.input_dir / filename
        if not csv_path.exists():
            continue
        local_count = _count_rows(csv_path)
        query = f"SELECT COUNT(*) AS cnt FROM `{dataset}.{table}`"
        bq_count = int(client.query(query).result().to_dataframe()["cnt"].iloc[0])
        logger.info("Row count %s: local=%s bq=%s", table, local_count, bq_count)


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def _check_nulls(client: bigquery.Client, dataset: str) -> None:
    query = f"""
    SELECT
      SUM(CASE WHEN show_id IS NULL THEN 1 ELSE 0 END) AS null_show_id,
      SUM(CASE WHEN imdb_tconst IS NULL THEN 1 ELSE 0 END) AS null_imdb_tconst,
      COUNT(*) AS total_rows
    FROM `{dataset}.shows`
    """.strip()
    result = client.query(query).result().to_dataframe().iloc[0]
    logger.info("Null check shows: %s", result.to_dict())


def _check_views(client: bigquery.Client, dataset: str) -> None:
    for view in ["v_finale_analysis", "v_finale_rankings", "v_top_bottom"]:
        query = f"SELECT 1 FROM `{dataset}.{view}` LIMIT 1"
        client.query(query).result()
        logger.info("View ok: %s", view)
