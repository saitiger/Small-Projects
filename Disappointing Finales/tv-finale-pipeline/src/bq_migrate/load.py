from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
try:
    from google.cloud import bigquery
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery"
    ) from exc

from bq_migrate.config import BQConfig
from bq_migrate.schemas import schema_for_table

logger = logging.getLogger(__name__)


DATE_COLUMNS = {
    "episodes": ["air_date"],
    "features": ["final_season_premiere_date", "penultimate_season_finale_date"],
    "mapping": ["tvmaze_premiered", "tvmaze_ended"],
}


def ensure_dataset(client: bigquery.Client, config: BQConfig) -> None:
    dataset_ref = bigquery.Dataset(f"{config.project_id}.{config.dataset_id}")
    dataset_ref.location = config.location
    try:
        client.get_dataset(dataset_ref)
        logger.info("Dataset exists: %s", dataset_ref.dataset_id)
    except Exception:
        logger.info("Creating dataset: %s", dataset_ref.dataset_id)
        client.create_dataset(dataset_ref)


def _clean_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _write_temp_csv(df: pd.DataFrame, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_table(client: bigquery.Client, config: BQConfig, table: str, csv_path: Path) -> int:
    table_id = f"{config.project_id}.{config.dataset_id}.{table}"
    schema = schema_for_table(table)

    df = pd.read_csv(csv_path)
    df = _clean_dates(df, DATE_COLUMNS.get(table, []))

    temp_dir = config.input_dir / "tmp_bq"
    temp_csv = _write_temp_csv(df, temp_dir, table)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_jagged_rows=True,
        allow_quoted_newlines=True,
    )

    with temp_csv.open("rb") as handle:
        job = client.load_table_from_file(handle, table_id, job_config=job_config)
    job.result()

    table_ref = client.get_table(table_id)
    logger.info("Loaded %s rows into %s", table_ref.num_rows, table_id)
    return int(table_ref.num_rows)


def load_all(client: bigquery.Client, config: BQConfig) -> dict[str, int]:
    ensure_dataset(client, config)
    results: dict[str, int] = {}
    for table, filename in config.tables.items():
        csv_path = config.input_dir / filename
        if not csv_path.exists():
            logger.warning("Missing CSV for table %s: %s", table, csv_path)
            continue
        results[table] = load_table(client, config, table, csv_path)
    return results
