from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline.config import Config


def ensure_dirs(config: Config) -> None:
    for path in [
        config.data_dir,
        config.raw_dir,
        config.processed_dir,
        config.manual_dir,
        config.imdb_dir,
        config.seeds_dir,
        config.cache_dir,
        config.tvmaze_cache_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv_with_comment(path: Path, comment: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {comment}\n")
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
