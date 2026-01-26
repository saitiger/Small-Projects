from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class BQConfig:
    project_id: str
    dataset_id: str
    location: str
    input_dir: Path
    tables: Dict[str, str]


def get_config(project_id: str | None = None, dataset_id: str | None = None, location: str | None = None) -> BQConfig:
    project = project_id or os.getenv("BQ_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("Project ID not provided. Use --project or set BQ_PROJECT_ID.")

    dataset = dataset_id or os.getenv("BQ_DATASET_ID") or "tv_finals"
    loc = location or os.getenv("BQ_LOCATION") or "US"

    input_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    tables = {
        "shows": "shows.csv",
        "episodes": "episodes.csv",
        "features": "features.csv",
        "mapping": "mapping.csv",
    }
    return BQConfig(project_id=project, dataset_id=dataset, location=loc, input_dir=input_dir, tables=tables)
