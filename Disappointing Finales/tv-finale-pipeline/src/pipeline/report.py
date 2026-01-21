from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_json

logger = logging.getLogger(__name__)


def build_report(
    config: Config,
    shows_df: pd.DataFrame | None = None,
    episodes_df: pd.DataFrame | None = None,
    mapping_df: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a quality report with missingness and mapping stats."""
    ensure_dirs(config)

    if config.quality_report_json.exists() and not overwrite:
        return json.loads(config.quality_report_json.read_text(encoding="utf-8"))

    shows_df = shows_df if shows_df is not None else read_parquet(config.shows_parquet)
    episodes_df = episodes_df if episodes_df is not None else read_parquet(config.episodes_parquet)
    mapping_df = mapping_df if mapping_df is not None else read_parquet(config.mapping_parquet)

    total_shows = int(len(shows_df))
    mapped_shows = int(mapping_df["tvmaze_id"].notna().sum()) if not mapping_df.empty else 0
    mapping_success_rate = mapped_shows / total_shows if total_shows else 0.0

    total_episodes = int(len(episodes_df))
    missing_runtime_pct = float(episodes_df["runtime_minutes"].isna().mean()) if total_episodes else 0.0
    missing_airdate_pct = float(episodes_df["air_date"].isna().mean()) if total_episodes else 0.0
    missing_ratings_pct = float(episodes_df["imdb_rating"].isna().mean()) if total_episodes else 0.0

    failed_shows = mapping_df[mapping_df["tvmaze_id"].isna()] if not mapping_df.empty else pd.DataFrame()
    failed_shows = failed_shows.merge(shows_df[["imdb_tconst", "title"]], on="imdb_tconst", how="left")
    top_failed = (
        failed_shows[["imdb_tconst", "title", "match_confidence", "chosen_reason"]]
        .head(10)
        .to_dict(orient="records")
        if not failed_shows.empty
        else []
    )

    warnings = []
    if not episodes_df.empty:
        coverage = episodes_df.groupby("show_id")["has_tvmaze_match"].mean().reset_index()
        coverage = coverage[coverage["has_tvmaze_match"] < 0.5]
        if not coverage.empty:
            warnings.append(
                {
                    "type": "low_tvmaze_match_coverage",
                    "count": int(len(coverage)),
                    "examples": coverage.merge(shows_df[["show_id", "title"]], on="show_id", how="left")
                    .head(5)
                    .to_dict(orient="records"),
                }
            )

    report = {
        "counts": {
            "total_shows": total_shows,
            "mapped_shows": mapped_shows,
            "mapping_success_rate": mapping_success_rate,
            "total_episodes": total_episodes,
        },
        "missingness": {
            "missing_runtime_pct": missing_runtime_pct,
            "missing_airdate_pct": missing_airdate_pct,
            "missing_ratings_pct": missing_ratings_pct,
        },
        "top_failed_shows": top_failed,
        "warnings": warnings,
    }

    write_json(config.quality_report_json, report)
    logger.info("Quality report written: %s", config.quality_report_json)
    return report
