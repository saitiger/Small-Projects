from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    manual_dir: Path
    imdb_dir: Path
    seeds_dir: Path
    cache_dir: Path
    tvmaze_cache_dir: Path
    seed_file: Path
    imdb_shows_raw_parquet: Path
    imdb_episodes_raw_parquet: Path
    tvmaze_episodes_parquet: Path
    tvmaze_unmatched_parquet: Path
    mapping_parquet: Path
    curated_unresolved_csv: Path
    curated_ingest_report_json: Path
    shows_parquet: Path
    episodes_parquet: Path
    features_parquet: Path
    quality_report_json: Path
    labels_template_csv: Path
    labels_csv: Path
    match_threshold: float
    tvmaze_rate_limit_seconds: float
    tvmaze_timeout_seconds: int
    tvmaze_max_retries: int
    pandemic_start: str
    pandemic_end: str
    post_pandemic_start: str
    runway_bucket_low: int
    runway_bucket_high: int


IMDB_DATASET_BASE = "https://datasets.imdbws.com"
IMDB_DATASETS = {
    "title.basics.tsv.gz": f"{IMDB_DATASET_BASE}/title.basics.tsv.gz",
    "title.episode.tsv.gz": f"{IMDB_DATASET_BASE}/title.episode.tsv.gz",
    "title.ratings.tsv.gz": f"{IMDB_DATASET_BASE}/title.ratings.tsv.gz",
}


def get_config() -> Config:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    manual_dir = data_dir / "manual"
    imdb_dir = raw_dir / "imdb"
    seeds_dir = raw_dir / "seeds"
    cache_dir = raw_dir / "cache"
    tvmaze_cache_dir = cache_dir / "tvmaze"

    return Config(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        manual_dir=manual_dir,
        imdb_dir=imdb_dir,
        seeds_dir=seeds_dir,
        cache_dir=cache_dir,
        tvmaze_cache_dir=tvmaze_cache_dir,
        seed_file=seeds_dir / "imdb_top100.csv",
        imdb_shows_raw_parquet=processed_dir / "imdb_shows_raw.parquet",
        imdb_episodes_raw_parquet=processed_dir / "imdb_episodes_raw.parquet",
        tvmaze_episodes_parquet=processed_dir / "tvmaze_episodes.parquet",
        tvmaze_unmatched_parquet=processed_dir / "tvmaze_unmatched_episodes.parquet",
        mapping_parquet=processed_dir / "mapping.parquet",
        curated_unresolved_csv=processed_dir / "curated_unresolved.csv",
        curated_ingest_report_json=processed_dir / "curated_ingest_report.json",
        shows_parquet=processed_dir / "shows.parquet",
        episodes_parquet=processed_dir / "episodes.parquet",
        features_parquet=processed_dir / "features.parquet",
        quality_report_json=processed_dir / "quality_report.json",
        labels_template_csv=manual_dir / "labels_template.csv",
        labels_csv=manual_dir / "labels.csv",
        match_threshold=0.78,
        tvmaze_rate_limit_seconds=0.35,
        tvmaze_timeout_seconds=20,
        tvmaze_max_retries=3,
        pandemic_start="2020-03-11",
        pandemic_end="2022-12-31",
        post_pandemic_start="2023-01-01",
        runway_bucket_low=600,
        runway_bucket_high=1500,
    )
