from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)


def build_imdb_tables(config: Config, seed_path: Path | None = None, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build show and episode tables from IMDb TSV datasets."""
    ensure_dirs(config)
    seed_path = seed_path or config.seed_file
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    if config.imdb_shows_raw_parquet.exists() and config.imdb_episodes_raw_parquet.exists() and not overwrite:
        logger.info("Loading cached IMDb tables")
        return read_parquet(config.imdb_shows_raw_parquet), read_parquet(config.imdb_episodes_raw_parquet)

    seed_df = pd.read_csv(seed_path, comment="#", dtype=str)
    seed_df = seed_df.dropna(subset=["imdb_tconst"]).copy()
    seed_tconsts = set(seed_df["imdb_tconst"].tolist())

    shows = _build_shows_table(config, seed_tconsts)
    episodes = _build_episodes_table(config, seed_tconsts)

    write_parquet(shows, config.imdb_shows_raw_parquet)
    write_parquet(episodes, config.imdb_episodes_raw_parquet)
    return shows, episodes


def _build_shows_table(config: Config, seed_tconsts: set[str]) -> pd.DataFrame:
    basics_path = config.imdb_dir / "title.basics.tsv.gz"
    if not basics_path.exists():
        raise FileNotFoundError("Missing IMDb title.basics.tsv.gz; run imdb-download.")

    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(basics_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["tconst"].isin(seed_tconsts)]
        if chunk.empty:
            continue
        chunk = chunk[["tconst", "primaryTitle", "originalTitle", "startYear", "endYear", "genres"]]
        rows.append(chunk)

    if not rows:
        raise RuntimeError("No IMDb shows matched seed list.")

    shows = pd.concat(rows, ignore_index=True)
    shows = shows.rename(columns={"tconst": "imdb_tconst"})
    return shows


def _build_episodes_table(config: Config, seed_tconsts: set[str]) -> pd.DataFrame:
    episode_path = config.imdb_dir / "title.episode.tsv.gz"
    ratings_path = config.imdb_dir / "title.ratings.tsv.gz"
    basics_path = config.imdb_dir / "title.basics.tsv.gz"
    if not episode_path.exists() or not ratings_path.exists() or not basics_path.exists():
        raise FileNotFoundError("Missing IMDb datasets. Run imdb-download.")

    episode_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(episode_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["parentTconst"].isin(seed_tconsts)]
        if chunk.empty:
            continue
        chunk = chunk[["parentTconst", "tconst", "seasonNumber", "episodeNumber"]]
        episode_chunks.append(chunk)

    if not episode_chunks:
        logger.warning("No IMDb episodes matched seed list.")
        return pd.DataFrame(
            columns=[
                "parentTconst",
                "episode_tconst",
                "seasonNumber",
                "episodeNumber",
                "imdb_rating",
                "imdb_votes",
                "episode_title",
            ]
        )

    episodes = pd.concat(episode_chunks, ignore_index=True)
    episodes = episodes.rename(columns={"tconst": "episode_tconst"})

    ratings = pd.read_csv(ratings_path, sep="\t", dtype={"tconst": str, "averageRating": float, "numVotes": "Int64"})
    ratings = ratings.rename(columns={"tconst": "episode_tconst", "averageRating": "imdb_rating", "numVotes": "imdb_votes"})
    episodes = episodes.merge(ratings, on="episode_tconst", how="left")

    episode_ids = set(episodes["episode_tconst"].dropna().tolist())
    title_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(basics_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["tconst"].isin(episode_ids)]
        if chunk.empty:
            continue
        chunk = chunk[["tconst", "primaryTitle"]].rename(columns={"tconst": "episode_tconst", "primaryTitle": "episode_title"})
        title_chunks.append(chunk)

    if title_chunks:
        titles = pd.concat(title_chunks, ignore_index=True)
        episodes = episodes.merge(titles, on="episode_tconst", how="left")
    else:
        episodes["episode_title"] = None

    return episodes
