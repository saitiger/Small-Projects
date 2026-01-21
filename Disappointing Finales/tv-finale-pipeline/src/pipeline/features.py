from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)


def build_features(config: Config, shows_df: pd.DataFrame | None = None, episodes_df: pd.DataFrame | None = None, overwrite: bool = False) -> pd.DataFrame:
    """Compute show-level features from canonical episodes."""
    ensure_dirs(config)
    if config.features_parquet.exists() and not overwrite:
        logger.info("Loading cached features")
        return read_parquet(config.features_parquet)

    shows_df = shows_df if shows_df is not None else read_parquet(config.shows_parquet)
    episodes_df = episodes_df if episodes_df is not None else read_parquet(config.episodes_parquet)

    features = _compute_features(config, episodes_df)
    features = features.merge(shows_df[["show_id"]], on="show_id", how="right")

    write_parquet(features, config.features_parquet)
    return features


def _compute_features(config: Config, episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    df = episodes.copy()
    df["air_date"] = pd.to_datetime(df["air_date"], errors="coerce")
    df["runtime_minutes"] = pd.to_numeric(df["runtime_minutes"], errors="coerce")
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")

    group = df.groupby("show_id", dropna=False)

    total_episodes = group.size().rename("total_episodes")
    missing_runtime_pct = group["runtime_minutes"].apply(lambda s: s.isna().mean()).rename("missing_runtime_pct")
    missing_airdate_pct = group["air_date"].apply(lambda s: s.isna().mean()).rename("missing_airdate_pct")
    missing_ratings_pct = group["imdb_rating"].apply(lambda s: s.isna().mean()).rename("missing_ratings_pct")

    median_runtime = group["runtime_minutes"].median()
    global_median_runtime = df["runtime_minutes"].median()

    def total_runtime(series: pd.Series) -> float:
        if series.notna().any():
            return series.fillna(series.median()).sum()
        return series.fillna(global_median_runtime).sum()

    total_runtime_minutes = group["runtime_minutes"].apply(total_runtime).rename("total_runtime_minutes")
    avg_episode_runtime = (total_runtime_minutes / total_episodes).rename("avg_episode_runtime")

    series_avg_rating = group["imdb_rating"].mean().rename("series_avg_episode_rating")

    final_season_number = group["season_number"].max().rename("final_season_number")

    final_season_avg_rating = (
        df.merge(final_season_number, on="show_id", how="left")
        .query("season_number == final_season_number")
        .groupby("show_id")["imdb_rating"]
        .mean()
        .rename("final_season_avg_rating")
    )

    finale_info = _identify_finale(df)
    finale_rating = finale_info["finale_episode_rating"]
    finale_date = finale_info["finale_air_date"]

    finale_delta = (finale_rating - series_avg_rating).rename("finale_delta")
    closure_deficit = (finale_rating - final_season_avg_rating).rename("closure_deficit")

    final_season_premiere_date = (
        df.merge(final_season_number, on="show_id", how="left")
        .query("season_number == final_season_number")
        .groupby("show_id")["air_date"]
        .min()
        .rename("final_season_premiere_date")
    )

    penultimate_season_finale_date = (
        df.merge(final_season_number, on="show_id", how="left")
        .assign(penultimate=lambda x: x["final_season_number"] - 1)
        .query("season_number == penultimate")
        .groupby("show_id")["air_date"]
        .max()
        .rename("penultimate_season_finale_date")
    )

    gap_days = (
        (final_season_premiere_date - penultimate_season_finale_date)
        .dt.days
        .rename("gap_days_to_final_season")
    )

    era_bucket = finale_date.apply(lambda date: _bucket_era(date, config)).rename("era_bucket")
    runway_bucket = total_runtime_minutes.apply(lambda minutes: _bucket_runway(minutes, config)).rename("runway_bucket")

    features = pd.concat(
        [
            total_episodes,
            total_runtime_minutes,
            avg_episode_runtime,
            series_avg_rating,
            final_season_number,
            final_season_avg_rating,
            finale_rating,
            finale_delta,
            closure_deficit,
            final_season_premiere_date,
            penultimate_season_finale_date,
            gap_days,
            era_bucket,
            runway_bucket,
            missing_runtime_pct,
            missing_airdate_pct,
            missing_ratings_pct,
            finale_info["finale_id_method"],
        ],
        axis=1,
    ).reset_index()

    return features


def _identify_finale(df: pd.DataFrame) -> pd.DataFrame:
    def pick_finale(group: pd.DataFrame) -> pd.Series:
        group = group.copy()
        group = group.sort_values(["air_date", "season_number", "episode_number"], ascending=True)

        if group["air_date"].notna().any():
            finale = group.dropna(subset=["air_date"]).iloc[-1]
            method = "air_date"
        else:
            finale = group.sort_values(["season_number", "episode_number"]).iloc[-1]
            method = "season_episode"

        return pd.Series(
            {
                "finale_episode_rating": finale.get("imdb_rating"),
                "finale_air_date": finale.get("air_date"),
                "finale_id_method": method,
            }
        )

    subset = df[["show_id", "air_date", "season_number", "episode_number", "imdb_rating"]]
    records = []
    for show_id, group in subset.groupby("show_id"):
        finale = pick_finale(group.drop(columns=["show_id"]))
        finale["show_id"] = show_id
        records.append(finale)

    result = pd.DataFrame(records).set_index("show_id")
    result["finale_id_method"] = result["finale_id_method"].astype(str)
    return result


def _bucket_era(date: pd.Timestamp, config: Config) -> str:
    if pd.isna(date):
        return "unknown"
    pandemic_start = datetime.fromisoformat(config.pandemic_start).date()
    pandemic_end = datetime.fromisoformat(config.pandemic_end).date()
    post_start = datetime.fromisoformat(config.post_pandemic_start).date()

    date_val = date.date()
    if date_val < pandemic_start:
        return "pre_pandemic"
    if pandemic_start <= date_val <= pandemic_end:
        return "pandemic"
    if date_val >= post_start:
        return "post_pandemic"
    return "unknown"


def _bucket_runway(minutes: float, config: Config) -> str:
    if pd.isna(minutes):
        return "unknown"
    if minutes < config.runway_bucket_low:
        return "short"
    if minutes <= config.runway_bucket_high:
        return "medium"
    return "long"
