from __future__ import annotations

import json
import logging
import uuid

import pandas as pd

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)


def normalize(
    config: Config,
    shows_df: pd.DataFrame | None = None,
    episodes_df: pd.DataFrame | None = None,
    mapping_df: pd.DataFrame | None = None,
    tvmaze_episodes_df: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create canonical show and episode tables with manual label support."""
    ensure_dirs(config)

    if config.shows_parquet.exists() and config.episodes_parquet.exists() and not overwrite:
        logger.info("Loading cached normalized tables")
        return read_parquet(config.shows_parquet), read_parquet(config.episodes_parquet)

    shows_df = shows_df if shows_df is not None else read_parquet(config.imdb_shows_raw_parquet)
    episodes_df = episodes_df if episodes_df is not None else read_parquet(config.imdb_episodes_raw_parquet)
    mapping_df = mapping_df if mapping_df is not None else read_parquet(config.mapping_parquet)
    tvmaze_episodes_df = tvmaze_episodes_df if tvmaze_episodes_df is not None else read_parquet(config.tvmaze_episodes_parquet)

    seed_df = pd.read_csv(config.seed_file, comment="#", dtype=str)

    shows = _build_shows(shows_df, mapping_df, seed_df, config)
    filtered_tconsts = set(shows["imdb_tconst"])
    mapping_df = mapping_df[mapping_df["imdb_tconst"].isin(filtered_tconsts)].copy()
    if not mapping_df.empty:
        write_parquet(mapping_df, config.mapping_parquet)

    episodes = _build_episodes(episodes_df, mapping_df, tvmaze_episodes_df, shows, config)

    write_parquet(shows, config.shows_parquet)
    write_parquet(episodes, config.episodes_parquet)

    return shows, episodes


def _build_shows(shows_df: pd.DataFrame, mapping_df: pd.DataFrame, seed_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = shows_df.merge(mapping_df, on="imdb_tconst", how="left")
    df = df.merge(seed_df[["imdb_tconst", "imdb_url"]], on="imdb_tconst", how="left")

    df["show_id"] = df["imdb_tconst"].apply(_stable_show_id)
    df["title"] = df["primaryTitle"].fillna(df["originalTitle"]).fillna("")

    df["start_year"] = pd.to_numeric(df.get("startYear"), errors="coerce").astype("Int64")
    imdb_end_raw = df.get("endYear")
    imdb_has_end = imdb_end_raw.notna() & (imdb_end_raw != "\\N") & (imdb_end_raw != "")
    df["end_year"] = pd.to_numeric(imdb_end_raw.where(imdb_has_end), errors="coerce").astype("Int64")

    df["genres"] = df.get("genres").fillna("").apply(_parse_genres)

    df["platform_raw"] = df["tvmaze_network"].fillna(df["tvmaze_web_channel"])
    df["platform_norm"] = df.apply(_normalize_platform, axis=1)

    df["ip_type_norm"] = "unknown"
    df["ending_type_norm"] = "unknown"
    df["lead_exit_before_final"] = "unknown"

    df["sources"] = df.apply(
        lambda row: json.dumps({"imdb_url": row.get("imdb_url"), "tvmaze_url": row.get("tvmaze_url")}),
        axis=1,
    )

    is_running = _is_running(imdb_has_end, df.get("tvmaze_status"))
    if is_running.any():
        logger.info("Filtering out running shows: %s", int(is_running.sum()))
        df = df[~is_running].copy()

    _ensure_labels_template(config)
    if config.labels_csv.exists():
        labels = pd.read_csv(config.labels_csv, dtype=str)
        df = _apply_manual_labels(df, labels)

    columns = [
        "show_id",
        "imdb_tconst",
        "title",
        "start_year",
        "end_year",
        "genres",
        "platform_raw",
        "platform_norm",
        "ip_type_norm",
        "ending_type_norm",
        "lead_exit_before_final",
        "sources",
    ]

    return df[columns]


def _build_episodes(
    episodes_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    tvmaze_episodes_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    episodes = episodes_df.merge(mapping_df[["imdb_tconst", "tvmaze_id"]], left_on="parentTconst", right_on="imdb_tconst", how="left")
    episodes = episodes.merge(
        shows_df[["show_id", "imdb_tconst"]],
        left_on="parentTconst",
        right_on="imdb_tconst",
        how="left",
    )
    episodes = episodes[episodes["show_id"].notna()].copy()

    episodes["seasonNumber"] = pd.to_numeric(episodes.get("seasonNumber"), errors="coerce").astype("Int64")
    episodes["episodeNumber"] = pd.to_numeric(episodes.get("episodeNumber"), errors="coerce").astype("Int64")
    episodes["season_number"] = episodes["seasonNumber"]
    episodes["episode_number"] = episodes["episodeNumber"]

    tvmaze = tvmaze_episodes_df.copy()
    if not tvmaze.empty:
        tvmaze["seasonNumber"] = pd.to_numeric(tvmaze.get("seasonNumber"), errors="coerce").astype("Int64")
        tvmaze["episodeNumber"] = pd.to_numeric(tvmaze.get("episodeNumber"), errors="coerce").astype("Int64")

    episodes = episodes.merge(
        tvmaze,
        on=["tvmaze_id", "seasonNumber", "episodeNumber"],
        how="left",
        suffixes=("_imdb", "_tvmaze"),
    )

    episodes["air_date"] = episodes["air_date"].where(episodes["air_date"].notna(), None)
    episodes["runtime_minutes"] = episodes["runtime_minutes"].where(episodes["runtime_minutes"].notna(), None)

    episodes["episode_title"] = episodes.get("episode_title_tvmaze").combine_first(episodes.get("episode_title_imdb"))

    episodes["has_tvmaze_match"] = episodes["air_date"].notna() | episodes["runtime_minutes"].notna()
    episodes["has_imdb_rating"] = episodes["imdb_rating"].notna()

    unmatched = _find_unmatched_tvmaze(tvmaze, episodes)
    if not unmatched.empty:
        write_parquet(unmatched, config.tvmaze_unmatched_parquet)

    columns = [
        "show_id",
        "season_number",
        "episode_number",
        "air_date",
        "runtime_minutes",
        "imdb_rating",
        "imdb_votes",
        "episode_title",
        "has_tvmaze_match",
        "has_imdb_rating",
    ]

    return episodes[columns]


def _stable_show_id(imdb_tconst: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"imdb:{imdb_tconst}"))


def _parse_genres(genres: str | None) -> list[str]:
    if not genres or genres == "\\N":
        return []
    return [g.strip() for g in genres.split(",") if g.strip()]


def _normalize_platform(row: pd.Series) -> str:
    if pd.notna(row.get("tvmaze_web_channel")):
        return "streaming"
    if pd.notna(row.get("tvmaze_network")):
        return "network"
    return "unknown"


def _is_running(imdb_has_end: pd.Series, tvmaze_status: pd.Series | None) -> pd.Series:
    if tvmaze_status is None:
        tvmaze_status = pd.Series([None] * len(imdb_has_end), index=imdb_has_end.index)
    return (~imdb_has_end) & (tvmaze_status != "Ended")


def _ensure_labels_template(config: Config) -> None:
    if config.labels_template_csv.exists():
        return
    config.labels_template_csv.parent.mkdir(parents=True, exist_ok=True)
    config.labels_template_csv.write_text(
        "imdb_tconst,platform_norm_override,ip_type_norm,ending_type_norm,lead_exit_before_final\n",
        encoding="utf-8",
    )


def _apply_manual_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.replace({"": None})
    merged = df.merge(labels, on="imdb_tconst", how="left", suffixes=("", "_manual"))

    if "platform_norm_override" in merged:
        merged["platform_norm"] = merged["platform_norm_override"].combine_first(merged["platform_norm"])

    for col in ["ip_type_norm", "ending_type_norm", "lead_exit_before_final"]:
        manual_col = f"{col}_manual"
        if manual_col in merged:
            merged[col] = merged[manual_col].combine_first(merged[col])

    return merged


def _find_unmatched_tvmaze(tvmaze: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if tvmaze.empty:
        return tvmaze

    matched_keys = episodes[["tvmaze_id", "seasonNumber", "episodeNumber"]].dropna().drop_duplicates()
    tvmaze_keys = tvmaze[["tvmaze_id", "seasonNumber", "episodeNumber"]].dropna().drop_duplicates()
    unmatched = tvmaze_keys.merge(matched_keys, on=["tvmaze_id", "seasonNumber", "episodeNumber"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    return tvmaze.merge(unmatched, on=["tvmaze_id", "seasonNumber", "episodeNumber"], how="inner")
