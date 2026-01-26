from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_json, write_parquet
from pipeline.tvmaze_client import TVMazeClient
from pipeline.matching import choose_best_match
from pipeline.normalize import normalize
from pipeline.features import build_features
from pipeline.report import build_report

logger = logging.getLogger(__name__)


CURATED_SHOWS: list[dict[str, str]] = [
    {"group": "best", "title": "Breaking Bad"},
    {"group": "best", "title": "Six Feet Under"},
    {"group": "best", "title": "Fleabag"},
    {"group": "best", "title": "Angel"},
    {"group": "worst", "title": "Game of Thrones"},
    {"group": "worst", "title": "House of Cards"},
    {"group": "worst", "title": "Two and a Half Men"},
    {"group": "worst", "title": "Dexter"},
    {"group": "worst", "title": "Killing Eve"},
    {"group": "worst", "title": "How I Met Your Mother"},
    {"group": "worst", "title": "Lost"},
    {"group": "worst", "title": "Glee"},
    {"group": "worst", "title": "True Blood"},
    {"group": "worst", "title": "Scrubs"},
    {"group": "worst", "title": "Friends"},
    {"group": "worst", "title": "Modern Family"},
    {"group": "worst", "title": "The Crown"},
    {"group": "worst", "title": "The Office"},
    {"group": "worst", "title": "Will & Grace"},
    {"group": "worst", "title": "Sherlock"},
    {"group": "worst", "title": "Prison Break"},
    {"group": "worst", "title": "Supernatural"},
    {"group": "worst", "title": "Seinfeld"},
    {"group": "worst", "title": "House"},
    {"group": "worst", "title": "Once Upon a Time"},
    {"group": "worst", "title": "The Vampire Diaries"},
    {"group": "worst", "title": "Sex Education"},
    {"group": "worst", "title": "The 100"},
]


def ingest_curated_once(config: Config) -> None:
    """One-time ingestion of curated best/worst finales list."""
    ensure_dirs(config)

    curated = _dedupe_curated(CURATED_SHOWS)
    curated_groups = _build_group_map(curated)

    shows = read_parquet(config.shows_parquet)
    episodes = read_parquet(config.episodes_parquet)
    features = read_parquet(config.features_parquet)
    mapping = read_parquet(config.mapping_parquet)

    existing_titles = shows["title"].fillna("").apply(_normalize_title)
    existing_by_title = dict(zip(existing_titles, shows["imdb_tconst"]))
    existing_tconsts = set(shows["imdb_tconst"].dropna().tolist())

    unresolved_rows: list[dict[str, str]] = []
    resolved_rows: list[dict[str, Any]] = []

    logger.info("Resolving curated list against existing dataset")
    basics_path = config.imdb_dir / "title.basics.tsv.gz"
    ratings_path = config.imdb_dir / "title.ratings.tsv.gz"
    if not basics_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("IMDb datasets missing. Run imdb-download first.")

    ratings = pd.read_csv(ratings_path, sep="\t", dtype={"tconst": str, "numVotes": "Int64"})
    ratings = ratings.rename(columns={"tconst": "imdb_tconst"})
    ratings_map = ratings.set_index("imdb_tconst")["numVotes"].to_dict()

    candidates = _find_imdb_candidates(basics_path, curated)

    for item in curated:
        title = item["title"]
        group = item["group"]
        normalized = _normalize_title(title)

        if normalized in existing_by_title:
            tconst = existing_by_title[normalized]
            resolved_rows.append({"group": group, "title": title, "imdb_tconst": tconst, "status": "already_present"})
            continue

        candidate_list = candidates.get(normalized, [])
        if not candidate_list:
            unresolved_rows.append({"group": group, "title": title, "reason": "no_imdb_match", "details": "no candidates"})
            continue

        best = _pick_best_candidate(candidate_list, ratings_map)
        if best is None:
            unresolved_rows.append({"group": group, "title": title, "reason": "no_imdb_match", "details": "no ranked candidate"})
            continue

        resolved_rows.append({"group": group, "title": title, "imdb_tconst": best["tconst"], "status": "resolved"})

    resolved_df = pd.DataFrame(resolved_rows)
    if resolved_df.empty:
        _write_unresolved(config, unresolved_rows)
        write_json(config.curated_ingest_report_json, _report_counts(curated, resolved_df, unresolved_rows, 0))
        return

    new_tconsts = set(resolved_df.loc[resolved_df["status"] == "resolved", "imdb_tconst"].tolist())
    new_tconsts = new_tconsts - existing_tconsts

    running_excluded = 0
    if new_tconsts:
        new_shows = _load_imdb_shows(basics_path, new_tconsts)
        new_shows = _filter_running_shows(new_shows, mapping)
        running_excluded = int((~new_shows["is_included"]).sum())
        excluded_titles = new_shows.loc[~new_shows["is_included"], "title"].tolist()
        for title in excluded_titles:
            group = curated_groups.get(_normalize_title(title), "unknown")
            unresolved_rows.append({"group": group, "title": title, "reason": "running_show_excluded", "details": "imdb_endYear_missing_or_running"})
        new_shows = new_shows[new_shows["is_included"]].drop(columns=["is_included"])
        new_tconsts = set(new_shows["imdb_tconst"].tolist())
    else:
        new_shows = pd.DataFrame()

    if new_tconsts:
        new_episodes = _load_imdb_episodes(config, new_tconsts)
        new_shows_raw = _load_imdb_shows_raw(basics_path, new_tconsts)

        mapping_updated, tvmaze_episodes_updated, tvmaze_unmapped = _tvmaze_enrich_new(config, new_shows_raw, mapping)
        mapping = mapping_updated
        if tvmaze_unmapped:
            unresolved_rows.extend(tvmaze_unmapped)

        imdb_shows_raw = _merge_tables(config.imdb_shows_raw_parquet, new_shows_raw, ["imdb_tconst"])
        imdb_episodes_raw = _merge_tables(config.imdb_episodes_raw_parquet, new_episodes, ["parentTconst", "episode_tconst"])
        write_parquet(imdb_shows_raw, config.imdb_shows_raw_parquet)
        write_parquet(imdb_episodes_raw, config.imdb_episodes_raw_parquet)

        shows, episodes = normalize(config, overwrite=True)
        features = build_features(config, shows_df=shows, episodes_df=episodes, overwrite=True)
        build_report(config, shows_df=shows, episodes_df=episodes, mapping_df=mapping, overwrite=True)

        shows = _apply_curated_group(shows, curated_groups)
        features = features.merge(shows[["show_id", "curated_group"]], on="show_id", how="left")
        write_parquet(shows, config.shows_parquet)
        write_parquet(episodes, config.episodes_parquet)
        write_parquet(features, config.features_parquet)
    else:
        shows = _apply_curated_group(shows, curated_groups)
        if not features.empty:
            features = features.merge(shows[["show_id", "curated_group"]], on="show_id", how="left")
        write_parquet(shows, config.shows_parquet)
        write_parquet(features, config.features_parquet)

    _write_unresolved(config, unresolved_rows)
    tvmaze_unmapped_count = int(sum(1 for row in unresolved_rows if row.get("reason") == "tvmaze_unmapped"))
    write_json(
        config.curated_ingest_report_json,
        _report_counts(curated, resolved_df, unresolved_rows, tvmaze_unmapped_count, running_excluded),
    )


def _dedupe_curated(curated: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    results: list[dict[str, str]] = []
    for item in curated:
        title = item["title"].strip()
        key = _normalize_title(title)
        if key in seen:
            continue
        seen.add(key)
        results.append({"group": item["group"], "title": title})
    return results


def _build_group_map(curated: list[dict[str, str]]) -> dict[str, str]:
    groups = defaultdict(set)
    for item in curated:
        key = _normalize_title(item["title"])
        groups[key].add(item["group"])

    result: dict[str, str] = {}
    for key, group_set in groups.items():
        if len(group_set) == 1:
            result[key] = next(iter(group_set))
        else:
            result[key] = "both"
    return result


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _find_imdb_candidates(basics_path: Path, curated: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    targets = {_normalize_title(item["title"]) for item in curated}
    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in targets}

    for chunk in pd.read_csv(basics_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["titleType"].isin(["tvSeries", "tvMiniSeries"])]
        chunk["normalized_primary"] = chunk["primaryTitle"].fillna("").apply(_normalize_title)
        chunk["normalized_original"] = chunk["originalTitle"].fillna("").apply(_normalize_title)

        mask = chunk["normalized_primary"].isin(targets) | chunk["normalized_original"].isin(targets)
        if not mask.any():
            continue

        matched = chunk[mask]
        for _, row in matched.iterrows():
            for key in {row["normalized_primary"], row["normalized_original"]} & targets:
                candidates[key].append(
                    {
                        "tconst": row["tconst"],
                        "primaryTitle": row["primaryTitle"],
                        "originalTitle": row["originalTitle"],
                        "startYear": row["startYear"],
                        "endYear": row["endYear"],
                    }
                )

    return candidates


def _pick_best_candidate(candidates: list[dict[str, Any]], ratings_map: dict[str, Any]) -> dict[str, Any] | None:
    if not candidates:
        return None

    ranked = []
    for candidate in candidates:
        votes = ratings_map.get(candidate["tconst"], 0)
        ranked.append((votes if votes is not None else 0, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else None


def _load_imdb_shows(basics_path: Path, tconsts: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(basics_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["tconst"].isin(tconsts)]
        if chunk.empty:
            continue
        chunk = chunk[["tconst", "primaryTitle", "originalTitle", "startYear", "endYear", "genres"]]
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["imdb_tconst", "primaryTitle", "originalTitle", "startYear", "endYear", "genres"])
    shows = pd.concat(rows, ignore_index=True)
    return shows.rename(columns={"tconst": "imdb_tconst"})


def _load_imdb_shows_raw(basics_path: Path, tconsts: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(basics_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["tconst"].isin(tconsts)]
        if chunk.empty:
            continue
        chunk = chunk[["tconst", "primaryTitle", "originalTitle", "startYear", "endYear", "genres"]]
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["imdb_tconst", "primaryTitle", "originalTitle", "startYear", "endYear", "genres"])
    shows = pd.concat(rows, ignore_index=True)
    return shows.rename(columns={"tconst": "imdb_tconst"})


def _load_imdb_episodes(config: Config, tconsts: set[str]) -> pd.DataFrame:
    episode_path = config.imdb_dir / "title.episode.tsv.gz"
    ratings_path = config.imdb_dir / "title.ratings.tsv.gz"
    basics_path = config.imdb_dir / "title.basics.tsv.gz"

    episode_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(episode_path, sep="\t", dtype=str, chunksize=200_000):
        chunk = chunk[chunk["parentTconst"].isin(tconsts)]
        if chunk.empty:
            continue
        chunk = chunk[["parentTconst", "tconst", "seasonNumber", "episodeNumber"]]
        episode_chunks.append(chunk)

    if not episode_chunks:
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


def _tvmaze_enrich_new(
    config: Config,
    shows_df: pd.DataFrame,
    existing_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    client = TVMazeClient(config)
    mapping_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []

    for _, show in tqdm(shows_df.iterrows(), total=len(shows_df), desc="TVMaze curated"):
        title = show.get("primaryTitle") or show.get("originalTitle") or ""
        start_year = show.get("startYear")
        imdb_tconst = show.get("imdb_tconst")

        try:
            candidates = client.search_show(title)
            match = choose_best_match(title, start_year, candidates, config)
            if match.tvmaze_id is None:
                unresolved.append({"group": "unknown", "title": title, "reason": "tvmaze_unmapped", "details": "no match"})
                mapping_rows.append(
                    {
                        "imdb_tconst": imdb_tconst,
                        "tvmaze_id": None,
                        "tvmaze_name": None,
                        "tvmaze_url": None,
                        "match_score": match.match_score,
                        "match_confidence": match.match_confidence,
                        "chosen_reason": match.chosen_reason,
                        "match_error": None,
                        "tvmaze_genres": None,
                        "tvmaze_network": None,
                        "tvmaze_web_channel": None,
                        "tvmaze_premiered": None,
                        "tvmaze_ended": None,
                        "tvmaze_status": None,
                        "tvmaze_type": None,
                    }
                )
                continue

            best = next((c for c in candidates if (c.get("show") or c).get("id") == match.tvmaze_id), None)
            show_info = best.get("show", {}) if isinstance(best, dict) else {}
            mapping_rows.append(
                {
                    "imdb_tconst": imdb_tconst,
                    "tvmaze_id": match.tvmaze_id,
                    "tvmaze_name": show_info.get("name"),
                    "tvmaze_url": show_info.get("url"),
                    "match_score": match.match_score,
                    "match_confidence": match.match_confidence,
                    "chosen_reason": match.chosen_reason,
                    "match_error": None,
                    "tvmaze_genres": show_info.get("genres"),
                    "tvmaze_network": (show_info.get("network") or {}).get("name"),
                    "tvmaze_web_channel": (show_info.get("webChannel") or {}).get("name"),
                    "tvmaze_premiered": show_info.get("premiered"),
                    "tvmaze_ended": show_info.get("ended"),
                    "tvmaze_status": show_info.get("status"),
                    "tvmaze_type": show_info.get("type"),
                }
            )

            episodes = client.get_episodes(match.tvmaze_id)
            for episode in episodes:
                episode_rows.append(
                    {
                        "tvmaze_id": match.tvmaze_id,
                        "seasonNumber": episode.get("season"),
                        "episodeNumber": episode.get("number"),
                        "air_date": episode.get("airdate"),
                        "runtime_minutes": episode.get("runtime"),
                        "episode_title": episode.get("name"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            mapping_rows.append(
                {
                    "imdb_tconst": imdb_tconst,
                    "tvmaze_id": None,
                    "tvmaze_name": None,
                    "tvmaze_url": None,
                    "match_score": 0.0,
                    "match_confidence": "none",
                    "chosen_reason": "error",
                    "match_error": str(exc),
                    "tvmaze_genres": None,
                    "tvmaze_network": None,
                    "tvmaze_web_channel": None,
                    "tvmaze_premiered": None,
                    "tvmaze_ended": None,
                    "tvmaze_status": None,
                    "tvmaze_type": None,
                }
            )
            unresolved.append({"group": "unknown", "title": title, "reason": "tvmaze_error", "details": str(exc)})

    mapping_new = pd.DataFrame(mapping_rows)
    tvmaze_new = pd.DataFrame(
        episode_rows,
        columns=["tvmaze_id", "seasonNumber", "episodeNumber", "air_date", "runtime_minutes", "episode_title"],
    )

    mapping_combined = pd.concat([existing_mapping, mapping_new], ignore_index=True).drop_duplicates(subset=["imdb_tconst"], keep="last")
    tvmaze_existing = read_parquet(config.tvmaze_episodes_parquet) if config.tvmaze_episodes_parquet.exists() else pd.DataFrame()
    tvmaze_combined = pd.concat([tvmaze_existing, tvmaze_new], ignore_index=True).drop_duplicates(subset=["tvmaze_id", "seasonNumber", "episodeNumber"], keep="last")

    write_parquet(mapping_combined, config.mapping_parquet)
    write_parquet(tvmaze_combined, config.tvmaze_episodes_parquet)

    return mapping_combined, tvmaze_combined, unresolved


def _filter_running_shows(shows: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if shows.empty:
        return shows.assign(is_included=False)

    shows["imdb_endYear"] = shows.get("endYear")
    imdb_has_end = shows["imdb_endYear"].notna() & (shows["imdb_endYear"] != "\\N") & (shows["imdb_endYear"] != "")
    merged = shows.merge(mapping[["imdb_tconst", "tvmaze_status"]], on="imdb_tconst", how="left")
    is_running = (~imdb_has_end) & (merged["tvmaze_status"] != "Ended")
    merged["is_included"] = ~is_running
    merged["title"] = merged["primaryTitle"].fillna(merged["originalTitle"]).fillna("")
    return merged


def _apply_curated_group(shows: pd.DataFrame, curated_groups: dict[str, str]) -> pd.DataFrame:
    shows = shows.copy()
    shows["curated_group"] = shows["title"].fillna("").apply(lambda t: curated_groups.get(_normalize_title(t)))
    return shows


def _merge_tables(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        existing = read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    return combined.drop_duplicates(subset=key_cols, keep="last")


def _write_unresolved(config: Config, unresolved_rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(unresolved_rows, columns=["group", "title", "reason", "details"])
    df.to_csv(config.curated_unresolved_csv, index=False)


def _report_counts(
    curated: list[dict[str, str]],
    resolved_df: pd.DataFrame,
    unresolved_rows: list[dict[str, str]],
    tvmaze_unmapped: int,
    running_excluded: int = 0,
) -> dict[str, Any]:
    already_present = int((resolved_df["status"] == "already_present").sum()) if not resolved_df.empty else 0
    added = int((resolved_df["status"] == "resolved").sum()) if not resolved_df.empty else 0
    unresolved = len(unresolved_rows)
    return {
        "counts": {
            "requested": len(curated),
            "already_present": already_present,
            "added": added,
            "unresolved": unresolved,
            "running_excluded": running_excluded,
            "tvmaze_unmapped": tvmaze_unmapped,
        }
    }
