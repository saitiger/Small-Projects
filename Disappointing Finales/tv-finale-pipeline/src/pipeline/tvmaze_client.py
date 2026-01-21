from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from pipeline.config import Config
from pipeline.io import ensure_dirs, read_parquet, write_parquet
from pipeline.matching import score_candidate, choose_best_match

logger = logging.getLogger(__name__)


class TVMazeClient:
    """TVMaze client with disk caching and rate limiting."""
    def __init__(self, config: Config) -> None:
        self.cache_dir = config.tvmaze_cache_dir
        self.rate_limit_seconds = config.tvmaze_rate_limit_seconds
        self.timeout = config.tvmaze_timeout_seconds
        self.max_retries = config.tvmaze_max_retries
        self._last_request_time: float | None = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cache_key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", cache_key)[:200]
        return self.cache_dir / f"{safe_key}.json"

    def get_json(self, url: str, params: dict[str, Any] | None = None, cache_key: str | None = None) -> Any:
        cache_path = self._cache_path(cache_key or re.sub(r"[^a-zA-Z0-9_-]+", "_", url))
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - elapsed)

        for attempt in range(1, self.max_retries + 1):
            response = requests.get(url, params=params, timeout=self.timeout)
            self._last_request_time = time.time()

            if response.status_code == 429 or response.status_code >= 500:
                backoff = 1.5 * attempt
                time.sleep(backoff)
                continue

            response.raise_for_status()
            data = response.json()
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            return data

        raise RuntimeError(f"TVMaze request failed after {self.max_retries} attempts: {url}")

    def search_show(self, title: str) -> list[dict[str, Any]]:
        url = "https://api.tvmaze.com/search/shows"
        cache_key = f"search_{title.lower()}"
        return self.get_json(url, params={"q": title}, cache_key=cache_key)

    def get_episodes(self, tvmaze_id: int) -> list[dict[str, Any]]:
        url = f"https://api.tvmaze.com/shows/{tvmaze_id}/episodes"
        cache_key = f"episodes_{tvmaze_id}"
        return self.get_json(url, cache_key=cache_key)


def tvmaze_enrich(config: Config, shows_df: pd.DataFrame | None = None, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match IMDb shows to TVMaze and fetch episode metadata."""
    ensure_dirs(config)
    if config.mapping_parquet.exists() and config.tvmaze_episodes_parquet.exists() and not overwrite:
        logger.info("Loading cached TVMaze outputs")
        return read_parquet(config.mapping_parquet), read_parquet(config.tvmaze_episodes_parquet)

    if shows_df is None:
        if not config.imdb_shows_raw_parquet.exists():
            raise FileNotFoundError("IMDb shows table missing; run build-imdb.")
        shows_df = read_parquet(config.imdb_shows_raw_parquet)

    client = TVMazeClient(config)
    mapping_rows: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []

    for _, show in tqdm(shows_df.iterrows(), total=len(shows_df), desc="TVMaze enrich"):
        title = show.get("primaryTitle") or show.get("originalTitle") or ""
        start_year = show.get("startYear")

        row = {
            "imdb_tconst": show.get("imdb_tconst"),
            "tvmaze_id": None,
            "tvmaze_name": None,
            "tvmaze_url": None,
            "match_score": 0.0,
            "match_confidence": "none",
            "chosen_reason": "no candidates",
            "match_error": None,
            "tvmaze_genres": None,
            "tvmaze_network": None,
            "tvmaze_web_channel": None,
            "tvmaze_premiered": None,
            "tvmaze_ended": None,
            "tvmaze_status": None,
            "tvmaze_type": None,
        }

        try:
            candidates = client.search_show(title)
            match = choose_best_match(title, start_year, candidates, config)
            best_candidate = _select_best_candidate(title, start_year, candidates)

            row.update(asdict(match))

            if match.tvmaze_id is None:
                mapping_rows.append(row)
                continue

            show_info = best_candidate.get("show", {}) if best_candidate else {}
            row.update(
                {
                    "tvmaze_id": match.tvmaze_id,
                    "tvmaze_name": show_info.get("name"),
                    "tvmaze_url": show_info.get("url"),
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
                episodes_rows.append(
                    {
                        "tvmaze_id": match.tvmaze_id,
                        "seasonNumber": episode.get("season"),
                        "episodeNumber": episode.get("number"),
                        "air_date": episode.get("airdate"),
                        "runtime_minutes": episode.get("runtime"),
                        "episode_title": episode.get("name"),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - keep pipeline resilient
            row["match_error"] = str(exc)
            mapping_rows.append(row)
            logger.warning("TVMaze enrich failed for %s: %s", title, exc)
            continue

        mapping_rows.append(row)

    mapping_df = pd.DataFrame(mapping_rows)
    tvmaze_episodes_df = pd.DataFrame(
        episodes_rows,
        columns=[
            "tvmaze_id",
            "seasonNumber",
            "episodeNumber",
            "air_date",
            "runtime_minutes",
            "episode_title",
        ],
    )

    write_parquet(mapping_df, config.mapping_parquet)
    write_parquet(tvmaze_episodes_df, config.tvmaze_episodes_parquet)

    return mapping_df, tvmaze_episodes_df


def _select_best_candidate(title: str, start_year: str | None, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    best_score = -1.0
    best_candidate: dict[str, Any] | None = None
    for candidate in candidates:
        score = score_candidate(title, start_year, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate
