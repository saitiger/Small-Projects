from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from pipeline.config import Config

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    tvmaze_id: int | None
    tvmaze_name: str | None
    tvmaze_url: str | None
    match_score: float
    match_confidence: str
    chosen_reason: str


def score_candidate(title: str, start_year: str | None, candidate: dict[str, Any]) -> float:
    candidate_show = candidate.get("show", {}) if "show" in candidate else candidate
    cand_title = candidate_show.get("name") or ""
    cand_year = None
    premiered = candidate_show.get("premiered") or ""
    if premiered:
        cand_year = premiered.split("-")[0]

    title_score = fuzz.token_set_ratio(title, cand_title) / 100.0 if cand_title else 0.0

    year_score = 0.0
    if start_year and cand_year and start_year.isdigit() and cand_year.isdigit():
        diff = abs(int(start_year) - int(cand_year))
        if diff <= 1:
            year_score = 1.0
        elif diff <= 3:
            year_score = 0.5

    genres = set((candidate_show.get("genres") or []))
    genre_score = 0.0
    if genres:
        genre_score = 0.2

    return 0.7 * title_score + 0.2 * year_score + 0.1 * genre_score


def choose_best_match(title: str, start_year: str | None, candidates: list[dict[str, Any]], config: Config) -> MatchResult:
    """Select the best TVMaze candidate for a given show title."""
    if not candidates:
        return MatchResult(None, None, None, 0.0, "none", "no candidates")

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        score = score_candidate(title, start_year, candidate)
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    candidate_show = best_candidate.get("show", {}) if "show" in best_candidate else best_candidate

    if best_score < config.match_threshold:
        return MatchResult(None, None, None, best_score, "low", "below threshold")

    if best_score >= 0.9:
        confidence = "high"
    elif best_score >= 0.82:
        confidence = "medium"
    else:
        confidence = "low"

    return MatchResult(
        tvmaze_id=candidate_show.get("id"),
        tvmaze_name=candidate_show.get("name"),
        tvmaze_url=candidate_show.get("url"),
        match_score=best_score,
        match_confidence=confidence,
        chosen_reason="best fuzzy match",
    )
