from __future__ import annotations

import csv
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from pipeline.config import Config
from pipeline.io import ensure_dirs, write_csv_with_comment

logger = logging.getLogger(__name__)


def seed_top100(config: Config, input_csv: Path | None = None, force: bool = False) -> Path:
    """Create or reuse the IMDb Top 100 seed list."""
    ensure_dirs(config)
    target = config.seed_file

    if input_csv is not None:
        logger.info("Using provided seed file: %s", input_csv)
        target.write_text(input_csv.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    if target.exists() and not force:
        logger.info("Seed file exists, skipping: %s", target)
        return target

    try:
        rows = _scrape_imdb_top100()
        provenance = f"Seeded from IMDb Top Rated TV Series page on {datetime.utcnow().isoformat()}Z"
        _write_seed_csv(target, rows, provenance)
        return target
    except Exception as exc:  # noqa: BLE001 - fallback is intentional
        logger.warning("IMDb scraping failed (%s); falling back to IMDb datasets.", exc)

    rows = _fallback_from_imdb_datasets(config)
    provenance = (
        "Seeded from IMDb public datasets by ranking tvSeries using rating * log10(votes). "
        f"Generated on {datetime.utcnow().isoformat()}Z"
    )
    _write_seed_csv(target, rows, provenance)
    return target


def _write_seed_csv(target: Path, rows: Iterable[dict[str, str | int | None]], provenance: str) -> None:
    fieldnames = ["rank", "title", "imdb_url", "imdb_tconst", "year"]
    write_csv_with_comment(target, provenance, rows, fieldnames)
    logger.info("Wrote seed file: %s", target)


def _scrape_imdb_top100() -> list[dict[str, str | int | None]]:
    url = "https://www.imdb.com/chart/toptv/"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200 or not response.text:
        raise RuntimeError(f"IMDb page blocked with status {response.status_code}")

    html = response.text
    rows = _parse_imdb_html(html)
    if not rows:
        raise RuntimeError("Failed to parse IMDb Top Rated TV list")
    return rows[:100]


def _parse_imdb_html(html: str) -> list[dict[str, str | int | None]]:
    rows: list[dict[str, str | int | None]] = []

    json_ld_match = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
    if json_ld_match:
        data = json.loads(json_ld_match.group(1))
        items = data.get("itemListElement", [])
        for item in items:
            entry = item.get("item", {}) if isinstance(item, dict) else {}
            url = entry.get("url")
            title = entry.get("name")
            if not url:
                continue
            tconst_match = re.search(r"/title/(tt\d+)/", url)
            if not tconst_match:
                continue
            rows.append(
                {
                    "rank": int(item.get("position") or len(rows) + 1),
                    "title": title or "",
                    "imdb_url": url,
                    "imdb_tconst": tconst_match.group(1),
                    "year": None,
                }
            )

    if rows:
        return rows

    next_data_match = re.search(r'__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    if not next_data_match:
        return rows

    data = json.loads(next_data_match.group(1))
    chart_titles = _find_nested_key(data, "chartTitles")
    if not isinstance(chart_titles, dict):
        return rows

    edges = chart_titles.get("edges") or chart_titles.get("titles")
    if not isinstance(edges, list):
        return rows

    for idx, edge in enumerate(edges, start=1):
        node = edge.get("node") if isinstance(edge, dict) else {}
        tconst = node.get("id") or node.get("titleId")
        title = node.get("titleText") or node.get("title")
        if isinstance(title, dict):
            title = title.get("text")
        year = None
        release_year = node.get("releaseYear") or {}
        if isinstance(release_year, dict):
            year = release_year.get("year")
        if tconst and title:
            rows.append(
                {
                    "rank": idx,
                    "title": str(title),
                    "imdb_url": f"https://www.imdb.com/title/{tconst}/",
                    "imdb_tconst": tconst,
                    "year": year,
                }
            )

    return rows


def _find_nested_key(data: object, target: str) -> object | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target:
                return value
            found = _find_nested_key(value, target)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_nested_key(item, target)
            if found is not None:
                return found
    return None


def _fallback_from_imdb_datasets(config: Config) -> list[dict[str, str | int | None]]:
    basics_path = config.imdb_dir / "title.basics.tsv.gz"
    ratings_path = config.imdb_dir / "title.ratings.tsv.gz"
    if not basics_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("IMDb datasets missing. Run imdb-download first or provide a seed CSV.")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        dtype={"tconst": str, "averageRating": float, "numVotes": int},
    )
    ratings = ratings[ratings["numVotes"] > 5000].copy()
    ratings["score"] = ratings["averageRating"] * ratings["numVotes"].apply(lambda v: math.log10(v))

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        basics_path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
    ):
        chunk = chunk[(chunk["titleType"] == "tvSeries") & (chunk["isAdult"] == "0")]
        chunk = chunk[["tconst", "primaryTitle", "startYear"]]
        chunks.append(chunk)
    basics = pd.concat(chunks, ignore_index=True)

    merged = basics.merge(ratings, left_on="tconst", right_on="tconst", how="inner")
    merged = merged.sort_values(["score", "numVotes"], ascending=False).head(100)

    rows: list[dict[str, str | int | None]] = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        rows.append(
            {
                "rank": idx + 1,
                "title": row["primaryTitle"],
                "imdb_url": f"https://www.imdb.com/title/{row['tconst']}/",
                "imdb_tconst": row["tconst"],
                "year": None if row["startYear"] in ("\\N", None) else int(row["startYear"]),
            }
        )

    return rows
