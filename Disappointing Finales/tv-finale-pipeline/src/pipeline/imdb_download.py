from __future__ import annotations

import logging
from pathlib import Path

import requests

from pipeline.config import Config, IMDB_DATASETS
from pipeline.io import ensure_dirs

logger = logging.getLogger(__name__)


def _download_file(url: str, destination: Path, timeout: int = 60) -> None:
    """Download a file with streaming to the destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_imdb_datasets(config: Config) -> list[Path]:
    """Download required IMDb datasets if they are missing."""
    ensure_dirs(config)
    downloaded: list[Path] = []
    for filename, url in IMDB_DATASETS.items():
        destination = config.imdb_dir / filename
        if destination.exists():
            logger.info("IMDb dataset exists, skipping: %s", destination)
            continue
        logger.info("Downloading IMDb dataset: %s", url)
        _download_file(url, destination)
        downloaded.append(destination)
    return downloaded
