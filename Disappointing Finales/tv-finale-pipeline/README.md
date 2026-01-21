# TV Finale Pipeline

This repo builds an analysis-ready dataset for research on TV-series finales. It seeds from IMDb's Top Rated TV list, enriches with IMDb public TSVs + TVMaze, and outputs canonical tables plus derived features.

## What it does
- Seeds the Top 100 list (ranked) with IMDb tconsts and titles.
- Downloads and parses IMDb datasets for shows, episodes, and ratings.
- Matches each show to TVMaze with fuzzy title/year matching and caches API calls.
- Builds canonical tables: `shows.parquet`, `episodes.parquet`, `features.parquet`.
- Generates a JSON quality report with missingness and mapping coverage.

## Quickstart
1) Install dependencies

```bash
python -m pip install -e .
```

2) Run the full pipeline

```bash
PYTHONPATH=src python -m pipeline.cli run-all
```

Outputs land in `data/processed/`.

## Outputs
- `data/processed/shows.parquet` - one row per show (canonical).
- `data/processed/episodes.parquet` - one row per episode (canonical + ratings + airdate + runtime).
- `data/processed/features.parquet` - derived show-level metrics.
- `data/processed/mapping.parquet` - IMDb to TVMaze matching results.
- `data/processed/quality_report.json` - counts and warnings.

## Manual labels
A template is provided at `data/manual/labels_template.csv`. Create `data/manual/labels.csv` with:

- `imdb_tconst`
- `platform_norm_override`
- `ip_type_norm`
- `ending_type_norm`
- `lead_exit_before_final`

These values override defaults during the `normalize` stage.

## Known limitations
- IMDb's Top Rated TV page may block scraping. If so, the pipeline falls back to an IMDb-dataset-derived Top 100 list (not identical to the website ranking). You can also drop in your own seed CSV.
- Season/episode matching uses season + episode numbers, which may miss specials.
- Mapping ambiguity exists for shows with title collisions or remakes.

## CLI
Run individual stages:

```bash
PYTHONPATH=src python -m pipeline.cli seed-top100
PYTHONPATH=src python -m pipeline.cli imdb-download
PYTHONPATH=src python -m pipeline.cli build-imdb
PYTHONPATH=src python -m pipeline.cli tvmaze-enrich
PYTHONPATH=src python -m pipeline.cli normalize
PYTHONPATH=src python -m pipeline.cli features
PYTHONPATH=src python -m pipeline.cli report
```

Seed CSV expected columns:
`rank,title,imdb_url,imdb_tconst,year`
