from __future__ import annotations

import logging

try:
    from google.cloud import bigquery
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery"
    ) from exc

from bq_migrate.config import BQConfig

logger = logging.getLogger(__name__)


def create_views(client: bigquery.Client, config: BQConfig) -> list[str]:
    dataset = f"{config.project_id}.{config.dataset_id}"

    view_sql = _finale_analysis_sql(dataset)
    _create_view(client, f"{dataset}.v_finale_analysis", view_sql)

    rankings_sql = _rankings_sql(dataset)
    _create_view(client, f"{dataset}.v_finale_rankings", rankings_sql)

    top_bottom_sql = _top_bottom_sql(dataset)
    _create_view(client, f"{dataset}.v_top_bottom", top_bottom_sql)

    return ["v_finale_analysis", "v_finale_rankings", "v_top_bottom"]


def _create_view(client: bigquery.Client, view_id: str, query: str) -> None:
    view = bigquery.Table(view_id)
    view.view_query = query
    view.view_use_legacy_sql = False
    client.create_table(view, exists_ok=True)
    logger.info("Created view %s", view_id)


def _finale_analysis_sql(dataset: str) -> str:
    return f"""
    SELECT
      f.show_id,
      f.imdb_tconst,
      f.title,
      s.start_year,
      s.end_year,
      s.genres,
      s.platform_norm,
      f.total_episodes,
      f.total_runtime_minutes,
      f.series_avg_episode_rating,
      f.final_season_number,
      f.final_season_avg_rating,
      f.finale_episode_rating,
      f.finale_delta,
      f.closure_deficit,
      SAFE_CAST(f.final_season_premiere_date AS DATE) AS final_season_premiere_date,
      SAFE_CAST(f.penultimate_season_finale_date AS DATE) AS penultimate_season_finale_date,
      f.gap_days_to_final_season,
      f.era_bucket,
      f.runway_bucket,
      f.tvmaze_match_rate,
      f.data_quality_score,
      f.finale_id_method,
      f.finale_is_true_series_finale,
      f.curated_group
    FROM `{dataset}.features` f
    JOIN `{dataset}.shows` s
      ON f.show_id = s.show_id
    """.strip()


def _rankings_sql(dataset: str) -> str:
    return f"""
    SELECT
      *
    FROM (
      SELECT
        f.*,
        PERCENT_RANK() OVER (ORDER BY finale_delta) AS finale_delta_percentile,
        PERCENT_RANK() OVER (ORDER BY closure_deficit) AS closure_deficit_percentile,
        CASE
          WHEN finale_delta >= 1.0 THEN 'great'
          WHEN finale_delta >= 0.5 THEN 'good'
          WHEN finale_delta >= -0.5 THEN 'neutral'
          WHEN finale_delta >= -1.5 THEN 'bad'
          ELSE 'disaster'
        END AS finale_quality_bucket
      FROM `{dataset}.v_finale_analysis` f
    )
    """.strip()


def _top_bottom_sql(dataset: str) -> str:
    return f"""
    SELECT * FROM (
      SELECT 'top_10' AS segment, title, finale_delta
      FROM `{dataset}.v_finale_analysis`
      ORDER BY finale_delta DESC
      LIMIT 10
    )
    UNION ALL
    SELECT * FROM (
      SELECT 'bottom_10' AS segment, title, finale_delta
      FROM `{dataset}.v_finale_analysis`
      ORDER BY finale_delta ASC
      LIMIT 10
    )
    """.strip()
