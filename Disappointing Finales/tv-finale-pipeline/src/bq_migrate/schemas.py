from __future__ import annotations

from google.cloud import bigquery


def schema_for_table(table: str) -> list[bigquery.SchemaField]:
    if table == "shows":
        return [
            bigquery.SchemaField("show_id", "STRING"),
            bigquery.SchemaField("imdb_tconst", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("start_year", "INT64"),
            bigquery.SchemaField("end_year", "INT64"),
            bigquery.SchemaField("genres", "STRING"),
            bigquery.SchemaField("platform_raw", "STRING"),
            bigquery.SchemaField("platform_norm", "STRING"),
            bigquery.SchemaField("ip_type_norm", "STRING"),
            bigquery.SchemaField("ending_type_norm", "STRING"),
            bigquery.SchemaField("lead_exit_before_final", "STRING"),
            bigquery.SchemaField("sources", "STRING"),
            bigquery.SchemaField("curated_group", "STRING"),
        ]

    if table == "episodes":
        return [
            bigquery.SchemaField("show_id", "STRING"),
            bigquery.SchemaField("season_number", "INT64"),
            bigquery.SchemaField("episode_number", "INT64"),
            bigquery.SchemaField("air_date", "DATE"),
            bigquery.SchemaField("runtime_minutes", "FLOAT64"),
            bigquery.SchemaField("imdb_rating", "FLOAT64"),
            bigquery.SchemaField("imdb_votes", "INT64"),
            bigquery.SchemaField("episode_title", "STRING"),
            bigquery.SchemaField("has_tvmaze_match", "BOOL"),
            bigquery.SchemaField("has_imdb_rating", "BOOL"),
        ]

    if table == "features":
        return [
            bigquery.SchemaField("show_id", "STRING"),
            bigquery.SchemaField("imdb_tconst", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("total_episodes", "INT64"),
            bigquery.SchemaField("total_runtime_minutes", "FLOAT64"),
            bigquery.SchemaField("avg_episode_runtime", "FLOAT64"),
            bigquery.SchemaField("series_avg_episode_rating", "FLOAT64"),
            bigquery.SchemaField("final_season_number", "INT64"),
            bigquery.SchemaField("final_season_avg_rating", "FLOAT64"),
            bigquery.SchemaField("finale_episode_rating", "FLOAT64"),
            bigquery.SchemaField("finale_delta", "FLOAT64"),
            bigquery.SchemaField("closure_deficit", "FLOAT64"),
            bigquery.SchemaField("final_season_premiere_date", "DATE"),
            bigquery.SchemaField("penultimate_season_finale_date", "DATE"),
            bigquery.SchemaField("gap_days_to_final_season", "FLOAT64"),
            bigquery.SchemaField("era_bucket", "STRING"),
            bigquery.SchemaField("runway_bucket", "STRING"),
            bigquery.SchemaField("tvmaze_match_rate", "FLOAT64"),
            bigquery.SchemaField("missing_runtime_pct", "FLOAT64"),
            bigquery.SchemaField("missing_airdate_pct", "FLOAT64"),
            bigquery.SchemaField("missing_ratings_pct", "FLOAT64"),
            bigquery.SchemaField("airdate_coverage_pct", "FLOAT64"),
            bigquery.SchemaField("runtime_coverage_pct", "FLOAT64"),
            bigquery.SchemaField("ratings_coverage_pct", "FLOAT64"),
            bigquery.SchemaField("data_quality_score", "FLOAT64"),
            bigquery.SchemaField("finale_id_method", "STRING"),
            bigquery.SchemaField("finale_is_true_series_finale", "BOOL"),
            bigquery.SchemaField("curated_group", "STRING"),
        ]

    if table == "mapping":
        return [
            bigquery.SchemaField("imdb_tconst", "STRING"),
            bigquery.SchemaField("tvmaze_id", "INT64"),
            bigquery.SchemaField("tvmaze_name", "STRING"),
            bigquery.SchemaField("tvmaze_url", "STRING"),
            bigquery.SchemaField("match_score", "FLOAT64"),
            bigquery.SchemaField("match_confidence", "STRING"),
            bigquery.SchemaField("chosen_reason", "STRING"),
            bigquery.SchemaField("match_error", "STRING"),
            bigquery.SchemaField("tvmaze_genres", "STRING"),
            bigquery.SchemaField("tvmaze_network", "STRING"),
            bigquery.SchemaField("tvmaze_web_channel", "STRING"),
            bigquery.SchemaField("tvmaze_premiered", "DATE"),
            bigquery.SchemaField("tvmaze_ended", "DATE"),
            bigquery.SchemaField("tvmaze_status", "STRING"),
            bigquery.SchemaField("tvmaze_type", "STRING"),
        ]

    raise ValueError(f"Unknown table: {table}")
