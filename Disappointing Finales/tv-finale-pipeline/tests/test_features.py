import pandas as pd

from pipeline.config import get_config
from pipeline.features import _compute_features


def test_features_basic_metrics():
    config = get_config()
    episodes = pd.DataFrame(
        [
            {
                "show_id": "s1",
                "season_number": 1,
                "episode_number": 1,
                "air_date": "2020-01-01",
                "runtime_minutes": 60,
                "imdb_rating": 8.0,
            },
            {
                "show_id": "s1",
                "season_number": 1,
                "episode_number": 2,
                "air_date": "2020-01-08",
                "runtime_minutes": 60,
                "imdb_rating": 7.0,
            },
        ]
    )

    features = _compute_features(config, episodes)
    row = features.set_index("show_id").loc["s1"]

    assert row["total_episodes"] == 2
    assert row["total_runtime_minutes"] == 120
    assert round(row["series_avg_episode_rating"], 2) == 7.5
    assert row["final_season_number"] == 1
    assert row["finale_id_method"] in {"air_date", "season_episode_fallback"}
