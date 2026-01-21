from pipeline.config import get_config
from pipeline.matching import choose_best_match


def test_choose_best_match_high_confidence():
    config = get_config()
    candidates = [
        {"show": {"id": 1, "name": "Breaking Bad", "premiered": "2008-01-20", "genres": ["Crime"]}},
        {"show": {"id": 2, "name": "Breaking Bad: Extras", "premiered": "2010-01-01"}},
    ]
    result = choose_best_match("Breaking Bad", "2008", candidates, config)
    assert result.tvmaze_id == 1
    assert result.match_confidence in {"high", "medium"}


def test_choose_best_match_low_confidence():
    config = get_config()
    candidates = [{"show": {"id": 3, "name": "Some Other Show", "premiered": "1999-01-01"}}]
    result = choose_best_match("Completely Different", "2020", candidates, config)
    assert result.tvmaze_id is None
