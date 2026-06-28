import pandas as pd
from fifa2026.features.elo import EloEngine, expected_score

def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2010-02-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [3, 0], "away_score": [0, 0],
        "neutral": [True, True],
    })

def test_expected_score_symmetry():
    assert abs(expected_score(1500, 1500, 0) - 0.5) < 1e-9
    assert expected_score(1700, 1500, 0) > 0.5

def test_winner_gains_rating_and_pointintime():
    eng = EloEngine(k=40, home_advantage=0, initial=1500).fit(_matches())
    # Before any match both start at 1500
    assert eng.rating_before("A", pd.Timestamp("2010-01-01")) == 1500
    # After A beats B on 2010-01-01, A's pre-match rating on 2010-02-01 is higher
    assert eng.rating_before("A", pd.Timestamp("2010-02-01")) > 1500
    assert eng.rating_before("B", pd.Timestamp("2010-02-01")) < 1500
