import pandas as pd
from fifa2026.features.form import FormFeatures

def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "home_team": ["A", "C", "A"], "away_team": ["B", "A", "D"],
        "home_score": [2, 1, 0], "away_score": [0, 1, 0],
        "neutral": [True, True, True],
    })

def test_form_is_point_in_time():
    ff = FormFeatures().fit(_matches())
    # Before any games, zeros
    early = ff.team_form("A", pd.Timestamp("2010-01-01"), window=5)
    assert early["ppg_5"] == 0.0
    # On 2010-03-01, A has played: win (3 pts) then draw (1 pt) -> ppg = 2.0 over 2 games
    later = ff.team_form("A", pd.Timestamp("2010-03-01"), window=5)
    assert later["ppg_5"] == 2.0
    assert later["gf_rate_5"] == 1.5   # scored 2 then 1
    assert later["ga_rate_5"] == 0.5   # conceded 0 then 1
