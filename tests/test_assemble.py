import numpy as np
import pandas as pd
from fifa2026.features.elo import EloEngine
from fifa2026.features.form import FormFeatures
from fifa2026.features.context import ContextFeatures
from fifa2026.features.assemble import FeatureBuilder

def _matches():
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"],
        "home_score": [2, 1, 0], "away_score": [0, 0, 1],
        "neutral": [True, True, True], "country": ["X", "X", "Y"],
    })

def _builder(m):
    return FeatureBuilder(
        elo=EloEngine(home_advantage=0).fit(m),
        form=FormFeatures().fit(m),
        context=ContextFeatures().fit(m),
        confederations={"A": "UEFA", "B": "UEFA"},
        squad_agg=None, hosts=[], form_windows=[5],
    )

def test_row_has_differential_features():
    m = _matches()
    fb = _builder(m)
    row = fb.row("A", "B", pd.Timestamp("2010-03-01"), venue_country="Y", neutral=True)
    assert "elo_diff" in row and "ppg_5_diff" in row and "same_confed" in row
    assert row["same_confed"] == 1
    assert row["elo_diff"] > 0  # A beat B twice before this date

def test_no_leakage_features_ignore_future():
    """Feature row for a match must not change if FUTURE matches are added."""
    m = _matches()
    fb_now = _builder(m)
    row_now = fb_now.row("A", "B", pd.Timestamp("2010-02-01"), "X", True)

    future = pd.concat([m, pd.DataFrame({
        "match_id": ["m9"], "date": pd.to_datetime(["2010-05-01"]),
        "home_team": ["A"], "away_team": ["B"], "home_score": [9], "away_score": [0],
        "neutral": [True], "country": ["X"],
    })], ignore_index=True)
    fb_future = _builder(future)
    row_future = fb_future.row("A", "B", pd.Timestamp("2010-02-01"), "X", True)
    assert row_now == row_future  # adding a 2010-05 match cannot change a 2010-02 feature row
