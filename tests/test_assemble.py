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

def _builder(m, rating_adjustment=None):
    return FeatureBuilder(
        elo=EloEngine(home_advantage=0).fit(m),
        form=FormFeatures().fit(m),
        context=ContextFeatures().fit(m),
        confederations={"A": "UEFA", "B": "UEFA"},
        hosts=[], form_windows=[5], rating_adjustment=rating_adjustment,
    )

def test_row_has_differential_features():
    m = _matches()
    fb = _builder(m)
    row = fb.row("A", "B", pd.Timestamp("2010-03-01"), venue_country="Y", neutral=True)
    assert "elo_diff" in row and "ppg_5_diff" in row and "same_confed" in row
    assert row["same_confed"] == 1
    assert row["elo_diff"] > 0  # A beat B twice before this date

def test_build_training_matrix_returns_4tuple_aligned():
    m = _matches()
    fb = _builder(m)
    result = fb.build_training_matrix(m)
    assert len(result) == 4, "build_training_matrix must return a 4-tuple (X, y, goals_home, goals_away)"
    X, y, gh, ga = result
    assert len(X) == len(y) == len(gh) == len(ga), "all arrays must be aligned"
    # Data sorted by date: [m1: 2-0, m2: 1-0, m3: 0-1]
    assert list(gh) == [2, 1, 0], "goals_home must match date-sorted home_score"
    assert list(ga) == [0, 0, 1], "goals_away must match date-sorted away_score"


def test_rating_adjustment_shifts_elo_diff_symmetrically():
    m = _matches()
    base = _builder(m).row("A", "B", pd.Timestamp("2010-03-01"), "Y", True)
    adj = _builder(m, rating_adjustment={"A": -30.0}).row("A", "B", pd.Timestamp("2010-03-01"), "Y", True)
    # A penalized by 30 -> elo_diff drops by exactly 30; swapping teams flips the sign.
    assert abs((adj["elo_diff"] - base["elo_diff"]) - (-30.0)) < 1e-9
    swapped = _builder(m, rating_adjustment={"A": -30.0}).row("B", "A", pd.Timestamp("2010-03-01"), "Y", True)
    assert abs(adj["elo_diff"] + swapped["elo_diff"]) < 1e-9  # symmetric


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
