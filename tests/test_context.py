import pandas as pd
from fifa2026.features.context import haversine_km, ContextFeatures, home_flag

def test_haversine_known_distance():
    # Mexico City to Dallas ~ 1400 km (allow tolerance)
    d = haversine_km(19.43, -99.13, 32.78, -96.80)
    assert 1300 < d < 1600

def test_home_flag():
    assert home_flag("Mexico", "Mexico", ["Mexico", "United States", "Canada"]) == 1
    assert home_flag("Brazil", "Mexico", ["Mexico"]) == 0

def test_rest_days_point_in_time():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2010-06-01", "2010-06-08"]),
        "home_team": ["A", "A"], "away_team": ["B", "C"],
        "home_score": [1, 1], "away_score": [0, 0], "neutral": [True, True],
    })
    cf = ContextFeatures().fit(m)
    assert cf.rest_days("A", pd.Timestamp("2010-06-08")) == 7
    assert cf.rest_days("A", pd.Timestamp("2010-06-01")) == 30  # no prior match -> default cap
