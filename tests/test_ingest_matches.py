from pathlib import Path
import pandas as pd
from fifa2026.ingest.matches import load_matches, outcome

FIX = Path(__file__).parent / "fixtures" / "results_sample.csv"

def test_outcome_encoding():
    assert outcome(3, 1) == 0
    assert outcome(1, 1) == 1
    assert outcome(0, 2) == 2

def test_load_matches_filters_and_normalizes():
    df = load_matches(FIX, train_start="2010-01-01")
    assert list(df.columns) == [
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "neutral", "tournament", "city", "country",
    ]
    assert (df["date"] >= pd.Timestamp("2010-01-01")).all()
    assert len(df) == 3                      # 2009 row filtered out
    assert df["neutral"].dtype == bool
    assert df.iloc[0]["match_id"] == "20100611-South Africa-Mexico"
    assert df["date"].is_monotonic_increasing
