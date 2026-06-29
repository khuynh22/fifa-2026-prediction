import numpy as np
import pandas as pd
from fifa2026.cli import build_win_prob
from fifa2026.knockout.bracket import predict_champion

class _Model:
    """Returns a fixed strong-home probability for any feature row."""
    def predict_proba(self, X):
        return np.tile(np.array([0.6, 0.2, 0.2]), (len(X), 1))

class _FB:
    def row(self, a, b, date, venue_country, neutral):
        return {"elo_diff": 1.0}

def test_build_win_prob_and_champion():
    win_prob = build_win_prob(_Model(), _FB(), as_of_date=pd.Timestamp("2026-07-01"))
    p = win_prob("A", "B")
    assert 0.0 <= p <= 1.0
    champ, prob = predict_champion(["A", "B", "C", "D"], win_prob)
    assert champ in {"A", "B", "C", "D"}
    assert 0.0 <= prob <= 1.0
