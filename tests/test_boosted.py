import numpy as np
import pandas as pd
from fifa2026.models.boosted import BoostedModel

def test_boosted_predicts_calibrated_shape_and_order():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n)})
    # home win when elo_diff high, away win when low, draw in middle
    y = np.where(X["elo_diff"] > 50, 0, np.where(X["elo_diff"] < -50, 2, 1))
    model = BoostedModel(n_estimators=50).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # A strongly-favored home case should put most mass on column 0
    strong = model.predict_proba(pd.DataFrame({"elo_diff": [400.0]}))
    assert strong[0].argmax() == 0
