import numpy as np
import pandas as pd
from fifa2026.models.poisson import PoissonModel, scoreline_probs

def test_scoreline_probs_sum_to_one_and_favor_higher_lambda():
    p = scoreline_probs(2.0, 0.5)
    assert abs(p.sum() - 1.0) < 1e-6
    assert p[0] > p[2]  # home (higher lambda) more likely to win

def test_scoreline_probs_extreme_lambda_no_nan():
    import numpy as np
    from fifa2026.models.poisson import scoreline_probs
    p = scoreline_probs(800.0, 800.0)  # both lambdas >> max_goals -> float underflow -> NaN without guard
    assert np.all(np.isfinite(p))
    assert abs(p.sum() - 1.0) < 1e-9

def test_poisson_fits_and_predicts_shape():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n)})
    # stronger team (higher elo_diff) scores more
    yh = rng.poisson(np.clip(1.3 + X["elo_diff"] / 200, 0.1, None))
    ya = rng.poisson(np.clip(1.3 - X["elo_diff"] / 200, 0.1, None))
    y = np.where(yh > ya, 0, np.where(yh == ya, 1, 2))
    model = PoissonModel().fit(X, y, goals_home=yh.values if hasattr(yh, "values") else yh,
                              goals_away=ya.values if hasattr(ya, "values") else ya)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
