import numpy as np
from fifa2026.evaluate.benchmark import odds_to_probs, compare_to_market

def test_odds_to_probs_devig_sums_to_one():
    p = odds_to_probs(2.0, 3.5, 4.0)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[2]  # shorter odds -> higher prob

def test_compare_to_market():
    y = np.array([0, 0, 2])
    model = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    market = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]])
    out = compare_to_market(y, model, market)
    assert out["agreement_rate"] == 1.0
    assert out["model_beats_market"] is True
