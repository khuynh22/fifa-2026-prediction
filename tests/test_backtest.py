import numpy as np
import pandas as pd
from fifa2026.evaluate.backtest import temporal_split, evaluate_probs

def test_temporal_split_respects_time():
    m = pd.DataFrame({"date": pd.to_datetime(["2018-01-01", "2022-01-01", "2024-01-01"])})
    train, test = temporal_split(m, cutoff="2022-01-01")
    assert train.tolist() == [0]
    assert test.tolist() == [1, 2]

def test_metrics_perfect_and_shapes():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)[y]
    out = evaluate_probs(y, perfect)
    assert out["accuracy"] == 1.0
    assert out["brier"] < 1e-9
    assert out["log_loss"] < 1e-6
