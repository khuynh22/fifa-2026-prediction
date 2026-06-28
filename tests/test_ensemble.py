import numpy as np
import pandas as pd
from fifa2026.models.ensemble import EnsembleModel

class _Fake:
    def __init__(self, proba): self._p = np.array(proba)
    def predict_proba(self, X): return np.tile(self._p, (len(X), 1))

def test_blend_is_convex_combo():
    a = _Fake([0.7, 0.2, 0.1])
    b = _Fake([0.1, 0.2, 0.7])
    ens = EnsembleModel(a, b, weight=0.5)
    X = pd.DataFrame({"x": [0, 0]})
    out = ens.predict_proba(X)
    assert np.allclose(out[0], [0.4, 0.2, 0.4])
    assert np.allclose(out.sum(axis=1), 1.0)

def test_tune_weight_prefers_better_model():
    good = _Fake([0.9, 0.05, 0.05])   # matches label 0
    bad = _Fake([0.05, 0.05, 0.9])
    ens = EnsembleModel(good, bad)
    X = pd.DataFrame({"x": np.zeros(20)})
    y = np.zeros(20, dtype=int)        # all home wins
    w = ens.tune_weight(X, y)
    assert w > 0.5                     # lean toward the good (poisson-slot) model
