from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

class EnsembleModel:
    def __init__(self, poisson, boosted, weight: float = 0.5):
        self.poisson = poisson
        self.boosted = boosted
        self.weight = weight

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.poisson.predict_proba(X)
        b = self.boosted.predict_proba(X)
        out = self.weight * p + (1 - self.weight) * b
        return out / out.sum(axis=1, keepdims=True)

    def tune_weight(self, X_val: pd.DataFrame, y_val, grid=None) -> float:
        grid = grid if grid is not None else np.linspace(0, 1, 21)
        p = self.poisson.predict_proba(X_val)
        b = self.boosted.predict_proba(X_val)
        best_w, best_ll = 0.5, float("inf")
        for w in grid:
            blend = w * p + (1 - w) * b
            blend = blend / blend.sum(axis=1, keepdims=True)
            ll = log_loss(y_val, blend, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll, best_w = ll, w
        self.weight = float(best_w)
        return self.weight
