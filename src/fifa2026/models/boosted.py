from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class BoostedModel:
    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.03,
                 num_leaves: int = 31, random_state: int = 42):
        self.clf = LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            random_state=random_state, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BoostedModel":
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        # reorder columns to canonical [0,1,2] regardless of clf.classes_
        order = np.argsort(self.clf.classes_)
        full = np.zeros((proba.shape[0], 3))
        for j, cls in enumerate(self.clf.classes_[order]):
            full[:, int(cls)] = proba[:, order[j]]
        return full
