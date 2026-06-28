from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

def scoreline_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), lam_h)
    a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(h, a)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return np.array([p_home, p_draw, p_away]) / total

class PoissonModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self._home = None
        self._away = None
        self._cols = None

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None,
            goals_home=None, goals_away=None) -> "PoissonModel":
        self._cols = list(X.columns)
        Xc = sm.add_constant(X, has_constant="add")
        self._home = sm.GLM(goals_home, Xc, family=sm.families.Poisson(),
                            freq_weights=sample_weight).fit()
        self._away = sm.GLM(goals_away, Xc, family=sm.families.Poisson(),
                            freq_weights=sample_weight).fit()
        return self

    def expected_goals(self, X: pd.DataFrame):
        Xc = sm.add_constant(X[self._cols], has_constant="add")
        return self._home.predict(Xc), self._away.predict(Xc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(X)
        return np.array([scoreline_probs(h, a, self.max_goals)
                         for h, a in zip(np.asarray(lam_h), np.asarray(lam_a))])
