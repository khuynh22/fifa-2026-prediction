from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss

def odds_to_probs(odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return raw / raw.sum()

def compare_to_market(y_true, model_proba, market_proba) -> dict:
    y_true = np.asarray(y_true)
    model_proba = np.asarray(model_proba)
    market_proba = np.asarray(market_proba)
    model_ll = float(log_loss(y_true, model_proba, labels=[0, 1, 2]))
    market_ll = float(log_loss(y_true, market_proba, labels=[0, 1, 2]))
    agreement = float(np.mean(model_proba.argmax(axis=1) == market_proba.argmax(axis=1)))
    return {
        "model_log_loss": model_ll,
        "market_log_loss": market_ll,
        "agreement_rate": agreement,
        "model_beats_market": bool(model_ll < market_ll),
    }
