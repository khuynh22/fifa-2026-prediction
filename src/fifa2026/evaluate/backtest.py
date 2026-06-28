from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

def temporal_split(matches: pd.DataFrame, cutoff: str):
    date = pd.to_datetime(matches["date"])
    cut = pd.Timestamp(cutoff)
    train = matches.index[date < cut]
    test = matches.index[date >= cut]
    return train, test

def evaluate_probs(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    ll = float(log_loss(y_true, proba, labels=[0, 1, 2]))
    acc = float(accuracy_score(y_true, proba.argmax(axis=1)))
    return {"log_loss": ll, "brier": brier, "accuracy": acc}
