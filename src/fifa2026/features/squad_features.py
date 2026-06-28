from __future__ import annotations
import numpy as np
import pandas as pd

def team_aggregates(players: pd.DataFrame) -> pd.DataFrame:
    g = players.groupby("team")
    out = pd.DataFrame({
        "squad_value": g["market_value_eu"].sum(),
        "top_xg": g["season_xg"].max(),
        "total_xg": g["season_xg"].sum(),
        "mean_age": g["age"].mean(),
        "n_injured": g["injured"].sum().astype(int),
    })
    return out

def impute_tier_b(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in features.select_dtypes(include="number").columns:
        out[f"{col}_isna"] = out[col].isna().astype(int)
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    return out
