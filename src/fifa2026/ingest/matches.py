from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = ["match_id", "date", "home_team", "away_team",
           "home_score", "away_score", "neutral", "tournament", "city", "country"]

def outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2

def load_matches(csv_path: str | Path, train_start: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    if isinstance(df["neutral"].dtype, object) or df["neutral"].dtype != bool:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    for col in ("city", "country", "tournament"):
        df[col] = df[col].fillna("")
    if train_start is not None:
        df = df[df["date"] >= pd.Timestamp(train_start)]
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = (df["date"].dt.strftime("%Y%m%d") + "-"
                      + df["home_team"] + "-" + df["away_team"])
    return df[COLUMNS]
