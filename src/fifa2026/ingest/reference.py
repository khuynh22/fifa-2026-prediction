from __future__ import annotations
import pandas as pd

def load_confederations(csv_path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["team"], df["confederation"]))

def load_venues(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["altitude_m"] = df["altitude_m"].fillna(0).astype(float)
    return df[["city", "country", "lat", "lon", "altitude_m"]]

def team_country_coords(venues_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    first = venues_df.groupby("country").first()
    return {c: (float(row["lat"]), float(row["lon"])) for c, row in first.iterrows()}

def is_host(team: str, hosts: list[str]) -> bool:
    return team in hosts
