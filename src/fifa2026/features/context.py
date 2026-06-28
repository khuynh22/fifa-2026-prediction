from __future__ import annotations
from math import radians, sin, cos, asin, sqrt
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))

def home_flag(team: str, venue_country: str, hosts: list[str]) -> int:
    return 1 if (team == venue_country or (team in hosts and venue_country in hosts and team == venue_country)) else 0

class ContextFeatures:
    REST_CAP = 30

    def __init__(self):
        self._matches: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "ContextFeatures":
        self._matches = matches.sort_values("date").reset_index(drop=True)
        return self

    def rest_days(self, team: str, date) -> float:
        m = self._matches
        date = pd.Timestamp(date)
        mask = ((m["home_team"] == team) | (m["away_team"] == team)) & (m["date"] < date)
        prior = m[mask]
        if prior.empty:
            return float(self.REST_CAP)
        last = prior["date"].max()
        return float(min((date - last).days, self.REST_CAP))

    def head_to_head(self, team_a: str, team_b: str, date) -> dict:
        m = self._matches
        date = pd.Timestamp(date)
        pair = (((m["home_team"] == team_a) & (m["away_team"] == team_b)) |
                ((m["home_team"] == team_b) & (m["away_team"] == team_a)))
        hist = m[pair & (m["date"] < date)].dropna(subset=["home_score", "away_score"])
        if hist.empty:
            return {"h2h_ppg": 0.0}
        pts = 0
        for _, g in hist.iterrows():
            is_home = g["home_team"] == team_a
            sa = g["home_score"] if is_home else g["away_score"]
            sb = g["away_score"] if is_home else g["home_score"]
            pts += 3 if sa > sb else (1 if sa == sb else 0)
        return {"h2h_ppg": pts / len(hist)}
