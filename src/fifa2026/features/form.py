from __future__ import annotations
import pandas as pd

class FormFeatures:
    def __init__(self):
        self._matches: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "FormFeatures":
        m = matches.dropna(subset=["home_score", "away_score"]).copy()
        self._matches = m.sort_values("date").reset_index(drop=True)
        return self

    def _team_history(self, team: str, date) -> pd.DataFrame:
        m = self._matches
        date = pd.Timestamp(date)
        mask = ((m["home_team"] == team) | (m["away_team"] == team)) & (m["date"] < date)
        return m[mask]

    def team_form(self, team: str, date, window: int) -> dict:
        hist = self._team_history(team, date).tail(window)
        key = f"_{window}"
        if hist.empty:
            return {f"ppg{key}": 0.0, f"gf_rate{key}": 0.0, f"ga_rate{key}": 0.0}
        pts = gf = ga = 0
        for _, m in hist.iterrows():
            is_home = m["home_team"] == team
            scored = m["home_score"] if is_home else m["away_score"]
            conceded = m["away_score"] if is_home else m["home_score"]
            gf += scored; ga += conceded
            if scored > conceded: pts += 3
            elif scored == conceded: pts += 1
        n = len(hist)
        return {f"ppg{key}": pts / n, f"gf_rate{key}": gf / n, f"ga_rate{key}": ga / n}
