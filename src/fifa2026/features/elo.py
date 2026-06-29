from __future__ import annotations
from collections import defaultdict
import pandas as pd

def expected_score(rating_a: float, rating_b: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((rating_a + home_adv) - rating_b) / 400.0))

class EloEngine:
    def __init__(self, k: float = 40, home_advantage: float = 65, initial: float = 1500):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self._current: dict[str, float] = defaultdict(lambda: initial)
        # pre-match rating snapshots: (team, date) -> rating before that match
        self._pre: dict[tuple[str, pd.Timestamp], float] = {}

    def fit(self, matches: pd.DataFrame) -> "EloEngine":
        for _, m in matches.sort_values("date").iterrows():
            if pd.isna(m["home_score"]) or pd.isna(m["away_score"]):
                continue
            h, a, date = m["home_team"], m["away_team"], m["date"]
            rh, ra = self._current[h], self._current[a]
            self._pre[(h, date)] = rh
            self._pre[(a, date)] = ra
            ha = 0 if m.get("neutral", False) else self.home_advantage
            exp_h = expected_score(rh, ra, ha)
            score_h = 1.0 if m["home_score"] > m["away_score"] else (
                0.5 if m["home_score"] == m["away_score"] else 0.0)
            self._current[h] = rh + self.k * (score_h - exp_h)
            self._current[a] = ra + self.k * ((1 - score_h) - (1 - exp_h))
        return self

    def rating_before(self, team: str, date) -> float:
        """Return the team's Elo rating immediately before `date`.

        If `date` was not seen during ``fit`` (e.g. a future prediction date),
        returns the latest known rating.  Do not query arbitrary past
        non-match dates — only match dates and future dates are meaningful.
        """
        return self._pre.get((team, pd.Timestamp(date)), self._current.get(team, self.initial))

    def rating_now(self, team: str) -> float:
        return self._current.get(team, self.initial)
