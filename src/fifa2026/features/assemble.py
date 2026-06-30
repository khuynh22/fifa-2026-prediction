from __future__ import annotations
import numpy as np
import pandas as pd
from fifa2026.ingest.matches import outcome
from fifa2026.features.context import home_flag

class FeatureBuilder:
    def __init__(self, elo, form, context, confederations, hosts, form_windows,
                 rating_adjustment=None):
        """Builds per-match A-vs-B differential features.

        rating_adjustment : dict[str, float] | None
            Optional prediction-time Elo deltas per team (e.g. an availability
            penalty). Folded into ``elo_diff``. Leave None for training.
        """
        self.elo = elo
        self.form = form
        self.context = context
        self.confederations = confederations
        self.hosts = hosts
        self.form_windows = form_windows
        self.rating_adjustment = rating_adjustment or {}

    def row(self, home_team, away_team, date, venue_country, neutral) -> dict:
        a, b = home_team, away_team
        adj = self.rating_adjustment
        feats = {}
        feats["elo_diff"] = ((self.elo.rating_before(a, date) + adj.get(a, 0.0))
                             - (self.elo.rating_before(b, date) + adj.get(b, 0.0)))
        for w in self.form_windows:
            fa = self.form.team_form(a, date, w)
            fb = self.form.team_form(b, date, w)
            feats[f"ppg_{w}_diff"] = fa[f"ppg_{w}"] - fb[f"ppg_{w}"]
            feats[f"gf_rate_{w}_diff"] = fa[f"gf_rate_{w}"] - fb[f"gf_rate_{w}"]
            feats[f"ga_rate_{w}_diff"] = fa[f"ga_rate_{w}"] - fb[f"ga_rate_{w}"]
        feats["rest_diff"] = self.context.rest_days(a, date) - self.context.rest_days(b, date)
        feats["h2h_ppg"] = self.context.head_to_head(a, b, date)["h2h_ppg"]
        feats["home_diff"] = (home_flag(a, venue_country, self.hosts)
                              - home_flag(b, venue_country, self.hosts))
        feats["same_confed"] = int(self.confederations.get(a) == self.confederations.get(b)
                                   and self.confederations.get(a) is not None)
        return feats

    def build_training_matrix(self, matches: pd.DataFrame):
        rows, labels, goals_home, goals_away = [], [], [], []
        m = matches.dropna(subset=["home_score", "away_score"]).sort_values("date")
        for _, g in m.iterrows():
            rows.append(self.row(g["home_team"], g["away_team"], g["date"],
                                 g.get("country", ""), bool(g.get("neutral", True))))
            labels.append(outcome(int(g["home_score"]), int(g["away_score"])))
            goals_home.append(int(g["home_score"]))
            goals_away.append(int(g["away_score"]))
        X = pd.DataFrame(rows).fillna(0.0)
        return X, np.array(labels), np.array(goals_home), np.array(goals_away)
