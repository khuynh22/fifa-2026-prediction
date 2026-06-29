from __future__ import annotations
import numpy as np
import pandas as pd
from fifa2026.ingest.matches import outcome
from fifa2026.features.context import home_flag

class FeatureBuilder:
    def __init__(self, elo, form, context, confederations, squad_agg, hosts, form_windows):
        """Initialize a FeatureBuilder.

        Parameters
        ----------
        squad_agg : pd.DataFrame or None
            A STATIC (current) snapshot of squad-level aggregates with no date axis.
            It is intended for prediction-time use only. Pass ``None`` when calling
            ``build_training_matrix``; otherwise a ValueError is raised to prevent
            training-time leakage of future squad information.
        """
        self.elo = elo
        self.form = form
        self.context = context
        self.confederations = confederations
        self.squad_agg = squad_agg
        self.hosts = hosts
        self.form_windows = form_windows

    def row(self, home_team, away_team, date, venue_country, neutral) -> dict:
        a, b = home_team, away_team
        feats = {}
        feats["elo_diff"] = self.elo.rating_before(a, date) - self.elo.rating_before(b, date)
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
        if self.squad_agg is not None:
            for col, name in [("squad_value", "squad_value_diff"),
                              ("top_xg", "top_xg_diff"),
                              ("mean_age", "mean_age_diff"),
                              ("n_injured", "n_injured_diff")]:
                va = self.squad_agg[col].get(a, np.nan)
                vb = self.squad_agg[col].get(b, np.nan)
                feats[name] = float(va - vb) if pd.notna(va) and pd.notna(vb) else 0.0
        return feats

    def build_training_matrix(self, matches: pd.DataFrame):
        if self.squad_agg is not None:
            raise ValueError(
                "squad_agg is a static (current) snapshot with no date axis; using it "
                "for historical training rows leaks future information. Build the training "
                "matrix with squad_agg=None, and only attach squad features at prediction time."
            )
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
