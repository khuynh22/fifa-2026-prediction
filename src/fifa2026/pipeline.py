from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
from fifa2026.ingest.matches import load_matches
from fifa2026.ingest.reference import load_confederations
from fifa2026.features.elo import EloEngine
from fifa2026.features.form import FormFeatures
from fifa2026.features.context import ContextFeatures
from fifa2026.features.assemble import FeatureBuilder
from fifa2026.models.poisson import PoissonModel
from fifa2026.models.boosted import BoostedModel
from fifa2026.models.ensemble import EnsembleModel
from fifa2026.evaluate.backtest import temporal_split
from fifa2026.persistence import save_models

@dataclass
class PredictionResult:
    champion_probs: dict
    round_probs: dict
    tie_probs: list = field(default_factory=list)
    as_of: str = ""
    meta: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)
    @staticmethod
    def from_dict(d: dict) -> "PredictionResult":
        return PredictionResult(**d)

def _ref_dir(cfg) -> Path:
    return Path("data/reference")

def build_feature_builder(cfg, matches, squad_agg=None):
    elo = EloEngine(
        k=cfg.raw["elo"]["k"], home_advantage=cfg.raw["elo"]["home_advantage"],
        initial=cfg.raw["elo"]["initial"]).fit(matches)
    form = FormFeatures().fit(matches)
    context = ContextFeatures().fit(matches)
    confed = load_confederations(_ref_dir(cfg) / "confederations.csv")
    return FeatureBuilder(elo=elo, form=form, context=context, confederations=confed,
                          squad_agg=squad_agg, hosts=cfg.raw["hosts_2026"],
                          form_windows=cfg.raw["features"]["form_windows"])

def run_train(cfg, matches_csv=None) -> Path:
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    fb = build_feature_builder(cfg, matches, squad_agg=None)
    X, y, gh, ga = fb.build_training_matrix(matches)
    # temporal split for ensemble weight tuning
    train_idx, val_idx = temporal_split(matches.dropna(subset=["home_score", "away_score"])
                                        .sort_values("date").reset_index(drop=True),
                                        cutoff=cfg.raw.get("val_cutoff", "2022-01-01"))
    poisson = PoissonModel().fit(X, y, goals_home=gh, goals_away=ga)
    boosted = BoostedModel().fit(X, y)
    ensemble = EnsembleModel(poisson, boosted)
    if len(val_idx) > 0:
        ensemble.tune_weight(X.iloc[val_idx], y[val_idx])
    save_models(cfg.models_dir, ensemble,
                {"feature_cols": list(X.columns), "trained_on": cfg.train_start,
                 "weight": ensemble.weight})
    return Path(cfg.models_dir)
