from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from fifa2026.ingest.matches import load_matches
from fifa2026.ingest.reference import load_confederations
from fifa2026.features.elo import EloEngine
from fifa2026.features.form import FormFeatures
from fifa2026.features.context import ContextFeatures
from fifa2026.features.assemble import FeatureBuilder
from fifa2026.models.poisson import PoissonModel
from fifa2026.models.boosted import BoostedModel
from fifa2026.models.ensemble import EnsembleModel
from fifa2026.evaluate.backtest import temporal_split, evaluate_probs
from fifa2026.persistence import save_models
from fifa2026.cli import build_win_prob
from fifa2026.knockout.bracket import champion_probabilities, round_probabilities

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

def build_feature_builder(cfg, matches, rating_adjustment=None):
    elo = EloEngine(
        k=cfg.raw["elo"]["k"], home_advantage=cfg.raw["elo"]["home_advantage"],
        initial=cfg.raw["elo"]["initial"]).fit(matches)
    form = FormFeatures().fit(matches)
    context = ContextFeatures().fit(matches)
    confed = load_confederations(_ref_dir(cfg) / "confederations.csv")
    return FeatureBuilder(elo=elo, form=form, context=context, confederations=confed,
                          hosts=cfg.raw["hosts_2026"],
                          form_windows=cfg.raw["features"]["form_windows"],
                          rating_adjustment=rating_adjustment)

def run_train(cfg, matches_csv=None) -> Path:
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    fb = build_feature_builder(cfg, matches)
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


def _load_bracket_cfg(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    teams = list(data["teams"])
    decided = {frozenset((d["winner"], d["loser"])): d["winner"]
               for d in data.get("decided", [])}
    return teams, decided, data.get("as_of", "")


def run_predict(cfg, models=None, matches_csv=None, bracket_path=None, injuries_path=None) -> PredictionResult:
    from fifa2026.persistence import load_models
    from fifa2026.squad_strength import load_injuries, availability_adjustment
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    ensemble = models if models is not None else load_models(cfg.models_dir)[0]
    bpath = bracket_path or cfg.raw.get("bracket_path", "config/bracket_2026.yaml")
    teams, decided, as_of = _load_bracket_cfg(bpath)
    as_of_date = pd.Timestamp(as_of) if as_of else matches["date"].max()
    matches = matches[matches["date"] <= as_of_date]
    ipath = injuries_path or cfg.raw.get("injuries_path", "data/reference/injuries_2026.yaml")
    injuries = load_injuries(ipath)
    av = cfg.raw.get("availability", {})
    rating_adjustment = availability_adjustment(
        injuries, penalty_per_player=av.get("penalty_per_player", 10.0),
        cap=av.get("cap", 40.0))
    fb = build_feature_builder(cfg, matches, rating_adjustment=rating_adjustment)
    win_prob = build_win_prob(ensemble, fb, as_of_date, decided=decided)
    champ = champion_probabilities(teams, win_prob)
    rounds = round_probabilities(teams, win_prob)
    ties = [{"home": teams[i], "away": teams[i + 1],
             "p_home": win_prob(teams[i], teams[i + 1])} for i in range(0, len(teams), 2)]
    availability_meta = {t: {"out": injuries.get(t, []), "elo_penalty": rating_adjustment[t]}
                         for t in rating_adjustment}
    result = PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties,
                              as_of=as_of, meta={"n_teams": len(teams),
                                                 "availability": availability_meta})
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "prediction.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _load_market_probs(cfg) -> dict:
    """Implied champion probabilities from data/reference/market_odds_2026.csv, or {}."""
    from fifa2026.ingest.odds import parse_winner_odds, implied_champion_probs
    path = Path("data/reference/market_odds_2026.csv")
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict("records")
    return implied_champion_probs(parse_winner_odds(rows))


def _calibration_bins(y_true, proba, n_bins=10) -> list:
    """Confidence-reliability curve: bin test predictions by their max probability
    (model confidence) and compare mean confidence (`pred`) to empirical accuracy
    (`obs`) in each bin. A well-calibrated model lies on the diagonal."""
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return []
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(1.0 / proba.shape[1], 1.0, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == n_bins - 1
        mask = (conf >= lo) & (conf <= hi) if last else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        out.append({"pred": float(conf[mask].mean()),
                    "obs": float(correct[mask].mean()),
                    "n": int(mask.sum())})
    return out


def run_evaluate(cfg, matches_csv=None) -> dict:
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    m = matches.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    market = _load_market_probs(cfg)
    cutoff = cfg.raw.get("val_cutoff", "2022-01-01")
    train_idx, test_idx = temporal_split(m, cutoff=cutoff)
    if len(test_idx) == 0:
        return {"metrics": {}, "calibration": [], "market": market}
    fb = build_feature_builder(cfg, m.iloc[train_idx])
    Xtr, ytr, gh, ga = fb.build_training_matrix(m.iloc[train_idx])
    poisson = PoissonModel().fit(Xtr, ytr, goals_home=gh, goals_away=ga)
    boosted = BoostedModel().fit(Xtr, ytr)
    ensemble = EnsembleModel(poisson, boosted)
    Xte = pd.DataFrame([fb.row(r["home_team"], r["away_team"], r["date"],
                               r.get("country", ""), bool(r.get("neutral", True)))
                        for _, r in m.iloc[test_idx].iterrows()]).fillna(0.0)
    from fifa2026.ingest.matches import outcome
    yte = np.array([outcome(int(r["home_score"]), int(r["away_score"]))
                    for _, r in m.iloc[test_idx].iterrows()])
    proba = ensemble.predict_proba(Xte[Xtr.columns])
    result = {"metrics": evaluate_probs(yte, proba),
              "calibration": _calibration_bins(yte, proba), "market": market}
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
