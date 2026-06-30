from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.config import load_config


def load_dotenv(path=".env") -> None:
    """Minimal .env loader: set KEY=VALUE pairs into os.environ if not already set.

    No third-party dependency. Lines starting with '#' and blank lines are skipped.
    Existing environment variables take precedence (setdefault)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None):
    pen = pen or {}
    depth = depth or {}
    decided = decided or {}
    hosts = getattr(feature_builder, "hosts", []) or []

    def _venue(team_a, team_b):
        if team_a in hosts:
            return team_a
        if team_b in hosts:
            return team_b
        return ""

    def win_prob(team_a: str, team_b: str) -> float:
        key = frozenset((team_a, team_b))
        if key in decided:
            return 1.0 if decided[key] == team_a else 0.0
        venue = _venue(team_a, team_b)
        row_ab = feature_builder.row(team_a, team_b, as_of_date, venue_country=venue, neutral=(venue == ""))
        row_ba = feature_builder.row(team_b, team_a, as_of_date, venue_country=venue, neutral=(venue == ""))
        p_ab = resolve_tie(model.predict_proba(pd.DataFrame([row_ab]))[0],
                           pen_a=pen.get(team_a, 0.5), pen_b=pen.get(team_b, 0.5),
                           depth_a=depth.get(team_a, 0.0), depth_b=depth.get(team_b, 0.0))
        p_ba = resolve_tie(model.predict_proba(pd.DataFrame([row_ba]))[0],
                           pen_a=pen.get(team_b, 0.5), pen_b=pen.get(team_a, 0.5),
                           depth_a=depth.get(team_b, 0.0), depth_b=depth.get(team_a, 0.0))
        return 0.5 * (p_ab + (1.0 - p_ba))
    return win_prob

def _cmd_data(args):
    from fifa2026.ingest.download import fetch_results_csv, RESULTS_URL
    cfg = load_config(args.config)
    dest = cfg.raw_dir / "results.csv"
    fetch_results_csv(RESULTS_URL, dest)
    print(f"data ready: {dest}")

def _cmd_train(args):
    from fifa2026.pipeline import run_train
    cfg = load_config(args.config)
    out = run_train(cfg)
    print(f"models saved: {out}")

def _cmd_evaluate(args):
    from fifa2026.pipeline import run_evaluate
    cfg = load_config(args.config)
    res = run_evaluate(cfg)
    print(f"metrics: {res.get('metrics')}")

def _cmd_predict(args):
    from fifa2026.pipeline import run_predict
    cfg = load_config(args.config)
    res = run_predict(cfg)
    top = sorted(res.champion_probs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Champion probabilities (top 8):")
    for team, p in top:
        print(f"  {team:25s} {p:6.1%}")

def main(argv=None):
    load_dotenv()  # pick up FOOTBALL_API_KEY from a local .env if present
    parser = argparse.ArgumentParser(prog="fifa2026")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "evaluate", "predict"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    {"data": _cmd_data, "train": _cmd_train,
     "evaluate": _cmd_evaluate, "predict": _cmd_predict}[args.cmd](args)

if __name__ == "__main__":
    main()
