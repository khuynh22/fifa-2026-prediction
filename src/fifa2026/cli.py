from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from fifa2026.config import load_config
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.knockout.bracket import load_bracket, champion_probabilities, predict_champion

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None):
    pen = pen or {}
    depth = depth or {}
    def win_prob(team_a: str, team_b: str) -> float:
        row = feature_builder.row(team_a, team_b, as_of_date, venue_country="", neutral=True)
        X = pd.DataFrame([row])
        p_reg = model.predict_proba(X)[0]
        return resolve_tie(
            p_reg,
            pen_a=pen.get(team_a, 0.5), pen_b=pen.get(team_b, 0.5),
            depth_a=depth.get(team_a, 0.0), depth_b=depth.get(team_b, 0.0),
        )
    return win_prob

def _cmd_predict(args):
    cfg = load_config(args.config)
    teams = load_bracket(cfg.raw.get("bracket_path", "config/bracket_2026.yaml"))
    # NOTE: model + feature_builder loaded from models_dir in full wiring.
    raise SystemExit("predict requires trained model artifacts in models/ (see README)")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="fifa2026")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "evaluate", "predict"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    {"predict": _cmd_predict}.get(args.cmd, lambda a: print(f"{args.cmd}: see README"))(args)

if __name__ == "__main__":
    main()
