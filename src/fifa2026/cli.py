from __future__ import annotations
import argparse
import pandas as pd
from fifa2026.config import load_config
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.knockout.bracket import load_bracket

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None):
    pen = pen or {}
    depth = depth or {}
    hosts = getattr(feature_builder, "hosts", []) or []

    def _venue(team_a, team_b):
        if team_a in hosts:
            return team_a
        if team_b in hosts:
            return team_b
        return ""

    def win_prob(team_a: str, team_b: str) -> float:
        venue = _venue(team_a, team_b)
        row_ab = feature_builder.row(team_a, team_b, as_of_date, venue_country=venue, neutral=(venue == ""))
        row_ba = feature_builder.row(team_b, team_a, as_of_date, venue_country=venue, neutral=(venue == ""))
        p_ab = resolve_tie(
            model.predict_proba(pd.DataFrame([row_ab]))[0],
            pen_a=pen.get(team_a, 0.5), pen_b=pen.get(team_b, 0.5),
            depth_a=depth.get(team_a, 0.0), depth_b=depth.get(team_b, 0.0),
        )
        p_ba = resolve_tie(
            model.predict_proba(pd.DataFrame([row_ba]))[0],
            pen_a=pen.get(team_b, 0.5), pen_b=pen.get(team_a, 0.5),
            depth_a=depth.get(team_b, 0.0), depth_b=depth.get(team_a, 0.0),
        )
        return 0.5 * (p_ab + (1.0 - p_ba))
    return win_prob

def _cmd_predict(args):
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
