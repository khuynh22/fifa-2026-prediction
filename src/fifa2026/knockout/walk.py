from __future__ import annotations
import pandas as pd
from fifa2026.knockout.resolve import shootout_prob

ROUND_NAMES = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
               4: "Semi-finals", 2: "Final"}

def match_breakdown(model, feature_builder, team_a, team_b, as_of_date,
                    pen=None, depth=None, decided=None) -> dict:
    pen = pen or {}
    depth = depth or {}
    decided = decided or {}
    key = frozenset((team_a, team_b))
    if key in decided:
        a_adv = 1.0 if decided[key] == team_a else 0.0
        return {"team_a": team_a, "team_b": team_b, "decided": True,
                "winner": decided[key], "p_a_reg": a_adv, "p_draw": 0.0,
                "p_b_reg": 1.0 - a_adv, "p_a_shootout": a_adv,
                "p_a_advance": a_adv, "p_b_advance": 1.0 - a_adv}
    hosts = getattr(feature_builder, "hosts", []) or []
    venue = team_a if team_a in hosts else (team_b if team_b in hosts else "")
    row_ab = feature_builder.row(team_a, team_b, as_of_date, venue_country=venue, neutral=(venue == ""))
    row_ba = feature_builder.row(team_b, team_a, as_of_date, venue_country=venue, neutral=(venue == ""))
    pab = model.predict_proba(pd.DataFrame([row_ab]))[0]
    pba = model.predict_proba(pd.DataFrame([row_ba]))[0]
    p_a_reg = 0.5 * (pab[0] + pba[2])
    p_draw = 0.5 * (pab[1] + pba[1])
    p_b_reg = 0.5 * (pab[2] + pba[0])
    s_a = shootout_prob(pen.get(team_a, 0.5), pen.get(team_b, 0.5),
                        depth.get(team_a, 0.0), depth.get(team_b, 0.0))
    p_a_adv = float(p_a_reg + p_draw * s_a)
    winner = team_a if p_a_adv >= 0.5 else team_b
    return {"team_a": team_a, "team_b": team_b, "decided": False, "winner": winner,
            "p_a_reg": float(p_a_reg), "p_draw": float(p_draw), "p_b_reg": float(p_b_reg),
            "p_a_shootout": float(s_a), "p_a_advance": p_a_adv,
            "p_b_advance": float(1.0 - p_a_adv)}

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None):
    def win_prob(team_a: str, team_b: str) -> float:
        return match_breakdown(model, feature_builder, team_a, team_b, as_of_date,
                               pen=pen, depth=depth, decided=decided)["p_a_advance"]
    return win_prob

def walk_bracket(teams, breakdown_fn) -> dict:
    """Most-likely path: each tie's predicted winner advances to the next round."""
    rounds = []
    current = list(teams)
    while len(current) > 1:
        size = len(current)
        matches = [breakdown_fn(current[i], current[i + 1]) for i in range(0, size, 2)]
        rounds.append({"round": ROUND_NAMES.get(size, f"Round of {size}"), "matches": matches})
        current = [m["winner"] for m in matches]
    return {"rounds": rounds, "champion": current[0] if current else None}
