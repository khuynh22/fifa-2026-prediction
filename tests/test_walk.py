import numpy as np
import pandas as pd
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.knockout.walk import match_breakdown, build_win_prob, walk_bracket

class _FB:
    hosts = []
    def row(self, a, b, date, venue_country, neutral):
        # encodes which team is "home" so the two orderings differ
        return {"home_is_a": 1.0 if a == "A" else 0.0}

class _Model:
    def predict_proba(self, X):
        out = []
        for v in X["home_is_a"]:
            out.append([0.6, 0.2, 0.2] if v == 1.0 else [0.5, 0.2, 0.3])
        return np.array(out)

def test_breakdown_sums_and_consistency():
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"))
    assert abs(bd["p_a_reg"] + bd["p_draw"] + bd["p_b_reg"] - 1.0) < 1e-9
    assert abs(bd["p_a_advance"] + bd["p_b_advance"] - 1.0) < 1e-9
    # consistency with the old complementary win-prob formula
    pab = np.array([0.6, 0.2, 0.2]); pba = np.array([0.5, 0.2, 0.3])
    expected = 0.5 * (resolve_tie(pab) + (1 - resolve_tie(pba)))
    assert abs(bd["p_a_advance"] - expected) < 1e-9          # 0.55
    assert bd["winner"] == "A"                                # 0.55 >= 0.5

def test_build_win_prob_matches_breakdown():
    win_prob = build_win_prob(_Model(), _FB(), pd.Timestamp("2026-07-01"))
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"))
    assert abs(win_prob("A", "B") - bd["p_a_advance"]) < 1e-9

def test_decided_tie_locks():
    decided = {frozenset(("A", "B")): "A"}
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"), decided=decided)
    assert bd["decided"] and bd["winner"] == "A"
    assert bd["p_a_advance"] == 1.0 and bd["p_b_advance"] == 0.0

def test_walk_bracket_structure():
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strength = {t: s for t, s in zip(teams, [8, 1, 7, 2, 6, 3, 5, 4])}
    def bd(a, b):
        wa = strength[a] > strength[b]
        return {"team_a": a, "team_b": b, "winner": a if wa else b,
                "p_a_advance": 1.0 if wa else 0.0}
    res = walk_bracket(teams, bd)
    assert [len(r["matches"]) for r in res["rounds"]] == [4, 2, 1]
    assert res["rounds"][0]["round"] == "Quarter-finals"  # 8 teams
    assert res["champion"] == "A"
