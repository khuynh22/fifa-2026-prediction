import numpy as np
from fifa2026.knockout.bracket import round_probabilities

def test_round_probabilities_structure_and_sums():
    teams = ["A", "B", "C", "D"]
    strength = {"A": 8, "B": 2, "C": 6, "D": 4}
    def win_prob(a, b):
        return 1.0 / (1.0 + np.exp(-(strength[a] - strength[b])))
    rp = round_probabilities(teams, win_prob)
    # All five stage keys are always present (labels assume a 32-team bracket).
    assert set(rp["A"]) == {"reach_R16", "reach_QF", "reach_SF", "reach_final", "win"}
    # Reaching an earlier round is at least as likely as winning it all (monotonic).
    for t in teams:
        assert rp[t]["reach_R16"] >= rp[t]["win"] - 1e-9
    # Champion probabilities (the "win" column) sum to 1.
    assert abs(sum(rp[t]["win"] for t in teams) - 1.0) < 1e-9
    # Strongest team most likely to win.
    assert max(teams, key=lambda t: rp[t]["win"]) == "A"
