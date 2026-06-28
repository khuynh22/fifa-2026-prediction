import numpy as np
from fifa2026.knockout.bracket import champion_probabilities, predict_champion


def test_four_team_bracket_sums_to_one_and_favors_strong():
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.9, "B": 0.5, "C": 0.5, "D": 0.1}
    def win_prob(a, b):  # logistic on strength difference
        return 1.0 / (1.0 + np.exp(-(strength[a] - strength[b]) * 6))
    probs = champion_probabilities(teams, win_prob)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["A"] == max(probs.values())
    assert probs["D"] == min(probs.values())


def test_predict_champion_returns_argmax():
    teams = ["A", "B"]
    champ, p = predict_champion(teams, lambda a, b: 0.8 if a == "A" else 0.2)
    assert champ == "A" and abs(p - 0.8) < 1e-9
