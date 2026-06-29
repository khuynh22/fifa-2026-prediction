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


def test_champion_probs_sum_to_one_with_noncomplementary_winprob():
    teams = ["A", "B", "C", "D"]
    # deliberately NON-complementary: win_prob(a,b)+win_prob(b,a) != 1
    probs = champion_probabilities(teams, lambda a, b: 0.7)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_champion_probabilities_rejects_non_power_of_two():
    import pytest
    with pytest.raises(ValueError):
        champion_probabilities(["A", "B", "C"], lambda a, b: 0.5)


def test_predict_champion_returns_argmax():
    teams = ["A", "B"]
    champ, p = predict_champion(teams, lambda a, b: 0.8 if a == "A" else 0.2)
    assert champ == "A" and abs(p - 0.8) < 1e-9
