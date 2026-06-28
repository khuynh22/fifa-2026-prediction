from __future__ import annotations
from pathlib import Path
import yaml


def load_bracket(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(data["teams"])


def _solve(teams: list[str], win_prob) -> dict[str, float]:
    """Return {team: P(team wins this sub-bracket)} via exact DP."""
    if len(teams) == 1:
        return {teams[0]: 1.0}
    mid = len(teams) // 2
    left = _solve(teams[:mid], win_prob)
    right = _solve(teams[mid:], win_prob)
    out: dict[str, float] = {}
    for a, pa in left.items():
        out[a] = out.get(a, 0.0) + pa * sum(pb * win_prob(a, b) for b, pb in right.items())
    for b, pb in right.items():
        out[b] = out.get(b, 0.0) + pb * sum(pa * win_prob(b, a) for a, pa in left.items())
    return out


def champion_probabilities(teams: list[str], win_prob) -> dict[str, float]:
    if (len(teams) & (len(teams) - 1)) != 0:
        raise ValueError("bracket size must be a power of 2")
    return _solve(teams, win_prob)


def predict_champion(teams: list[str], win_prob):
    probs = champion_probabilities(teams, win_prob)
    champ = max(probs, key=probs.get)
    return champ, probs[champ]
