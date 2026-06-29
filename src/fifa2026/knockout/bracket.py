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
    if len(teams) < 1 or (len(teams) & (len(teams) - 1)) != 0:
        raise ValueError("bracket size must be a positive power of 2")
    probs = _solve(teams, win_prob)
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def predict_champion(teams: list[str], win_prob):
    probs = champion_probabilities(teams, win_prob)
    champ = max(probs, key=probs.get)
    return champ, probs[champ]


# Map subtree size -> the stage a team reaches by WINNING that subtree,
# for a full 32-team bracket: win a size-2 tie => reach R16, size-4 => QF, etc.
_STAGE_BY_SIZE = {2: "reach_R16", 4: "reach_QF", 8: "reach_SF", 16: "reach_final", 32: "win"}
_STAGES = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]


def round_probabilities(teams: list[str], win_prob) -> dict[str, dict[str, float]]:
    """For each team, probability of reaching each knockout stage.

    Keys: reach_R16, reach_QF, reach_SF, reach_final, win. A team's probability of
    reaching a stage equals its probability of WINNING the sub-bracket whose winner
    advances to that stage, computed by solving the chunk of the appropriate size.
    Stage labels assume a 32-team bracket; the top stage (chunk size == n) is always
    'win'. For smaller brackets, the deeper labels simply stay 0.0."""
    n = len(teams)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("bracket size must be a positive power of 2")
    out = {t: {s: 0.0 for s in _STAGES} for t in teams}
    for size in (2, 4, 8, 16, 32):
        if size > n:
            break
        key = "win" if size == n else _STAGE_BY_SIZE[size]
        for i in range(0, n, size):
            for t, p in _solve(teams[i:i + size], win_prob).items():
                out[t][key] = p
    return out
