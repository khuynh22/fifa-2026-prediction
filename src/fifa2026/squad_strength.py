from __future__ import annotations
from pathlib import Path
import yaml

def load_injuries(path) -> dict:
    """Parse the curated injuries file's `injuries:` map: {team: [players out]}.
    Returns {} if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    inj = data.get("injuries") or {}
    return {team: list(players or []) for team, players in inj.items()}

def availability_adjustment(injuries, penalty_per_player: float = 10.0,
                            cap: float = 40.0) -> dict:
    """Map team -> negative Elo delta from listed absences (capped). Teams with no
    listed players are omitted (no adjustment)."""
    adj = {}
    for team, players in injuries.items():
        n = len(players)
        if n <= 0:
            continue
        adj[team] = -min(n * penalty_per_player, cap)
    return adj
