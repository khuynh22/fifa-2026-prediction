from __future__ import annotations

def parse_winner_odds(rows: list[dict]) -> dict[str, float]:
    return {r["team"]: float(r["decimal_odds"]) for r in rows}

def implied_champion_probs(odds: dict[str, float]) -> dict[str, float]:
    raw = {t: 1.0 / o for t, o in odds.items() if o and o > 0}
    total = sum(raw.values())
    if total <= 0:
        return {t: 0.0 for t in odds}
    return {t: v / total for t, v in raw.items()}
