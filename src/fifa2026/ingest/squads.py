from __future__ import annotations
import pandas as pd

SQUAD_COLUMNS = ["team", "player", "position", "age", "market_value_eu",
                 "season_minutes", "season_xg", "season_xa", "injured"]

def parse_squad(payload: dict) -> pd.DataFrame:
    team = payload["team"]
    rows = []
    for p in payload["players"]:
        rows.append({
            "team": team,
            "player": p["name"],
            "position": p.get("position", ""),
            "age": p.get("age"),
            "market_value_eu": p.get("market_value"),
            "season_minutes": p.get("minutes"),
            "season_xg": p.get("xg"),
            "season_xa": p.get("xa"),
            "injured": bool(p.get("injured", False)),
        })
    return pd.DataFrame(rows, columns=SQUAD_COLUMNS)

def team_player_table(payloads: dict[str, dict]) -> pd.DataFrame:
    frames = [parse_squad(p) for p in payloads.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SQUAD_COLUMNS)
