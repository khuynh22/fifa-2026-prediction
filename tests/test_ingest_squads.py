import json
from pathlib import Path
from fifa2026.ingest.squads import parse_squad

FIX = Path(__file__).parent / "fixtures" / "squad_sample.json"

def test_parse_squad():
    payload = json.loads(FIX.read_text(encoding="utf-8"))
    df = parse_squad(payload)
    assert list(df.columns) == [
        "team", "player", "position", "age",
        "market_value_eu", "season_minutes", "season_xg", "season_xa", "injured",
    ]
    assert len(df) == 2
    assert df.iloc[0]["team"] == "Brazil"
    assert df.iloc[0]["market_value_eu"] == 90000000
    assert bool(df.iloc[1]["injured"]) is True
