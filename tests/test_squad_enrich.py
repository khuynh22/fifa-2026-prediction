import os
import pandas as pd
from fifa2026.config import load_config
from fifa2026.squad_enrich import build_squad_agg

class _FakeAPI:
    def get_json(self, endpoint, params):
        team = params["team"]
        return {"team": team, "players": [
            {"name": "P1", "position": "Attacker", "age": 25, "market_value": 5e7,
             "minutes": 2000, "xg": 10.0, "xa": 5.0, "injured": False}]}

def test_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    cfg = load_config()
    assert build_squad_agg(cfg, ["Brazil"], api=None) is None

def test_with_api_returns_aggregates():
    cfg = load_config()
    agg = build_squad_agg(cfg, ["Brazil", "France"], api=_FakeAPI())
    assert agg is not None
    assert "squad_value" in agg.columns
    assert "Brazil" in agg.index
