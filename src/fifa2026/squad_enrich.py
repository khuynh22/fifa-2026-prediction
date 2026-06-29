from __future__ import annotations
import os
from fifa2026.ingest.squads import parse_squad, team_player_table
from fifa2026.features.squad_features import team_aggregates, impute_tier_b

def build_squad_agg(cfg, teams, api=None):
    if api is None:
        key = os.environ.get(cfg.raw["api"]["key_env"])
        if not key:
            return None  # graceful fallback: team-strength features only
        from fifa2026.cache import DiskCache
        from fifa2026.ingest.api_client import FootballAPI
        api = FootballAPI(cfg.raw["api"]["base_url"], key, DiskCache(cfg.raw_dir / "api_cache"))
    payloads = {}
    for team in teams:
        try:
            payloads[team] = api.get_json("players/squads", {"team": team})
        except Exception:
            continue  # skip teams the API can't resolve; never crash the pipeline
    if not payloads:
        return None
    players = team_player_table(payloads)
    return impute_tier_b(team_aggregates(players))
