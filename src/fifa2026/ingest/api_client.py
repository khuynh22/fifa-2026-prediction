"""Generic cached API-Football client.

Currently unused by the live pipeline (player availability is sourced from a
curated file — the free API tier does not expose the 2026 season or
national-team injuries). Retained as infrastructure for a future paid-plan
enrichment. See docs/superpowers/specs/2026-06-29-availability-adjustment-design.md.
"""
from __future__ import annotations
import json
import requests
from fifa2026.cache import DiskCache

class FootballAPI:
    def __init__(self, base_url: str, api_key: str, cache: DiskCache):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache

    def get_json(self, endpoint: str, params: dict) -> dict:
        key = f"{endpoint}?{sorted(params.items())}"
        def fetch() -> str:
            resp = requests.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                params=params,
                headers={"x-apisports-key": self.api_key},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.text
        return json.loads(self.cache.get_or_fetch(key, fetch))
