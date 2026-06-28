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
