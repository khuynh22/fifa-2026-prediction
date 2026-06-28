from __future__ import annotations
from pathlib import Path
from typing import Callable
import hashlib

class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.root / f"{digest}.cache"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put(self, key: str, value: str) -> None:
        self._path(key).write_text(value, encoding="utf-8")

    def get_or_fetch(self, key: str, fetch: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fetch()
        self.put(key, value)
        return value
