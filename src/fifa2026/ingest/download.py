from __future__ import annotations
from pathlib import Path
import requests

# Public, free, regularly-updated international results dataset.
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def _http_get(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text

def fetch_results_csv(url: str, dest: Path, fetcher=None) -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = (fetcher or _http_get)(url)
    dest.write_text(text, encoding="utf-8")
    return dest
