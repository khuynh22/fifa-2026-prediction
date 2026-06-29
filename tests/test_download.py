from pathlib import Path
from fifa2026.ingest.download import fetch_results_csv

def test_fetch_writes_then_caches(tmp_path):
    dest = tmp_path / "results.csv"
    calls = []
    def fake_fetch(url):
        calls.append(url)
        return "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    p = fetch_results_csv("http://x/results.csv", dest, fetcher=fake_fetch)
    assert p == dest and dest.exists()
    # second call is served from cache (no second fetch)
    fetch_results_csv("http://x/results.csv", dest, fetcher=fake_fetch)
    assert len(calls) == 1
