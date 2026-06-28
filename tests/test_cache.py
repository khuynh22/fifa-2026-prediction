from fifa2026.cache import DiskCache

def test_cache_stores_and_returns(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("k1") is None
    cache.put("k1", "hello")
    assert cache.get("k1") == "hello"

def test_get_or_fetch_only_fetches_once(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []
    def fetch():
        calls.append(1)
        return "payload"
    assert cache.get_or_fetch("k2", fetch) == "payload"
    assert cache.get_or_fetch("k2", fetch) == "payload"
    assert len(calls) == 1  # second call served from cache
