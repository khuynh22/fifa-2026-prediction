from fifa2026.persistence import save_models, load_models

class _Dummy:
    def __init__(self, w): self.weight = w
    def __eq__(self, other): return isinstance(other, _Dummy) and other.weight == self.weight

def test_save_load_roundtrip(tmp_path):
    ens = _Dummy(0.42)
    save_models(tmp_path, ens, {"feature_cols": ["elo_diff"], "trained_on": "2010-2026"})
    loaded, meta = load_models(tmp_path)
    assert loaded == ens
    assert meta["feature_cols"] == ["elo_diff"]
