import pandas as pd
from fifa2026.config import load_config
from fifa2026.pipeline import run_train
from fifa2026.persistence import load_models

def _synth_csv(tmp_path):
    rng = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=60, freq="30D"),
        "home_team": (["A","B","C","D"] * 15),
        "away_team": (["B","C","D","A"] * 15),
        "home_score": ([2,1,0,1] * 15),
        "away_score": ([0,1,2,1] * 15),
        "tournament": ["Friendly"] * 60,
        "city": [""] * 60, "country": [""] * 60, "neutral": [True] * 60,
    })
    p = tmp_path / "results.csv"
    rng.to_csv(p, index=False)
    return p

def test_run_train_persists_loadable_models(tmp_path, monkeypatch):
    cfg = load_config()
    # redirect output dirs into tmp
    object.__setattr__(cfg, "models_dir", tmp_path / "models")
    csv = _synth_csv(tmp_path)
    run_train(cfg, matches_csv=csv)
    ensemble, meta = load_models(cfg.models_dir)
    assert hasattr(ensemble, "predict_proba")
    assert "feature_cols" in meta and len(meta["feature_cols"]) > 0
