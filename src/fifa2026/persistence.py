from __future__ import annotations
from pathlib import Path
import json
import joblib

def save_models(models_dir, ensemble, meta: dict) -> Path:
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, d / "ensemble.joblib")
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d

def load_models(models_dir):
    d = Path(models_dir)
    ensemble = joblib.load(d / "ensemble.joblib")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return ensemble, meta
