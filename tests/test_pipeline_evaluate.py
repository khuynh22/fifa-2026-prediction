import json
import pandas as pd
from fifa2026.config import load_config
from fifa2026.pipeline import run_evaluate

def _synth_csv(tmp_path):
    rows = []
    d = pd.Timestamp("2011-01-01")
    seq = [("A","B",2,0),("B","C",1,1),("C","D",0,2),("D","A",1,0)]
    for i in range(80):
        h,a,hs,as_ = seq[i % 4]
        rows.append({"date": d + pd.Timedelta(days=60*i), "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_, "tournament": "Friendly",
                     "city": "", "country": "", "neutral": True})
    p = tmp_path / "results.csv"; pd.DataFrame(rows).to_csv(p, index=False); return p

def test_run_evaluate_writes_metrics(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    out = run_evaluate(cfg, matches_csv=_synth_csv(tmp_path))
    assert "metrics" in out and "log_loss" in out["metrics"]
    assert (cfg.reports_dir / "evaluation.json").exists()
