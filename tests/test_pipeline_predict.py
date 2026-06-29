import json
import pandas as pd
from fifa2026.config import load_config
from fifa2026.pipeline import run_train, run_predict


def _synth_csv(tmp_path):
    teams = ["Germany", "Paraguay", "France", "Sweden", "Canada", "South Africa", "Netherlands", "Morocco"]
    rows = []
    import itertools
    d = pd.Timestamp("2011-01-01")
    for i, (h, a) in enumerate(itertools.cycle(itertools.permutations(teams, 2))):
        if i >= 200:
            break
        rows.append({"date": d + pd.Timedelta(days=7 * i), "home_team": h, "away_team": a,
                     "home_score": (i % 3), "away_score": ((i + 1) % 3),
                     "tournament": "Friendly", "city": "", "country": "", "neutral": True})
    p = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_run_predict_sums_to_one_and_pins(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "models_dir", tmp_path / "models")
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    csv = _synth_csv(tmp_path)
    run_train(cfg, matches_csv=csv)
    bracket = tmp_path / "bracket.yaml"
    bracket.write_text(
        "teams:\n" + "".join(f"  - {t}\n" for t in
            ["Germany", "Paraguay", "France", "Sweden", "Canada", "South Africa", "Netherlands", "Morocco"]) +
        "decided:\n  - {winner: Canada, loser: South Africa}\n", encoding="utf-8")
    res = run_predict(cfg, matches_csv=csv, bracket_path=bracket)
    assert abs(sum(res.champion_probs.values()) - 1.0) < 1e-6
    # Pinned winners cannot be eliminated by their decided opponent; losers are out.
    assert res.champion_probs["South Africa"] == 0.0  # lost a pinned tie
    out = json.loads((cfg.reports_dir / "prediction.json").read_text())
    assert "champion_probs" in out
