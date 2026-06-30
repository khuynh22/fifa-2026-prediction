import itertools
import json
import numpy as np
import pandas as pd
from fifa2026.config import load_config
from fifa2026.pipeline import run_train, run_predict


def _synth_csv(tmp_path):
    teams = ["Germany", "Paraguay", "France", "Sweden", "Canada", "South Africa", "Netherlands", "Morocco"]
    # Self-consistent Elo generation: outcomes are drawn from the CURRENT Elo
    # win-probability at each match, so the elo_diff in the training matrix IS
    # the signal that predicted each result.  This breaks the multicollinearity
    # that arises from deterministic quality-based data (where every feature
    # correlates perfectly with quality), giving the Poisson GLM and LightGBM
    # a clean elo_diff → outcome signal.
    # A quality prior seeds the tie-breaking at the very first cycle.
    quality = {"Germany": 4, "Paraguay": 2, "France": 7, "Sweden": 5,
               "Canada": 3, "South Africa": 1, "Netherlands": 6, "Morocco": 8}
    elo = {t: 1500.0 + 5.0 * quality[t] for t in teams}  # small quality seed
    K = 40.0
    rng = np.random.default_rng(0)
    rows = []
    d = pd.Timestamp("2011-01-01")
    for i, (h, a) in enumerate(itertools.cycle(itertools.permutations(teams, 2))):
        if i >= 560:
            break
        diff = elo[h] - elo[a]
        p_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        if rng.random() < p_home:
            home_score, away_score = 2, 1
        else:
            home_score, away_score = 1, 2
        rows.append({"date": d + pd.Timedelta(days=2 * i), "home_team": h, "away_team": a,
                     "home_score": home_score, "away_score": away_score,
                     "tournament": "Friendly", "city": "", "country": "", "neutral": True})
        score_h = 1.0 if home_score > away_score else 0.0
        exp_h = p_home
        elo[h] += K * (score_h - exp_h)
        elo[a] += K * ((1 - score_h) - (1 - exp_h))
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
    empty = tmp_path / "none.yaml"; empty.write_text("injuries: {}\n", encoding="utf-8")
    res = run_predict(cfg, matches_csv=csv, bracket_path=bracket, injuries_path=empty)
    assert abs(sum(res.champion_probs.values()) - 1.0) < 1e-6
    # Pinned winners cannot be eliminated by their decided opponent; losers are out.
    assert res.champion_probs["South Africa"] == 0.0  # lost a pinned tie
    out = json.loads((cfg.reports_dir / "prediction.json").read_text())
    assert "champion_probs" in out


def test_injuries_lower_champion_prob(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "models_dir", tmp_path / "models")
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    csv = _synth_csv(tmp_path)          # reuse this module's existing 8-team synth helper
    run_train(cfg, matches_csv=csv)
    teams8 = ["Germany","Paraguay","France","Sweden","Canada","South Africa","Netherlands","Morocco"]
    bracket = tmp_path / "bracket.yaml"
    bracket.write_text("teams:\n" + "".join(f"  - {t}\n" for t in teams8), encoding="utf-8")
    empty = tmp_path / "none.yaml"; empty.write_text("injuries: {}\n", encoding="utf-8")
    hurt = tmp_path / "hurt.yaml"
    hurt.write_text("injuries:\n  France: [P1, P2, P3, P4]\n", encoding="utf-8")
    base = run_predict(cfg, matches_csv=csv, bracket_path=bracket, injuries_path=empty)
    inj = run_predict(cfg, matches_csv=csv, bracket_path=bracket, injuries_path=hurt)
    assert inj.champion_probs["France"] < base.champion_probs["France"]
    assert abs(sum(inj.champion_probs.values()) - 1.0) < 1e-6
    assert inj.meta["availability"]["France"]["elo_penalty"] == -40.0


def test_predicted_path_present(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "models_dir", tmp_path / "models")
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    csv = _synth_csv(tmp_path)
    run_train(cfg, matches_csv=csv)
    teams8 = ["Germany", "Paraguay", "France", "Sweden", "Canada", "South Africa", "Netherlands", "Morocco"]
    bracket = tmp_path / "bracket.yaml"
    bracket.write_text("teams:\n" + "".join(f"  - {t}\n" for t in teams8), encoding="utf-8")
    empty = tmp_path / "none.yaml"; empty.write_text("injuries: {}\n", encoding="utf-8")
    res = run_predict(cfg, matches_csv=csv, bracket_path=bracket, injuries_path=empty)
    path = res.meta["predicted_path"]
    assert [len(r["matches"]) for r in path["rounds"]] == [4, 2, 1]   # 8-team bracket
    assert path["champion"] in teams8
    # the champion is the winner of the Final
    assert path["rounds"][-1]["matches"][0]["winner"] == path["champion"]
