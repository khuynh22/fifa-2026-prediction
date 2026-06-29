# FIFA 2026 End-to-End Real Prediction + Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the tested `fifa2026` library to real data and produce a real 2026 World Cup champion forecast from the current Round-of-32 state, presented in an interactive Streamlit demo with charts and data.

**Architecture:** Thin orchestration over already-tested units. New code = real-data downloaders, model persistence, a `pipeline` module (`run_train`/`run_evaluate`/`run_predict`), pure Plotly chart builders (`viz`), and a thin `app.py`. The real 2026 bracket (with pinned decided ties) drives the existing exact bracket DP.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, lightgbm, statsmodels, pyyaml, requests, joblib, plotly, streamlit, pytest.

## Global Constraints

- Python **3.11+**, package `fifa2026`, `src/` layout. Use the project venv `.venv/Scripts/python.exe` for all commands.
- Point-in-time correctness remains the central invariant; **Tier-B squad features are prediction-only** (`build_training_matrix` raises if `squad_agg` is not None — do not bypass).
- Outcome encoding `0/1/2 = [home win, draw, away win]`; every `predict_proba` returns `(n,3)` in that order.
- Bookmaker odds are a **benchmark only**, never a model input.
- Temporal validation only. Training window `date >= 2010-01-01`. Deterministic seed `RANDOM_SEED = 42`.
- API path must **degrade gracefully**: missing `FOOTBALL_API_KEY`/failed fetch → continue with team-strength features only, never crash the pipeline.
- Reference data lives under `data/reference/` (tracked; only `data/raw/` and `data/processed/` are gitignored). Raw downloads + API responses are cached and not committed.
- Every task ends green (`.venv/Scripts/python.exe -m pytest -q`) and is committed.

---

## Shared Data Contracts

**Existing (do not change), confirmed signatures:**
- `fifa2026.config.load_config(path=None) -> Config` (attrs: `raw_dir`, `processed_dir`, `models_dir`, `reports_dir`, `train_start`, `random_seed`, `raw` dict).
- `fifa2026.ingest.matches.load_matches(csv_path, train_start=None) -> DataFrame`; `outcome(h,a) -> int`.
- `fifa2026.ingest.reference.load_confederations(csv) -> dict`; `load_venues(csv) -> DataFrame`; `is_host(team, hosts) -> bool`.
- `fifa2026.ingest.api_client.FootballAPI(base_url, api_key, cache).get_json(endpoint, params) -> dict`.
- `fifa2026.ingest.squads.parse_squad(payload) -> DataFrame`; `team_player_table(payloads) -> DataFrame`.
- `fifa2026.features.elo.EloEngine(...).fit(matches)`, `.rating_before(team, date)`.
- `fifa2026.features.form.FormFeatures().fit(matches)`.
- `fifa2026.features.context.ContextFeatures().fit(matches)`.
- `fifa2026.features.squad_features.team_aggregates(players) -> DataFrame`; `impute_tier_b(df) -> DataFrame`.
- `fifa2026.features.assemble.FeatureBuilder(elo, form, context, confederations, squad_agg, hosts, form_windows)`; `.row(home, away, date, venue_country, neutral) -> dict`; `.build_training_matrix(matches) -> (X, y, goals_home, goals_away)`.
- `fifa2026.models.poisson.PoissonModel().fit(X, y=None, sample_weight=None, goals_home=, goals_away=)`, `.predict_proba(X)`.
- `fifa2026.models.boosted.BoostedModel().fit(X, y)`, `.predict_proba(X)`.
- `fifa2026.models.ensemble.EnsembleModel(poisson, boosted, weight=0.5)`, `.predict_proba(X)`, `.tune_weight(X_val, y_val, grid=None)`.
- `fifa2026.knockout.resolve.resolve_tie(p_reg, pen_a=.5, pen_b=.5, depth_a=0, depth_b=0) -> float`.
- `fifa2026.knockout.bracket.load_bracket(path) -> list[str]`; `champion_probabilities(teams, win_prob) -> dict`; `predict_champion(teams, win_prob)`.
- `fifa2026.cli.build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None) -> win_prob(a,b)`.
- `fifa2026.evaluate.backtest.temporal_split(matches, cutoff) -> (train_idx, test_idx)`; `evaluate_probs(y_true, proba) -> dict`.
- `fifa2026.evaluate.benchmark.odds_to_probs(h,d,a)`; `compare_to_market(y_true, model_proba, market_proba) -> dict`.

**New contracts (defined by this plan):**
- `PredictionResult` dataclass: `champion_probs: dict[str,float]`, `round_probs: dict[str,dict[str,float]]`, `tie_probs: list[dict]`, `as_of: str`, `meta: dict`. `.to_dict()` / `from_dict(d)` for JSON.
- `round_probabilities(teams, win_prob) -> dict[str, dict[str,float]]` with keys `reach_R16, reach_QF, reach_SF, reach_final, win`.

---

## File Structure

```
src/fifa2026/
  ingest/download.py        # fetch_results_csv (cached download)            [T2]
  ingest/odds.py            # parse_winner_odds, implied_champion_probs      [T3]
  persistence.py            # save_models / load_models (joblib)             [T4]
  knockout/bracket.py       # + round_probabilities                          [T5]
  squad_enrich.py           # build_squad_agg (API, graceful fallback)       [T9]
  pipeline.py               # PredictionResult, run_train/evaluate/predict   [T6,T7,T8]
  cli.py                    # flesh out command bodies + decided win_prob    [T7,T11]
  viz.py                    # Plotly chart builders                          [T10]
config/bracket_2026.yaml    # REAL 32-team bracket + decided ties            [T5]
data/reference/confederations.csv, venues_2026.csv, market_odds_2026.csv     [T1,T8]
.env.example                                                                 [T1]
app.py                      # Streamlit demo (repo root)                     [T12]
Makefile                    # + app/report/all targets                       [T12]
tests/...                   # one test module per new unit
```

---

### Task 1: Dependencies, reference data, and `.env.example`

**Files:**
- Modify: `pyproject.toml`
- Create: `data/reference/confederations.csv`, `data/reference/venues_2026.csv`, `.env.example`, `tests/test_reference_data.py`

**Interfaces:**
- Produces: committed reference CSVs loadable by the existing `load_confederations` / `load_venues`.

- [ ] **Step 1: Add deps to `pyproject.toml`** — add to the `dependencies` list: `"joblib>=1.3"`, `"plotly>=5.18"`, `"streamlit>=1.30"`. Then install:

Run: `.venv/Scripts/python.exe -m pip install -e ".[dev]" joblib plotly streamlit`
Expected: installs succeed.

- [ ] **Step 2: Create `data/reference/confederations.csv`** (the 2026 field; `team,confederation`):

```csv
team,confederation
United States,CONCACAF
Mexico,CONCACAF
Canada,CONCACAF
Brazil,CONMEBOL
Argentina,CONMEBOL
Colombia,CONMEBOL
Ecuador,CONMEBOL
Paraguay,CONMEBOL
France,UEFA
Germany,UEFA
Netherlands,UEFA
Spain,UEFA
Portugal,UEFA
England,UEFA
Belgium,UEFA
Croatia,UEFA
Switzerland,UEFA
Austria,UEFA
Sweden,UEFA
Norway,UEFA
Bosnia and Herzegovina,UEFA
Morocco,CAF
Senegal,CAF
Ivory Coast,CAF
Algeria,CAF
Ghana,CAF
South Africa,CAF
Cape Verde,CAF
Egypt,CAF
DR Congo,CAF
Japan,AFC
Australia,AFC
```

- [ ] **Step 3: Create `data/reference/venues_2026.csv`** (`city,country,lat,lon,altitude_m` — the 2026 hosts/major venues):

```csv
city,country,lat,lon,altitude_m
Mexico City,Mexico,19.43,-99.13,2240
Guadalajara,Mexico,20.67,-103.35,1566
Monterrey,Mexico,25.69,-100.32,540
Toronto,Canada,43.65,-79.38,76
Vancouver,Canada,49.28,-123.12,4
Inglewood,United States,33.95,-118.34,38
East Rutherford,United States,40.81,-74.07,2
Arlington,United States,32.75,-97.08,184
Houston,United States,29.76,-95.37,24
Foxborough,United States,42.09,-71.26,86
Atlanta,United States,33.75,-84.39,320
Miami Gardens,United States,25.96,-80.24,3
Kansas City,United States,39.10,-94.58,277
Philadelphia,United States,39.95,-75.17,12
Seattle,United States,47.61,-122.33,53
San Francisco Bay Area,United States,37.40,-121.97,9
```

- [ ] **Step 4: Create `.env.example`**:

```bash
# Copy to .env and fill in. Optional — the pipeline runs free-data-only without it.
FOOTBALL_API_KEY=your_api_football_key_here
```

- [ ] **Step 5: Write the failing test** — `tests/test_reference_data.py`:

```python
from pathlib import Path
from fifa2026.ingest.reference import load_confederations, load_venues

REF = Path(__file__).resolve().parents[1] / "data" / "reference"

def test_confederations_cover_2026_hosts():
    conf = load_confederations(REF / "confederations.csv")
    for host in ("United States", "Mexico", "Canada"):
        assert conf[host] == "CONCACAF"
    assert conf["Brazil"] == "CONMEBOL"
    assert conf["Morocco"] == "CAF"
    assert len(conf) >= 32

def test_venues_have_mexico_city_altitude():
    venues = load_venues(REF / "venues_2026.csv")
    mc = venues[venues["city"] == "Mexico City"].iloc[0]
    assert mc["altitude_m"] == 2240
```

- [ ] **Step 6: Run test, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reference_data.py -q`
Expected: PASS (files exist and parse).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml data/reference .env.example tests/test_reference_data.py
git commit -m "feat: add deps, 2026 reference data, and .env.example"
```

---

### Task 2: Results downloader (cached)

**Files:**
- Create: `src/fifa2026/ingest/download.py`, `tests/test_download.py`

**Interfaces:**
- Produces: `fetch_results_csv(url: str, dest: Path, fetcher=None) -> Path`. If `dest` exists, returns it without downloading (cache). Otherwise calls `fetcher(url) -> str` (defaults to a `requests.get(...).text` wrapper), writes the text to `dest`, returns `dest`. `fetcher` is injectable so tests never hit the network.

- [ ] **Step 1: Write the failing test** — `tests/test_download.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails** — `pytest tests/test_download.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `src/fifa2026/ingest/download.py`**:

```python
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
```

- [ ] **Step 4: Run test, verify pass** — `pytest tests/test_download.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/ingest/download.py tests/test_download.py
git commit -m "feat: cached international-results downloader"
```

---

### Task 3: Winner-odds parsing + implied champion probabilities

**Files:**
- Create: `src/fifa2026/ingest/odds.py`, `tests/test_odds.py`

**Interfaces:**
- Produces: `parse_winner_odds(rows: list[dict]) -> dict[str, float]` (each row `{"team","decimal_odds"}` → `{team: decimal_odds}`); `implied_champion_probs(odds: dict[str,float]) -> dict[str,float]` (reciprocals normalized across the field to sum to 1 — de-vig).

- [ ] **Step 1: Write the failing test** — `tests/test_odds.py`:

```python
from fifa2026.ingest.odds import parse_winner_odds, implied_champion_probs

def test_parse_and_implied():
    rows = [{"team": "Brazil", "decimal_odds": 5.0},
            {"team": "France", "decimal_odds": 6.0},
            {"team": "Spain", "decimal_odds": 7.0}]
    odds = parse_winner_odds(rows)
    assert odds["Brazil"] == 5.0
    probs = implied_champion_probs(odds)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["Brazil"] > probs["Spain"]  # shorter odds -> higher prob
```

- [ ] **Step 2: Run test, verify it fails.**

- [ ] **Step 3: Implement `src/fifa2026/ingest/odds.py`**:

```python
from __future__ import annotations

def parse_winner_odds(rows: list[dict]) -> dict[str, float]:
    return {r["team"]: float(r["decimal_odds"]) for r in rows}

def implied_champion_probs(odds: dict[str, float]) -> dict[str, float]:
    raw = {t: 1.0 / o for t, o in odds.items() if o and o > 0}
    total = sum(raw.values())
    if total <= 0:
        return {t: 0.0 for t in odds}
    return {t: v / total for t, v in raw.items()}
```

- [ ] **Step 4: Run test, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/ingest/odds.py tests/test_odds.py
git commit -m "feat: winner-odds parsing and de-vigged champion probabilities"
```

---

### Task 4: Model persistence (joblib)

**Files:**
- Create: `src/fifa2026/persistence.py`, `tests/test_persistence.py`

**Interfaces:**
- Produces: `save_models(models_dir, ensemble, meta: dict) -> Path` (writes `ensemble.joblib` + `meta.json`); `load_models(models_dir) -> tuple[object, dict]` returns `(ensemble, meta)`.

- [ ] **Step 1: Write the failing test** — `tests/test_persistence.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails.**

- [ ] **Step 3: Implement `src/fifa2026/persistence.py`**:

```python
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
```

- [ ] **Step 4: Run test, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/persistence.py tests/test_persistence.py
git commit -m "feat: joblib model persistence"
```

---

### Task 5: Real bracket + per-round survival + ordering validation

**Files:**
- Modify: `src/fifa2026/knockout/bracket.py`, `config/bracket_2026.yaml`
- Create: `tests/test_round_probabilities.py`, `tests/test_bracket_ordering.py`

**Interfaces:**
- Produces: `round_probabilities(teams, win_prob) -> dict[str, dict[str,float]]` with keys `reach_R16, reach_QF, reach_SF, reach_final, win` (a team's probability of reaching each stage). Reuses `_solve` on chunks of size 2/4/8/16/32.

- [ ] **Step 1: Rewrite `config/bracket_2026.yaml`** with the REAL bracket (32 teams in DP-bisection order derived from the M73→M103 tree) plus pinned decided ties:

```yaml
# 2026 FIFA World Cup Round-of-32 bracket (as of 2026-06-28), in bracket-slot
# order so the recursive bracket DP reproduces the real M73->M103 adjacency.
# Source: en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
as_of: "2026-06-28"
teams:
  - Germany          # M74
  - Paraguay
  - France           # M77
  - Sweden
  - Canada           # M73
  - South Africa
  - Netherlands      # M75
  - Morocco
  - Portugal         # M83
  - Croatia
  - Spain            # M84
  - Austria
  - United States    # M81
  - Bosnia and Herzegovina
  - Belgium          # M82
  - Senegal
  - Brazil           # M76
  - Japan
  - Ivory Coast      # M78
  - Norway
  - Mexico           # M79
  - Ecuador
  - England          # M80
  - DR Congo
  - Argentina        # M86
  - Cape Verde
  - Australia        # M88
  - Egypt
  - Switzerland      # M85
  - Algeria
  - Colombia         # M87
  - Ghana
# Already-played R32 ties (pinned to certainty in the forecast).
decided:
  - {winner: Canada, loser: South Africa}
  - {winner: Germany, loser: Paraguay}
  - {winner: Netherlands, loser: Morocco}
  - {winner: France, loser: Sweden}
```

- [ ] **Step 2: Write the ordering-validation test** — `tests/test_bracket_ordering.py` (asserts the slot order reproduces the real R16 pairings):

```python
from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "config" / "bracket_2026.yaml"

def _r16_pairs(teams):
    # Each consecutive block of 4 forms one R16 match between the two ties' slots.
    pairs = []
    for i in range(0, len(teams), 4):
        block = teams[i:i+4]
        pairs.append(((block[0], block[1]), (block[2], block[3])))
    return pairs

def test_slot_order_reproduces_real_r16_pairings():
    teams = yaml.safe_load(CFG.read_text(encoding="utf-8"))["teams"]
    assert len(teams) == 32
    pairs = _r16_pairs(teams)
    # M89 = (M74 Germany/Paraguay) vs (M77 France/Sweden)
    assert pairs[0] == (("Germany", "Paraguay"), ("France", "Sweden"))
    # M90 = (M73 Canada/SA) vs (M75 Netherlands/Morocco)
    assert pairs[1] == (("Canada", "South Africa"), ("Netherlands", "Morocco"))
    # M91 = (M76 Brazil/Japan) vs (M78 Ivory Coast/Norway)
    assert pairs[4] == (("Brazil", "Japan"), ("Ivory Coast", "Norway"))
    # M95 = (M86 Argentina/Cape Verde) vs (M88 Australia/Egypt)
    assert pairs[6] == (("Argentina", "Cape Verde"), ("Australia", "Egypt"))
```

- [ ] **Step 3: Run it, verify pass** (it tests data only; should pass once the yaml is correct). Fix the yaml order if it fails.

- [ ] **Step 4: Write the failing test** — `tests/test_round_probabilities.py`:

```python
import numpy as np
from fifa2026.knockout.bracket import round_probabilities

def test_round_probabilities_structure_and_sums():
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strength = {t: s for t, s in zip(teams, [8, 1, 7, 2, 6, 3, 5, 4])}
    def win_prob(a, b):
        return 1.0 / (1.0 + np.exp(-(strength[a] - strength[b])))
    rp = round_probabilities(teams, win_prob)
    assert set(rp["A"]) == {"reach_R16", "reach_QF", "reach_SF", "reach_final", "win"}
    # 8 teams -> rounds present: reach_QF (win of size-2), reach_SF (size-4), reach_final?(size-8 == win)
    # Every team reaches its first knockout match with prob 1 conceptually; check monotonicity:
    for t in teams:
        assert rp[t]["reach_QF"] >= rp[t]["reach_SF"] - 1e-9
        assert rp[t]["reach_SF"] >= rp[t]["win"] - 1e-9
    # Champion probabilities (the "win" column) sum to 1.
    assert abs(sum(rp[t]["win"] for t in teams) - 1.0) < 1e-9
    # Strongest team most likely to win.
    assert max(teams, key=lambda t: rp[t]["win"]) == "A"
```

- [ ] **Step 5: Run it, verify it fails** (function missing).

- [ ] **Step 6: Implement `round_probabilities` in `src/fifa2026/knockout/bracket.py`** (append; reuses `_solve`):

```python
# Map subtree size -> the stage a team reaches by WINNING that subtree.
_ROUND_BY_SIZE = {2: "reach_QF", 4: "reach_SF", 8: "reach_final", 16: "win", 32: "win"}

def round_probabilities(teams: list[str], win_prob) -> dict[str, dict[str, float]]:
    """For each team, probability of reaching each knockout stage.

    Keys: reach_R16, reach_QF, reach_SF, reach_final, win. 'reach_R16' is 1.0 for
    all teams already in the Round of 32. Other stages are computed by solving the
    sub-bracket of the appropriate size around each team."""
    n = len(teams)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("bracket size must be a positive power of 2")
    out = {t: {"reach_R16": 1.0, "reach_QF": 0.0, "reach_SF": 0.0,
               "reach_final": 0.0, "win": 0.0} for t in teams}
    # For each level, partition into chunks of that size and solve each chunk.
    for size in (2, 4, 8, 16, 32):
        if size > n:
            break
        key = "win" if size == n else _ROUND_BY_SIZE[size]
        for i in range(0, n, size):
            chunk = teams[i:i + size]
            for t, p in _solve(chunk, win_prob).items():
                # take the deepest (largest-size) solve for each labeled stage
                out[t][key] = p
    return out
```

- [ ] **Step 7: Run both tests, verify pass.**

- [ ] **Step 8: Commit**

```bash
git add src/fifa2026/knockout/bracket.py config/bracket_2026.yaml tests/test_round_probabilities.py tests/test_bracket_ordering.py
git commit -m "feat: real 2026 bracket with pinned results and per-round survival"
```

---

### Task 6: `pipeline.run_train` + `PredictionResult`

**Files:**
- Create: `src/fifa2026/pipeline.py`, `tests/test_pipeline_train.py`

**Interfaces:**
- Produces: `PredictionResult` dataclass (`champion_probs`, `round_probs`, `tie_probs`, `as_of`, `meta`; `.to_dict()`, `from_dict(d)`); `run_train(cfg, matches_csv=None) -> Path` — loads matches, fits Elo/Form/Context, builds the training matrix (`squad_agg=None`), fits Poisson+LightGBM, tunes the ensemble on a temporal split, persists models, returns `models_dir`.

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline_train.py`:

```python
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
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Implement `src/fifa2026/pipeline.py`** (`PredictionResult` + `run_train`):

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
from fifa2026.ingest.matches import load_matches
from fifa2026.ingest.reference import load_confederations
from fifa2026.features.elo import EloEngine
from fifa2026.features.form import FormFeatures
from fifa2026.features.context import ContextFeatures
from fifa2026.features.assemble import FeatureBuilder
from fifa2026.models.poisson import PoissonModel
from fifa2026.models.boosted import BoostedModel
from fifa2026.models.ensemble import EnsembleModel
from fifa2026.evaluate.backtest import temporal_split
from fifa2026.persistence import save_models

@dataclass
class PredictionResult:
    champion_probs: dict
    round_probs: dict
    tie_probs: list = field(default_factory=list)
    as_of: str = ""
    meta: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)
    @staticmethod
    def from_dict(d: dict) -> "PredictionResult":
        return PredictionResult(**d)

def _ref_dir(cfg) -> Path:
    return Path("data/reference")

def build_feature_builder(cfg, matches, squad_agg=None):
    elo = EloEngine(
        k=cfg.raw["elo"]["k"], home_advantage=cfg.raw["elo"]["home_advantage"],
        initial=cfg.raw["elo"]["initial"]).fit(matches)
    form = FormFeatures().fit(matches)
    context = ContextFeatures().fit(matches)
    confed = load_confederations(_ref_dir(cfg) / "confederations.csv")
    return FeatureBuilder(elo=elo, form=form, context=context, confederations=confed,
                          squad_agg=squad_agg, hosts=cfg.raw["hosts_2026"],
                          form_windows=cfg.raw["features"]["form_windows"])

def run_train(cfg, matches_csv=None) -> Path:
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    fb = build_feature_builder(cfg, matches, squad_agg=None)
    X, y, gh, ga = fb.build_training_matrix(matches)
    # temporal split for ensemble weight tuning
    train_idx, val_idx = temporal_split(matches.dropna(subset=["home_score", "away_score"])
                                        .sort_values("date").reset_index(drop=True),
                                        cutoff=cfg.raw.get("val_cutoff", "2022-01-01"))
    poisson = PoissonModel().fit(X, y, goals_home=gh, goals_away=ga)
    boosted = BoostedModel().fit(X, y)
    ensemble = EnsembleModel(poisson, boosted)
    if len(val_idx) > 0:
        ensemble.tune_weight(X.iloc[val_idx], y[val_idx])
    save_models(cfg.models_dir, ensemble,
                {"feature_cols": list(X.columns), "trained_on": cfg.train_start,
                 "weight": ensemble.weight})
    return Path(cfg.models_dir)
```

> Add `val_cutoff: "2022-01-01"` to `config/default.yaml` (top level) in this task.

- [ ] **Step 4: Run it, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/pipeline.py config/default.yaml tests/test_pipeline_train.py
git commit -m "feat: training pipeline with PredictionResult and model persistence"
```

---

### Task 7: `pipeline.run_predict` + decided-result win_prob

**Files:**
- Modify: `src/fifa2026/cli.py` (extend `build_win_prob` with `decided`), `src/fifa2026/pipeline.py`
- Create: `tests/test_pipeline_predict.py`

**Interfaces:**
- Modify: `build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None)` — `decided` is a dict `{frozenset({teamA,teamB}): winner}`; when a tie is decided, `win_prob` returns `1.0`/`0.0` before any model call.
- Produces: `run_predict(cfg, models=None) -> PredictionResult` — loads bracket + decided from `config/bracket_2026.yaml`, builds the feature builder (squad_agg via Task 9 if available, else None), builds `win_prob`, computes `champion_probabilities`, `round_probabilities`, and per-tie probs, writes `reports/prediction.json`, returns the result.

- [ ] **Step 1: Extend `build_win_prob` in `cli.py`** — add `decided` handling. Insert at the top of the inner `win_prob`:

```python
def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None):
    pen = pen or {}
    depth = depth or {}
    decided = decided or {}
    hosts = getattr(feature_builder, "hosts", []) or []

    def _venue(team_a, team_b):
        if team_a in hosts:
            return team_a
        if team_b in hosts:
            return team_b
        return ""

    def win_prob(team_a: str, team_b: str) -> float:
        key = frozenset((team_a, team_b))
        if key in decided:
            return 1.0 if decided[key] == team_a else 0.0
        venue = _venue(team_a, team_b)
        row_ab = feature_builder.row(team_a, team_b, as_of_date, venue_country=venue, neutral=(venue == ""))
        row_ba = feature_builder.row(team_b, team_a, as_of_date, venue_country=venue, neutral=(venue == ""))
        p_ab = resolve_tie(model.predict_proba(pd.DataFrame([row_ab]))[0],
                           pen_a=pen.get(team_a, 0.5), pen_b=pen.get(team_b, 0.5),
                           depth_a=depth.get(team_a, 0.0), depth_b=depth.get(team_b, 0.0))
        p_ba = resolve_tie(model.predict_proba(pd.DataFrame([row_ba]))[0],
                           pen_a=pen.get(team_b, 0.5), pen_b=pen.get(team_a, 0.5),
                           depth_a=depth.get(team_b, 0.0), depth_b=depth.get(team_a, 0.0))
        return 0.5 * (p_ab + (1.0 - p_ba))
    return win_prob
```

- [ ] **Step 2: Write the failing test** — `tests/test_pipeline_predict.py`:

```python
import json
import pandas as pd
from fifa2026.config import load_config
from fifa2026.pipeline import run_train, run_predict

def _synth_csv(tmp_path):
    teams = ["Germany","Paraguay","France","Sweden","Canada","South Africa","Netherlands","Morocco"]
    rows = []
    import itertools
    d = pd.Timestamp("2011-01-01")
    for i, (h, a) in enumerate(itertools.cycle(itertools.permutations(teams, 2))):
        if i >= 200: break
        rows.append({"date": d + pd.Timedelta(days=7*i), "home_team": h, "away_team": a,
                     "home_score": (i % 3), "away_score": ((i+1) % 3),
                     "tournament": "Friendly", "city": "", "country": "", "neutral": True})
    p = tmp_path / "results.csv"; pd.DataFrame(rows).to_csv(p, index=False); return p

def test_run_predict_sums_to_one_and_pins(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "models_dir", tmp_path / "models")
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    csv = _synth_csv(tmp_path)
    run_train(cfg, matches_csv=csv)
    res = run_predict(cfg, matches_csv=csv)
    assert abs(sum(res.champion_probs.values()) - 1.0) < 1e-6
    # Pinned winners cannot be eliminated by their decided opponent; losers are out.
    assert res.champion_probs["South Africa"] == 0.0  # lost a pinned tie
    out = json.loads((cfg.reports_dir / "prediction.json").read_text())
    assert "champion_probs" in out
```

> Note: this test uses an 8-team slice; `run_predict` must accept a `matches_csv` override and a bracket override so the test can use the 8-team subset. Implement `run_predict(cfg, models=None, matches_csv=None, bracket_path=None)`; default `bracket_path` to `config/bracket_2026.yaml`. For the test, also write a tiny 8-team bracket yaml to tmp and pass it.

Replace the test body's predict call + bracket with:

```python
    bracket = tmp_path / "bracket.yaml"
    bracket.write_text(
        "teams:\n" + "".join(f"  - {t}\n" for t in
            ["Germany","Paraguay","France","Sweden","Canada","South Africa","Netherlands","Morocco"]) +
        "decided:\n  - {winner: Canada, loser: South Africa}\n", encoding="utf-8")
    res = run_predict(cfg, matches_csv=csv, bracket_path=bracket)
```

- [ ] **Step 3: Run it, verify it fails.**

- [ ] **Step 4: Implement `run_predict` in `pipeline.py`** (append):

```python
import json
import yaml
from fifa2026.cli import build_win_prob
from fifa2026.knockout.bracket import champion_probabilities, round_probabilities

def _load_bracket_cfg(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    teams = list(data["teams"])
    decided = {frozenset((d["winner"], d["loser"])): d["winner"]
               for d in data.get("decided", [])}
    return teams, decided, data.get("as_of", "")

def run_predict(cfg, models=None, matches_csv=None, bracket_path=None, squad_agg=None) -> PredictionResult:
    from fifa2026.persistence import load_models
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    ensemble = models if models is not None else load_models(cfg.models_dir)[0]
    fb = build_feature_builder(cfg, matches, squad_agg=squad_agg)
    bpath = bracket_path or cfg.raw.get("bracket_path", "config/bracket_2026.yaml")
    teams, decided, as_of = _load_bracket_cfg(bpath)
    as_of_date = pd.Timestamp(as_of) if as_of else matches["date"].max()
    win_prob = build_win_prob(ensemble, fb, as_of_date, decided=decided)
    champ = champion_probabilities(teams, win_prob)
    rounds = round_probabilities(teams, win_prob)
    ties = [{"home": teams[i], "away": teams[i+1],
             "p_home": win_prob(teams[i], teams[i+1])} for i in range(0, len(teams), 2)]
    result = PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties,
                              as_of=as_of, meta={"n_teams": len(teams)})
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "prediction.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 5: Run it, verify pass.**

- [ ] **Step 6: Commit**

```bash
git add src/fifa2026/cli.py src/fifa2026/pipeline.py tests/test_pipeline_predict.py
git commit -m "feat: prediction pipeline with pinned decided ties"
```

---

### Task 8: `pipeline.run_evaluate` + market comparison

**Files:**
- Modify: `src/fifa2026/pipeline.py`
- Create: `data/reference/market_odds_2026.csv`, `tests/test_pipeline_evaluate.py`

**Interfaces:**
- Produces: `run_evaluate(cfg, matches_csv=None) -> dict` — temporal backtest (train on `< val_cutoff`, test on `>=`), returns `{"metrics": {...}, "calibration": [...], "market": {...}}` and writes `reports/evaluation.json`. Market section compares model champion probs to `implied_champion_probs` of `market_odds_2026.csv` (overlap teams), or `{}` if the file is absent.

- [ ] **Step 1: Create a starter `data/reference/market_odds_2026.csv`** (real values filled in Task 13; seed with a few rows so the path is testable):

```csv
team,decimal_odds
Spain,5.5
France,6.0
Argentina,7.0
England,8.0
Brazil,9.0
```

- [ ] **Step 2: Write the failing test** — `tests/test_pipeline_evaluate.py`:

```python
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
        rows.append({"date": d + pd.Timedelta(days=20*i), "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_, "tournament": "Friendly",
                     "city": "", "country": "", "neutral": True})
    p = tmp_path / "results.csv"; pd.DataFrame(rows).to_csv(p, index=False); return p

def test_run_evaluate_writes_metrics(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "reports_dir", tmp_path / "reports")
    out = run_evaluate(cfg, matches_csv=_synth_csv(tmp_path))
    assert "metrics" in out and "log_loss" in out["metrics"]
    assert (cfg.reports_dir / "evaluation.json").exists()
```

- [ ] **Step 3: Run it, verify it fails.**

- [ ] **Step 4: Implement `run_evaluate` in `pipeline.py`** (append):

```python
import numpy as np
from fifa2026.evaluate.backtest import evaluate_probs
from fifa2026.ingest.odds import implied_champion_probs

def run_evaluate(cfg, matches_csv=None) -> dict:
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    m = matches.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    cutoff = cfg.raw.get("val_cutoff", "2022-01-01")
    train_idx, test_idx = temporal_split(m, cutoff=cutoff)
    fb = build_feature_builder(cfg, m.iloc[train_idx], squad_agg=None)
    Xtr, ytr, gh, ga = fb.build_training_matrix(m.iloc[train_idx])
    poisson = PoissonModel().fit(Xtr, ytr, goals_home=gh, goals_away=ga)
    boosted = BoostedModel().fit(Xtr, ytr)
    ensemble = EnsembleModel(poisson, boosted)
    if len(test_idx) == 0:
        result = {"metrics": {}, "calibration": [], "market": {}}
    else:
        Xte = pd.DataFrame([fb.row(r["home_team"], r["away_team"], r["date"],
                                   r.get("country", ""), bool(r.get("neutral", True)))
                            for _, r in m.iloc[test_idx].iterrows()]).fillna(0.0)
        from fifa2026.ingest.matches import outcome
        yte = np.array([outcome(int(r["home_score"]), int(r["away_score"]))
                        for _, r in m.iloc[test_idx].iterrows()])
        proba = ensemble.predict_proba(Xte[Xtr.columns])
        result = {"metrics": evaluate_probs(yte, proba), "calibration": [], "market": {}}
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 5: Run it, verify pass.**

- [ ] **Step 6: Commit**

```bash
git add src/fifa2026/pipeline.py data/reference/market_odds_2026.csv tests/test_pipeline_evaluate.py
git commit -m "feat: evaluation pipeline with temporal backtest"
```

---

### Task 9: API squad enrichment (graceful)

**Files:**
- Create: `src/fifa2026/squad_enrich.py`, `tests/test_squad_enrich.py`

**Interfaces:**
- Produces: `build_squad_agg(cfg, teams, api=None) -> pd.DataFrame | None` — when no `FOOTBALL_API_KEY` env var and no injected `api`, returns `None` (graceful fallback). Otherwise fetches each team's squad via the API, parses with `parse_squad`, aggregates with `team_aggregates`, imputes with `impute_tier_b`, returns the frame.

- [ ] **Step 1: Write the failing test** — `tests/test_squad_enrich.py`:

```python
import os
import pandas as pd
from fifa2026.config import load_config
from fifa2026.squad_enrich import build_squad_agg

class _FakeAPI:
    def get_json(self, endpoint, params):
        team = params["team"]
        return {"team": team, "players": [
            {"name": "P1", "position": "Attacker", "age": 25, "market_value": 5e7,
             "minutes": 2000, "xg": 10.0, "xa": 5.0, "injured": False}]}

def test_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    cfg = load_config()
    assert build_squad_agg(cfg, ["Brazil"], api=None) is None

def test_with_api_returns_aggregates():
    cfg = load_config()
    agg = build_squad_agg(cfg, ["Brazil", "France"], api=_FakeAPI())
    assert agg is not None
    assert "squad_value" in agg.columns
    assert "Brazil" in agg.index
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Implement `src/fifa2026/squad_enrich.py`**:

```python
from __future__ import annotations
import os
from fifa2026.ingest.squads import parse_squad, team_player_table
from fifa2026.features.squad_features import team_aggregates, impute_tier_b

def build_squad_agg(cfg, teams, api=None):
    if api is None:
        key = os.environ.get(cfg.raw["api"]["key_env"])
        if not key:
            return None  # graceful fallback: team-strength features only
        from fifa2026.cache import DiskCache
        from fifa2026.ingest.api_client import FootballAPI
        api = FootballAPI(cfg.raw["api"]["base_url"], key, DiskCache(cfg.raw_dir / "api_cache"))
    payloads = {}
    for team in teams:
        try:
            payloads[team] = api.get_json("players/squads", {"team": team})
        except Exception:
            continue  # skip teams the API can't resolve; never crash the pipeline
    if not payloads:
        return None
    players = team_player_table(payloads)
    return impute_tier_b(team_aggregates(players))
```

- [ ] **Step 4: Run it, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/squad_enrich.py tests/test_squad_enrich.py
git commit -m "feat: graceful API squad enrichment"
```

---

### Task 10: Plotly chart builders (`viz.py`)

**Files:**
- Create: `src/fifa2026/viz.py`, `tests/test_viz.py`

**Interfaces:**
- Consumes: a `PredictionResult` and the evaluation dict.
- Produces (each returns a `plotly.graph_objects.Figure`):
  `champion_bar(result, top_n=12, market=None)`, `bracket_figure(result)`,
  `survival_heatmap(result, top_n=16)`, `market_compare(model_probs, market_probs, top_n=10)`,
  `calibration_curve(calibration)`.

- [ ] **Step 1: Write the failing test** — `tests/test_viz.py`:

```python
from fifa2026.pipeline import PredictionResult
from fifa2026 import viz

def _result():
    champ = {"France": 0.25, "Brazil": 0.2, "Spain": 0.18, "Argentina": 0.15,
             "England": 0.12, "Germany": 0.1}
    rounds = {t: {"reach_R16": 1.0, "reach_QF": 0.6, "reach_SF": 0.4,
                  "reach_final": 0.3, "win": p} for t, p in champ.items()}
    ties = [{"home": "France", "away": "Brazil", "p_home": 0.55}]
    return PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties, as_of="2026-06-28")

def test_champion_bar_has_data():
    fig = viz.champion_bar(_result(), top_n=3)
    assert len(fig.data) >= 1
    # sorted descending, top team first
    assert list(fig.data[0].x)[0] == "France" or list(fig.data[0].y)[0] == "France"

def test_market_compare_two_series():
    fig = viz.market_compare({"France": 0.25, "Brazil": 0.2},
                             {"France": 0.2, "Brazil": 0.25}, top_n=2)
    assert len(fig.data) == 2  # model + market

def test_survival_heatmap_builds():
    fig = viz.survival_heatmap(_result(), top_n=5)
    assert fig is not None and len(fig.data) >= 1
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Implement `src/fifa2026/viz.py`**:

```python
from __future__ import annotations
import plotly.graph_objects as go

def _sorted_items(d, top_n):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

def champion_bar(result, top_n=12, market=None) -> go.Figure:
    items = _sorted_items(result.champion_probs, top_n)
    teams = [t for t, _ in items]
    fig = go.Figure([go.Bar(x=teams, y=[p for _, p in items], name="Model")])
    if market:
        fig.add_bar(x=teams, y=[market.get(t, 0.0) for t in teams], name="Market")
    fig.update_layout(title="Champion probability", yaxis_tickformat=".0%", barmode="group")
    return fig

def bracket_figure(result) -> go.Figure:
    ties = result.tie_probs
    labels = [f'{t["home"]} vs {t["away"]}' for t in ties]
    fig = go.Figure([go.Bar(x=[t["p_home"] for t in ties], y=labels, orientation="h")])
    fig.update_layout(title="Round-of-32 ties: P(first team advances)", xaxis_tickformat=".0%")
    return fig

def survival_heatmap(result, top_n=16) -> go.Figure:
    stages = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]
    items = _sorted_items(result.champion_probs, top_n)
    teams = [t for t, _ in items]
    z = [[result.round_probs[t][s] for s in stages] for t in teams]
    fig = go.Figure(go.Heatmap(z=z, x=stages, y=teams, colorscale="Blues",
                               zmin=0, zmax=1))
    fig.update_layout(title="Per-round survival probability")
    return fig

def market_compare(model_probs, market_probs, top_n=10) -> go.Figure:
    teams = [t for t, _ in _sorted_items(model_probs, top_n)]
    fig = go.Figure()
    fig.add_bar(x=teams, y=[model_probs.get(t, 0.0) for t in teams], name="Model")
    fig.add_bar(x=teams, y=[market_probs.get(t, 0.0) for t in teams], name="Market")
    fig.update_layout(title="Model vs market", barmode="group", yaxis_tickformat=".0%")
    return fig

def calibration_curve(calibration) -> go.Figure:
    fig = go.Figure()
    if calibration:
        xs = [pt["pred"] for pt in calibration]
        ys = [pt["obs"] for pt in calibration]
        fig.add_scatter(x=xs, y=ys, mode="markers+lines", name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                    line=dict(dash="dash"))
    fig.update_layout(title="Calibration", xaxis_title="Predicted", yaxis_title="Observed")
    return fig
```

- [ ] **Step 4: Run it, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/viz.py tests/test_viz.py
git commit -m "feat: Plotly chart builders for the demo"
```

---

### Task 11: Wire CLI command bodies

**Files:**
- Modify: `src/fifa2026/cli.py`
- Create: `tests/test_cli_commands.py`

**Interfaces:**
- `data` downloads results.csv; `train` calls `run_train`; `evaluate` calls `run_evaluate`; `predict` builds squad_agg (graceful) and calls `run_predict`, printing the top champion picks.

- [ ] **Step 1: Write the failing test** — `tests/test_cli_commands.py`:

```python
from fifa2026 import cli

def test_cli_dispatch_table_has_all_commands():
    # main() builds a dispatch with real handlers (not the old print-stub)
    assert hasattr(cli, "_cmd_train") and hasattr(cli, "_cmd_predict")
    assert hasattr(cli, "_cmd_data") and hasattr(cli, "_cmd_evaluate")
```

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Implement command bodies in `cli.py`** (replace `_cmd_predict` and the dispatch):

```python
from fifa2026.config import load_config

def _cmd_data(args):
    from fifa2026.ingest.download import fetch_results_csv, RESULTS_URL
    cfg = load_config(args.config)
    dest = cfg.raw_dir / "results.csv"
    fetch_results_csv(RESULTS_URL, dest)
    print(f"data ready: {dest}")

def _cmd_train(args):
    from fifa2026.pipeline import run_train
    cfg = load_config(args.config)
    out = run_train(cfg)
    print(f"models saved: {out}")

def _cmd_evaluate(args):
    from fifa2026.pipeline import run_evaluate
    cfg = load_config(args.config)
    res = run_evaluate(cfg)
    print(f"metrics: {res.get('metrics')}")

def _cmd_predict(args):
    from fifa2026.pipeline import run_predict
    from fifa2026.squad_enrich import build_squad_agg
    from fifa2026.knockout.bracket import load_bracket
    cfg = load_config(args.config)
    teams = load_bracket(cfg.raw.get("bracket_path", "config/bracket_2026.yaml"))
    squad_agg = build_squad_agg(cfg, teams)  # None if no API key
    res = run_predict(cfg, squad_agg=squad_agg)
    top = sorted(res.champion_probs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Champion probabilities (top 8):")
    for team, p in top:
        print(f"  {team:25s} {p:6.1%}")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="fifa2026")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "evaluate", "predict"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    {"data": _cmd_data, "train": _cmd_train,
     "evaluate": _cmd_evaluate, "predict": _cmd_predict}[args.cmd](args)
```

(Keep the existing `build_win_prob` and imports; add `import argparse`/`pandas` already present.)

- [ ] **Step 4: Run it, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/cli.py tests/test_cli_commands.py
git commit -m "feat: wire CLI data/train/evaluate/predict command bodies"
```

---

### Task 12: Streamlit app + Makefile + README

**Files:**
- Create: `app.py`, `tests/test_app_smoke.py`
- Modify: `Makefile`, `README.md`

**Interfaces:**
- Produces: `build_dashboard(prediction: dict, evaluation: dict, market: dict | None) -> dict[str, go.Figure]` (pure, testable) in `app.py`; the Streamlit UI calls it. `app.py` loads `reports/prediction.json` + `reports/evaluation.json`.

- [ ] **Step 1: Write the failing smoke test** — `tests/test_app_smoke.py`:

```python
import importlib

def test_build_dashboard_returns_figures():
    app = importlib.import_module("app")
    prediction = {
        "champion_probs": {"France": 0.3, "Brazil": 0.25, "Spain": 0.2, "England": 0.25},
        "round_probs": {t: {"reach_R16": 1.0, "reach_QF": 0.5, "reach_SF": 0.4,
                            "reach_final": 0.3, "win": p}
                        for t, p in {"France":0.3,"Brazil":0.25,"Spain":0.2,"England":0.25}.items()},
        "tie_probs": [{"home": "France", "away": "Brazil", "p_home": 0.55}],
        "as_of": "2026-06-28", "meta": {},
    }
    figs = app.build_dashboard(prediction, {"metrics": {"log_loss": 1.0}, "calibration": []}, None)
    assert "champion" in figs and "bracket" in figs and "survival" in figs
```

> `app.py` must be importable without launching Streamlit — keep all `st.*` calls inside `if __name__ == "__main__":` or a `main()` guarded by a `_running_in_streamlit()` check. The `build_dashboard` function and figure construction must not call `st`.

- [ ] **Step 2: Run it, verify it fails.**

- [ ] **Step 3: Implement `app.py`** (repo root):

```python
from __future__ import annotations
import json
from pathlib import Path
from fifa2026.pipeline import PredictionResult
from fifa2026 import viz

REPORTS = Path("reports")

def _load(name, default):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

def build_dashboard(prediction: dict, evaluation: dict, market: dict | None) -> dict:
    result = PredictionResult.from_dict(prediction)
    figs = {
        "champion": viz.champion_bar(result, market=market),
        "bracket": viz.bracket_figure(result),
        "survival": viz.survival_heatmap(result),
        "calibration": viz.calibration_curve(evaluation.get("calibration", [])),
    }
    if market:
        figs["market"] = viz.market_compare(result.champion_probs, market)
    return figs

def main():
    import streamlit as st
    st.set_page_config(page_title="FIFA 2026 Champion Predictor", layout="wide")
    prediction = _load("prediction.json", None)
    if prediction is None:
        st.error("No prediction found. Run `make predict` first.")
        return
    evaluation = _load("evaluation.json", {"metrics": {}, "calibration": []})
    market = evaluation.get("market") or None
    st.title("🏆 FIFA World Cup 2026 — Champion Predictor")
    st.caption(f"Forecast as of {prediction.get('as_of', '')}")
    figs = build_dashboard(prediction, evaluation, market)
    tabs = st.tabs(["Champion odds", "Bracket", "Survival", "Team explorer", "Calibration"])
    with tabs[0]:
        st.plotly_chart(figs["champion"], use_container_width=True)
        if "market" in figs:
            st.plotly_chart(figs["market"], use_container_width=True)
    with tabs[1]:
        st.plotly_chart(figs["bracket"], use_container_width=True)
    with tabs[2]:
        st.plotly_chart(figs["survival"], use_container_width=True)
    with tabs[3]:
        result = PredictionResult.from_dict(prediction)
        team = st.selectbox("Team", sorted(result.champion_probs))
        st.write({"champion_prob": result.champion_probs[team], **result.round_probs[team]})
    with tabs[4]:
        st.plotly_chart(figs["calibration"], use_container_width=True)
        st.write(evaluation.get("metrics", {}))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `Makefile`** — add targets:

```makefile
.PHONY: data train evaluate predict app all test
data:      ; .venv/Scripts/python.exe -m fifa2026.cli data
train:     ; .venv/Scripts/python.exe -m fifa2026.cli train
evaluate:  ; .venv/Scripts/python.exe -m fifa2026.cli evaluate
predict:   ; .venv/Scripts/python.exe -m fifa2026.cli predict
app:       ; .venv/Scripts/streamlit run app.py
all:       ; $(MAKE) data && $(MAKE) train && $(MAKE) evaluate && $(MAKE) predict
test:      ; .venv/Scripts/python.exe -m pytest -q
```

- [ ] **Step 5: Update `README.md`** — replace the "What's left" framing with real usage: `pip install -e ".[dev]"`, copy `.env.example` → `.env` (optional key), then `make all && make app`. State the forecast is a dated snapshot of the live tournament.

- [ ] **Step 6: Run the smoke test + full suite, verify pass.**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add app.py Makefile README.md tests/test_app_smoke.py
git commit -m "feat: Streamlit demo app, make targets, and README usage"
```

---

### Task 13: Real end-to-end run + populate live odds (integration)

**Files:**
- Modify: `data/reference/market_odds_2026.csv` (real fetched odds), `config/default.yaml` (`bracket_path`, `results_csv` default), `reports/` (generated artifacts — gitignored, but verify)

**Interfaces:** none new — this task RUNS the real pipeline and verifies plausibility.

- [ ] **Step 1: Ensure config points at real paths** — in `config/default.yaml` confirm/add:
  `sources.results_csv: data/raw/results.csv`, and top-level `bracket_path: config/bracket_2026.yaml`.

- [ ] **Step 2: Download real data** — `.venv/Scripts/python.exe -m fifa2026.cli data`
  Expected: `data/raw/results.csv` written (tens of thousands of rows).

- [ ] **Step 3: Verify bracket team names resolve against the dataset.** Run a quick check that every team in `config/bracket_2026.yaml` appears in the downloaded results (else Elo/form default to baseline). Script:

```python
from fifa2026.ingest.matches import load_matches
from fifa2026.knockout.bracket import load_bracket
m = load_matches("data/raw/results.csv", train_start="2010-01-01")
known = set(m["home_team"]) | set(m["away_team"])
missing = [t for t in load_bracket("config/bracket_2026.yaml") if t not in known]
print("MISSING:", missing)
```
Expected: `MISSING: []`. If any team name differs from the dataset's spelling, fix the name in `config/bracket_2026.yaml` (and `confederations.csv`) to match the dataset, and re-run the ordering-validation test.

- [ ] **Step 4: Fill real winner odds** — fetch current bookmaker "2026 World Cup winner" decimal odds for the field and write them into `data/reference/market_odds_2026.csv` (`team,decimal_odds`). Wire `run_evaluate` to load this file (if present) → `implied_champion_probs` → store in `result["market"]`.

- [ ] **Step 5: Train, evaluate, predict on real data**

```bash
.venv/Scripts/python.exe -m fifa2026.cli train
.venv/Scripts/python.exe -m fifa2026.cli evaluate
.venv/Scripts/python.exe -m fifa2026.cli predict
```
Expected: `reports/prediction.json` + `reports/evaluation.json` written; printed top-8 champion table.

- [ ] **Step 6: Sanity-check plausibility.** Confirm: champion probs sum to ~1.0; the four pinned winners (Canada/Germany/Netherlands/France) have non-zero mass and their pinned opponents (South Africa/Paraguay/Morocco/Sweden) are 0.0; the favorites cluster among the strong sides (e.g. Argentina, France, Spain, Brazil, England, Portugal, Germany, Netherlands). If the table looks implausible (e.g. a minnow on top), investigate name-resolution and Elo before proceeding.

- [ ] **Step 7: Launch the app to confirm it renders** — `.venv/Scripts/streamlit run app.py` (headless check is fine: `streamlit run app.py --server.headless true` then stop). Confirm no import/render errors.

- [ ] **Step 8: Commit the real odds + any config/name fixes**

```bash
git add data/reference/market_odds_2026.csv config/default.yaml config/bracket_2026.yaml
git commit -m "chore: real 2026 winner odds and verified bracket names"
```

---

## Self-Review

**Spec coverage:**
- §4 data sources → T1 (reference), T2 (results download), T8/T13 (odds), T9 (API squads) ✔
- §5 real bracket + pinning + ordering validation → T5 (bracket+order test), T7 (decided win_prob) ✔
- §6 components (download, odds, persistence, pipeline, bracket round probs, cli, viz, app) → T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12 ✔
- §7 app five sections → T12 (champion, bracket, survival, team explorer, calibration) + market overlay ✔
- §8 reproducibility (make targets, .env) → T1 (.env), T12 (Makefile) ✔
- §9 testing (units + ordering + integration) → each task's tests + T13 integration ✔
- Graceful API fallback → T9 (returns None without key) + T11 (predict uses None) ✔
- Leakage guard respected → T6/T8 build training matrix with squad_agg=None ✔

**Placeholder scan:** no TBD/"handle errors"/"similar to". `market_odds_2026.csv` ships seeded and is filled with real values in T13 (explicitly flagged, not a silent gap). The calibration curve renders an empty-but-valid figure until calibration points are added (documented follow-up from the prior spec).

**Type consistency:** `PredictionResult` fields (`champion_probs`, `round_probs`, `tie_probs`, `as_of`, `meta`) consistent across T6/T7/T10/T12. `build_win_prob(..., decided=None)` signature consistent T7/T11. `round_probabilities` keys (`reach_R16/QF/SF/final/win`) consistent T5/T10/T12. `build_squad_agg(cfg, teams, api=None)` consistent T9/T11. Reports filenames `prediction.json`/`evaluation.json` consistent T7/T8/T12.
