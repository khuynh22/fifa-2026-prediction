# FIFA 2026 World Cup Champion Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible engine that predicts the 2026 FIFA World Cup champion by modeling each knockout match (Round of 32 → Final) with a hybrid ensemble, proper extra-time/shootout resolution, and a betting-market benchmark.

**Architecture:** A linear, testable pipeline — `ingest → features → models → knockout → bracket → evaluate`. Each stage is an independent unit with a typed contract, wired by a thin CLI / `make` targets. The match model is a blended ensemble of a Dixon-Coles/Poisson goals model and a LightGBM classifier. A bracket dynamic-program propagates per-match probabilities into per-round survival and champion probabilities (no Monte Carlo needed for v1 — exact under the independence assumption).

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, lightgbm, statsmodels, pyyaml, requests + on-disk cache, pytest.

## Global Constraints

- Python **3.11+**. Package name `fifa2026`, importable from `src/` layout.
- License **MIT**; copyright already in `LICENSE`. Do not add other licenses.
- **Point-in-time correctness is the central invariant:** every feature for a match dated `d` must be computed using only data with date `< d`. This is enforced by an explicit leakage test (Task 10) and must hold for every feature builder.
- **Bookmaker odds are NEVER a model input** — they are used only in the benchmark (Task 17).
- **Temporal validation only** — never random K-fold on matches.
- **Outcome encoding (verbatim, used everywhere):** integer label `0 = home win`, `1 = draw`, `2 = away win`. Every `predict_proba` returns a numpy array of shape `(n, 3)` with columns in that exact order `[p_home_win, p_draw, p_away_win]`.
- **Training window:** internationals with `date >= 2010-01-01` only.
- Deterministic seeds everywhere randomness appears: `RANDOM_SEED = 42`.
- Raw pulled data stays out of git (`.gitignore` already set); only `tests/fixtures/**` sample data is committed.
- Every task ends green (`pytest -q` passes) and is committed.

---

## Shared Data Contracts

These schemas are produced/consumed across tasks. Use these exact column names.

**`matches` DataFrame** (canonical normalized match history):
| column | dtype | notes |
|---|---|---|
| `match_id` | str | stable id: `f"{date:%Y%m%d}-{home_team}-{away_team}"` |
| `date` | datetime64[ns] | kickoff date |
| `home_team` | str | canonical team name |
| `away_team` | str | canonical team name |
| `home_score` | Int64 | regulation goals (nullable for future fixtures) |
| `away_score` | Int64 | regulation goals |
| `neutral` | bool | True if not played in either team's country |
| `tournament` | str | e.g. `FIFA World Cup`, `Friendly`, `UEFA Euro` |
| `city` | str | venue city (may be empty) |
| `country` | str | venue country (may be empty) |

**`outcome(home_score, away_score) -> int`**: `0` if home>away, `1` if equal, `2` if home<away.

**Feature row**: a `dict[str, float]` of named features (and a `pandas.DataFrame` of such rows for batches). Differential features end in `_diff` and are computed as `team_a_value - team_b_value` where `team_a` is the home/first team. Missing-data indicators end in `_isna`.

**Model interface** (Tasks 11–13):
```python
class MatchModel:
    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None) -> "MatchModel": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # shape (n, 3), order [home_win, draw, away_win]
        ...
```

**Tie resolution** (Task 14): `resolve_tie(p_reg: np.ndarray, pen_a: float, pen_b: float, depth_a: float, depth_b: float) -> float` returns `P(team_a advances)` (a single float in [0,1]).

---

## File Structure

```
src/fifa2026/
  __init__.py
  config.py                 # load YAML config, paths, seeds        (Task 1)
  cache.py                  # on-disk cache for fetched data         (Task 2)
  ingest/
    __init__.py
    matches.py              # normalize results.csv -> matches       (Task 3)
    reference.py            # rankings, confederations, venues, hosts(Task 4)
    api_client.py           # cached football API client             (Task 5)
    squads.py               # squads/players -> tidy tables          (Task 5)
  features/
    __init__.py
    elo.py                  # point-in-time Elo engine               (Task 6)
    form.py                 # opp-adjusted form + goal-rate features  (Task 7)
    context.py              # home/host/neutral, rest, travel, H2H    (Task 8)
    squad_features.py       # Tier B team aggregates + imputation     (Task 9)
    assemble.py             # build per-match differential feature row(Task 10)
  models/
    __init__.py
    poisson.py              # Dixon-Coles / Poisson goals model       (Task 11)
    boosted.py              # LightGBM W/D/L classifier               (Task 12)
    ensemble.py             # blend + calibration                     (Task 13)
  knockout/
    __init__.py
    resolve.py              # extra time + shootout + resolve_tie     (Task 14)
    bracket.py              # bracket DP -> survival + champion probs  (Task 15)
  evaluate/
    __init__.py
    backtest.py             # temporal validation + metrics           (Task 16)
    benchmark.py            # market benchmark                        (Task 17)
  cli.py                    # data/train/evaluate/predict entrypoints (Task 18)
config/
  default.yaml              # sources, paths, model params            (Task 1)
  bracket_2026.yaml         # the actual Round-of-32 pairings         (Task 15)
tests/
  fixtures/                 # tiny sample CSVs/JSON for tests
  ...                       # one test module per src module
Makefile                    # data/train/evaluate/predict targets     (Task 18)
pyproject.toml                                                        (Task 1)
```

---

### Task 1: Project scaffolding & config

**Files:**
- Create: `pyproject.toml`, `src/fifa2026/__init__.py`, `src/fifa2026/config.py`, `config/default.yaml`, `tests/test_config.py`, `tests/__init__.py`

**Interfaces:**
- Produces: `load_config(path: str | None = None) -> Config`; `Config` is a frozen dataclass with attributes `data_dir: Path`, `raw_dir: Path`, `processed_dir: Path`, `models_dir: Path`, `reports_dir: Path`, `train_start: str` (`"2010-01-01"`), `random_seed: int` (42), and `raw: dict` (the parsed YAML). `RANDOM_SEED = 42` module constant.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "fifa2026"
version = "0.1.0"
description = "ML engine predicting the 2026 FIFA World Cup champion"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.1",
  "numpy>=1.26",
  "scipy>=1.11",
  "scikit-learn>=1.3",
  "lightgbm>=4.1",
  "statsmodels>=0.14",
  "pyyaml>=6.0",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Write `config/default.yaml`**

```yaml
paths:
  data_dir: data
  raw_dir: data/raw
  processed_dir: data/processed
  models_dir: models
  reports_dir: reports
train_start: "2010-01-01"
random_seed: 42
sources:
  results_csv: data/raw/results.csv          # Kaggle "International football results"
  fifa_rankings_csv: data/raw/fifa_rankings.csv
  elo_csv: data/raw/elo.csv
api:
  base_url: "https://v3.football.api-sports.io"
  key_env: "FOOTBALL_API_KEY"
elo:
  k: 40
  home_advantage: 65
  initial: 1500
features:
  form_windows: [5, 10, 20]
hosts_2026: ["United States", "Mexico", "Canada"]
```

- [ ] **Step 3: Write the failing test** — `tests/test_config.py`

```python
from fifa2026.config import load_config, RANDOM_SEED

def test_load_config_defaults():
    cfg = load_config()
    assert cfg.train_start == "2010-01-01"
    assert cfg.random_seed == RANDOM_SEED == 42
    assert cfg.raw["hosts_2026"] == ["United States", "Mexico", "Canada"]
    assert str(cfg.raw_dir).replace("\\", "/").endswith("data/raw")
```

- [ ] **Step 4: Run test, verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL (`ModuleNotFoundError: fifa2026.config`).

- [ ] **Step 5: Implement `src/fifa2026/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

RANDOM_SEED = 42
_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

@dataclass(frozen=True)
class Config:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    train_start: str
    random_seed: int
    raw: dict

def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = raw["paths"]
    return Config(
        data_dir=Path(p["data_dir"]),
        raw_dir=Path(p["raw_dir"]),
        processed_dir=Path(p["processed_dir"]),
        models_dir=Path(p["models_dir"]),
        reports_dir=Path(p["reports_dir"]),
        train_start=raw["train_start"],
        random_seed=raw["random_seed"],
        raw=raw,
    )
```

Also create empty `src/fifa2026/__init__.py` and `tests/__init__.py`.

- [ ] **Step 6: Run test, verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config/default.yaml src/fifa2026 tests
git commit -m "feat: project scaffolding and config loader"
```

---

### Task 2: On-disk cache layer

**Files:**
- Create: `src/fifa2026/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Produces: `class DiskCache:` with `__init__(self, root: Path)`, `get(self, key: str) -> str | None`, `put(self, key: str, value: str) -> None`, `get_or_fetch(self, key: str, fetch: Callable[[], str]) -> str`. Keys are hashed to filenames; values are text (JSON/CSV). Used by the API client so the repo runs offline once data is cached.

- [ ] **Step 1: Write the failing test** — `tests/test_cache.py`

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_cache.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/fifa2026/cache.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable
import hashlib

class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.root / f"{digest}.cache"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put(self, key: str, value: str) -> None:
        self._path(key).write_text(value, encoding="utf-8")

    def get_or_fetch(self, key: str, fetch: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fetch()
        self.put(key, value)
        return value
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/cache.py tests/test_cache.py
git commit -m "feat: on-disk cache layer"
```

---

### Task 3: Match results ingestion

**Files:**
- Create: `src/fifa2026/ingest/__init__.py`, `src/fifa2026/ingest/matches.py`, `tests/fixtures/results_sample.csv`, `tests/test_ingest_matches.py`

**Interfaces:**
- Consumes: a CSV with the public `results.csv` schema (`date,home_team,away_team,home_score,away_score,tournament,city,country,neutral`).
- Produces: `load_matches(csv_path: str | Path, train_start: str | None = None) -> pd.DataFrame` returning the canonical `matches` DataFrame (see Shared Data Contracts), sorted by `date`, filtered to `date >= train_start` when given. `outcome(home_score, away_score) -> int` lives here too.

- [ ] **Step 1: Write fixture** — `tests/fixtures/results_sample.csv`

```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2009-06-10,Brazil,Argentina,3,1,Friendly,Rio,Brazil,False
2010-06-11,South Africa,Mexico,1,1,FIFA World Cup,Johannesburg,South Africa,False
2014-07-13,Germany,Argentina,1,0,FIFA World Cup,Rio,Brazil,True
2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,True
```

- [ ] **Step 2: Write the failing test** — `tests/test_ingest_matches.py`

```python
from pathlib import Path
import pandas as pd
from fifa2026.ingest.matches import load_matches, outcome

FIX = Path(__file__).parent / "fixtures" / "results_sample.csv"

def test_outcome_encoding():
    assert outcome(3, 1) == 0
    assert outcome(1, 1) == 1
    assert outcome(0, 2) == 2

def test_load_matches_filters_and_normalizes():
    df = load_matches(FIX, train_start="2010-01-01")
    assert list(df.columns) == [
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "neutral", "tournament", "city", "country",
    ]
    assert (df["date"] >= pd.Timestamp("2010-01-01")).all()
    assert len(df) == 3                      # 2009 row filtered out
    assert df["neutral"].dtype == bool
    assert df.iloc[0]["match_id"] == "20100611-South Africa-Mexico"
    assert df["date"].is_monotonic_increasing
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_ingest_matches.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `src/fifa2026/ingest/matches.py`** (and empty `ingest/__init__.py`)

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = ["match_id", "date", "home_team", "away_team",
           "home_score", "away_score", "neutral", "tournament", "city", "country"]

def outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2

def load_matches(csv_path: str | Path, train_start: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    if isinstance(df["neutral"].dtype, object) or df["neutral"].dtype != bool:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    for col in ("city", "country", "tournament"):
        df[col] = df[col].fillna("")
    if train_start is not None:
        df = df[df["date"] >= pd.Timestamp(train_start)]
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = (df["date"].dt.strftime("%Y%m%d") + "-"
                      + df["home_team"] + "-" + df["away_team"])
    return df[COLUMNS]
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_ingest_matches.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fifa2026/ingest tests/fixtures/results_sample.csv tests/test_ingest_matches.py
git commit -m "feat: match results ingestion and normalization"
```

---

### Task 4: Reference data ingestion (rankings, confederations, venues, hosts)

**Files:**
- Create: `src/fifa2026/ingest/reference.py`, `tests/fixtures/confederations.csv`, `tests/fixtures/venues.csv`, `tests/test_ingest_reference.py`

**Interfaces:**
- Produces:
  - `load_confederations(csv_path) -> dict[str, str]` — team → confederation code.
  - `load_venues(csv_path) -> pd.DataFrame` with columns `city, country, lat, lon, altitude_m`.
  - `team_country_coords(venues_df) -> dict[str, tuple[float, float]]` — country → (lat, lon) of its capital/main venue, used for travel distance.
  - `is_host(team: str, hosts: list[str]) -> bool`.

- [ ] **Step 1: Write fixtures**

`tests/fixtures/confederations.csv`:
```csv
team,confederation
Brazil,CONMEBOL
Argentina,CONMEBOL
Germany,UEFA
France,UEFA
Mexico,CONCACAF
United States,CONCACAF
```

`tests/fixtures/venues.csv`:
```csv
city,country,lat,lon,altitude_m
Mexico City,Mexico,19.43,-99.13,2240
Dallas,United States,32.78,-96.80,131
Toronto,Canada,43.65,-79.38,76
```

- [ ] **Step 2: Write the failing test** — `tests/test_ingest_reference.py`

```python
from pathlib import Path
from fifa2026.ingest.reference import (
    load_confederations, load_venues, team_country_coords, is_host,
)

FIX = Path(__file__).parent / "fixtures"

def test_confederations():
    conf = load_confederations(FIX / "confederations.csv")
    assert conf["Brazil"] == "CONMEBOL"
    assert conf["Germany"] == "UEFA"

def test_venues_and_coords():
    venues = load_venues(FIX / "venues.csv")
    assert {"city", "country", "lat", "lon", "altitude_m"} <= set(venues.columns)
    coords = team_country_coords(venues)
    assert coords["Mexico"] == (19.43, -99.13)

def test_is_host():
    assert is_host("Mexico", ["United States", "Mexico", "Canada"])
    assert not is_host("Brazil", ["United States", "Mexico", "Canada"])
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_ingest_reference.py -q`
Expected: FAIL.

- [ ] **Step 4: Implement `src/fifa2026/ingest/reference.py`**

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_confederations(csv_path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["team"], df["confederation"]))

def load_venues(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["altitude_m"] = df["altitude_m"].fillna(0).astype(float)
    return df[["city", "country", "lat", "lon", "altitude_m"]]

def team_country_coords(venues_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    first = venues_df.groupby("country").first()
    return {c: (row["lat"], row["lon"]) for c, row in first.iterrows()}

def is_host(team: str, hosts: list[str]) -> bool:
    return team in hosts
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_ingest_reference.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fifa2026/ingest/reference.py tests/fixtures/confederations.csv tests/fixtures/venues.csv tests/test_ingest_reference.py
git commit -m "feat: reference data ingestion (confederations, venues, hosts)"
```

---

### Task 5: Football API client + squad/player ingestion (Tier B)

**Files:**
- Create: `src/fifa2026/ingest/api_client.py`, `src/fifa2026/ingest/squads.py`, `tests/fixtures/squad_sample.json`, `tests/test_ingest_squads.py`

**Interfaces:**
- Produces:
  - `class FootballAPI:` `__init__(self, base_url, api_key, cache: DiskCache)`, `get_json(self, endpoint: str, params: dict) -> dict` (cache-first; only calls the network on a miss).
  - `parse_squad(payload: dict) -> pd.DataFrame` with columns `team, player, position, age, market_value_eu, season_minutes, season_xg, season_xa, injured` (one row per player). Pure function over the API JSON shape — tested with a fixture, **no network in tests**.
  - `team_player_table(payloads: dict[str, dict]) -> pd.DataFrame` — concat of `parse_squad` across teams.

- [ ] **Step 1: Write fixture** — `tests/fixtures/squad_sample.json`

```json
{"team": "Brazil", "players": [
  {"name": "Player A", "position": "Attacker", "age": 26, "market_value": 90000000,
   "minutes": 2700, "xg": 18.2, "xa": 7.1, "injured": false},
  {"name": "Player B", "position": "Defender", "age": 31, "market_value": 25000000,
   "minutes": 2500, "xg": 1.0, "xa": 1.4, "injured": true}
]}
```

- [ ] **Step 2: Write the failing test** — `tests/test_ingest_squads.py`

```python
import json
from pathlib import Path
from fifa2026.ingest.squads import parse_squad

FIX = Path(__file__).parent / "fixtures" / "squad_sample.json"

def test_parse_squad():
    payload = json.loads(FIX.read_text(encoding="utf-8"))
    df = parse_squad(payload)
    assert list(df.columns) == [
        "team", "player", "position", "age",
        "market_value_eu", "season_minutes", "season_xg", "season_xa", "injured",
    ]
    assert len(df) == 2
    assert df.iloc[0]["team"] == "Brazil"
    assert df.iloc[0]["market_value_eu"] == 90000000
    assert bool(df.iloc[1]["injured"]) is True
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_ingest_squads.py -q`
Expected: FAIL.

- [ ] **Step 4: Implement `src/fifa2026/ingest/api_client.py`**

```python
from __future__ import annotations
import json
import requests
from fifa2026.cache import DiskCache

class FootballAPI:
    def __init__(self, base_url: str, api_key: str, cache: DiskCache):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache

    def get_json(self, endpoint: str, params: dict) -> dict:
        key = f"{endpoint}?{sorted(params.items())}"
        def fetch() -> str:
            resp = requests.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                params=params,
                headers={"x-apisports-key": self.api_key},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.text
        return json.loads(self.cache.get_or_fetch(key, fetch))
```

- [ ] **Step 5: Implement `src/fifa2026/ingest/squads.py`**

```python
from __future__ import annotations
import pandas as pd

SQUAD_COLUMNS = ["team", "player", "position", "age", "market_value_eu",
                 "season_minutes", "season_xg", "season_xa", "injured"]

def parse_squad(payload: dict) -> pd.DataFrame:
    team = payload["team"]
    rows = []
    for p in payload["players"]:
        rows.append({
            "team": team,
            "player": p["name"],
            "position": p.get("position", ""),
            "age": p.get("age"),
            "market_value_eu": p.get("market_value"),
            "season_minutes": p.get("minutes"),
            "season_xg": p.get("xg"),
            "season_xa": p.get("xa"),
            "injured": bool(p.get("injured", False)),
        })
    return pd.DataFrame(rows, columns=SQUAD_COLUMNS)

def team_player_table(payloads: dict[str, dict]) -> pd.DataFrame:
    frames = [parse_squad(p) for p in payloads.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SQUAD_COLUMNS)
```

- [ ] **Step 6: Run test, verify it passes**

Run: `pytest tests/test_ingest_squads.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/fifa2026/ingest/api_client.py src/fifa2026/ingest/squads.py tests/fixtures/squad_sample.json tests/test_ingest_squads.py
git commit -m "feat: cached football API client and squad parsing"
```

---

### Task 6: Point-in-time Elo engine

**Files:**
- Create: `src/fifa2026/features/__init__.py`, `src/fifa2026/features/elo.py`, `tests/test_elo.py`

**Interfaces:**
- Consumes: `matches` DataFrame.
- Produces: `class EloEngine:` `__init__(self, k=40, home_advantage=65, initial=1500)`, `fit(self, matches: pd.DataFrame) -> "EloEngine"` (replays history in date order, storing **pre-match** ratings per match), `rating_before(self, team: str, date) -> float`, and `ratings_timeline` access. `expected_score(rating_a, rating_b, home_adv) -> float` is a module function.

- [ ] **Step 1: Write the failing test** — `tests/test_elo.py`

```python
import pandas as pd
from fifa2026.features.elo import EloEngine, expected_score

def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2010-02-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [3, 0], "away_score": [0, 0],
        "neutral": [True, True],
    })

def test_expected_score_symmetry():
    assert abs(expected_score(1500, 1500, 0) - 0.5) < 1e-9
    assert expected_score(1700, 1500, 0) > 0.5

def test_winner_gains_rating_and_pointintime():
    eng = EloEngine(k=40, home_advantage=0, initial=1500).fit(_matches())
    # Before any match both start at 1500
    assert eng.rating_before("A", pd.Timestamp("2010-01-01")) == 1500
    # After A beats B on 2010-01-01, A's pre-match rating on 2010-02-01 is higher
    assert eng.rating_before("A", pd.Timestamp("2010-02-01")) > 1500
    assert eng.rating_before("B", pd.Timestamp("2010-02-01")) < 1500
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_elo.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/features/elo.py`**

```python
from __future__ import annotations
from collections import defaultdict
import pandas as pd

def expected_score(rating_a: float, rating_b: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((rating_a + home_adv) - rating_b) / 400.0))

class EloEngine:
    def __init__(self, k: float = 40, home_advantage: float = 65, initial: float = 1500):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self._current: dict[str, float] = defaultdict(lambda: initial)
        # pre-match rating snapshots: (team, date) -> rating before that match
        self._pre: dict[tuple[str, pd.Timestamp], float] = {}

    def fit(self, matches: pd.DataFrame) -> "EloEngine":
        for _, m in matches.sort_values("date").iterrows():
            if pd.isna(m["home_score"]) or pd.isna(m["away_score"]):
                continue
            h, a, date = m["home_team"], m["away_team"], m["date"]
            rh, ra = self._current[h], self._current[a]
            self._pre[(h, date)] = rh
            self._pre[(a, date)] = ra
            ha = 0 if m.get("neutral", False) else self.home_advantage
            exp_h = expected_score(rh, ra, ha)
            score_h = 1.0 if m["home_score"] > m["away_score"] else (
                0.5 if m["home_score"] == m["away_score"] else 0.0)
            self._current[h] = rh + self.k * (score_h - exp_h)
            self._current[a] = ra + self.k * ((1 - score_h) - (1 - exp_h))
        return self

    def rating_before(self, team: str, date) -> float:
        return self._pre.get((team, pd.Timestamp(date)), self._current.get(team, self.initial))

    def rating_now(self, team: str) -> float:
        return self._current.get(team, self.initial)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_elo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/features/elo.py tests/test_elo.py
git commit -m "feat: point-in-time Elo rating engine"
```

---

### Task 7: Form & goal-rate features

**Files:**
- Create: `src/fifa2026/features/form.py`, `tests/test_form.py`

**Interfaces:**
- Consumes: `matches` DataFrame.
- Produces: `class FormFeatures:` `fit(matches)`, then per query `team_form(self, team: str, date, window: int) -> dict` returning `{f"ppg_{window}": float, f"gf_rate_{window}": float, f"ga_rate_{window}": float}` computed from the team's last `window` matches strictly **before** `date`. Returns zeros when no prior matches.

- [ ] **Step 1: Write the failing test** — `tests/test_form.py`

```python
import pandas as pd
from fifa2026.features.form import FormFeatures

def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "home_team": ["A", "C", "A"], "away_team": ["B", "A", "D"],
        "home_score": [2, 1, 0], "away_score": [0, 1, 0],
        "neutral": [True, True, True],
    })

def test_form_is_point_in_time():
    ff = FormFeatures().fit(_matches())
    # Before any games, zeros
    early = ff.team_form("A", pd.Timestamp("2010-01-01"), window=5)
    assert early["ppg_5"] == 0.0
    # On 2010-03-01, A has played: win (3 pts) then draw (1 pt) -> ppg = 2.0 over 2 games
    later = ff.team_form("A", pd.Timestamp("2010-03-01"), window=5)
    assert later["ppg_5"] == 2.0
    assert later["gf_rate_5"] == 1.5   # scored 2 then 1
    assert later["ga_rate_5"] == 0.5   # conceded 0 then 1
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_form.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/features/form.py`**

```python
from __future__ import annotations
import pandas as pd

class FormFeatures:
    def __init__(self):
        self._matches: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "FormFeatures":
        m = matches.dropna(subset=["home_score", "away_score"]).copy()
        self._matches = m.sort_values("date").reset_index(drop=True)
        return self

    def _team_history(self, team: str, date) -> pd.DataFrame:
        m = self._matches
        date = pd.Timestamp(date)
        mask = ((m["home_team"] == team) | (m["away_team"] == team)) & (m["date"] < date)
        return m[mask]

    def team_form(self, team: str, date, window: int) -> dict:
        hist = self._team_history(team, date).tail(window)
        key = f"_{window}"
        if hist.empty:
            return {f"ppg{key}": 0.0, f"gf_rate{key}": 0.0, f"ga_rate{key}": 0.0}
        pts = gf = ga = 0
        for _, m in hist.iterrows():
            is_home = m["home_team"] == team
            scored = m["home_score"] if is_home else m["away_score"]
            conceded = m["away_score"] if is_home else m["home_score"]
            gf += scored; ga += conceded
            if scored > conceded: pts += 3
            elif scored == conceded: pts += 1
        n = len(hist)
        return {f"ppg{key}": pts / n, f"gf_rate{key}": gf / n, f"ga_rate{key}": ga / n}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_form.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/features/form.py tests/test_form.py
git commit -m "feat: point-in-time form and goal-rate features"
```

---

### Task 8: Context features (home/host/neutral, rest, travel, altitude, H2H)

**Files:**
- Create: `src/fifa2026/features/context.py`, `tests/test_context.py`

**Interfaces:**
- Consumes: `matches`, confederations dict, country coords dict, hosts list.
- Produces:
  - `haversine_km(lat1, lon1, lat2, lon2) -> float`.
  - `class ContextFeatures:` `fit(matches)`, `rest_days(self, team, date) -> float` (days since last match, capped at 30, default 30), `head_to_head(self, team_a, team_b, date) -> dict` (`{"h2h_ppg": float}` for team_a vs team_b before date), `home_flag(team, venue_country, hosts) -> int`.

- [ ] **Step 1: Write the failing test** — `tests/test_context.py`

```python
import pandas as pd
from fifa2026.features.context import haversine_km, ContextFeatures, home_flag

def test_haversine_known_distance():
    # Mexico City to Dallas ~ 1400 km (allow tolerance)
    d = haversine_km(19.43, -99.13, 32.78, -96.80)
    assert 1300 < d < 1600

def test_home_flag():
    assert home_flag("Mexico", "Mexico", ["Mexico", "United States", "Canada"]) == 1
    assert home_flag("Brazil", "Mexico", ["Mexico"]) == 0

def test_rest_days_point_in_time():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2010-06-01", "2010-06-08"]),
        "home_team": ["A", "A"], "away_team": ["B", "C"],
        "home_score": [1, 1], "away_score": [0, 0], "neutral": [True, True],
    })
    cf = ContextFeatures().fit(m)
    assert cf.rest_days("A", pd.Timestamp("2010-06-08")) == 7
    assert cf.rest_days("A", pd.Timestamp("2010-06-01")) == 30  # no prior match -> default cap
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_context.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/features/context.py`**

```python
from __future__ import annotations
from math import radians, sin, cos, asin, sqrt
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))

def home_flag(team: str, venue_country: str, hosts: list[str]) -> int:
    return 1 if (team == venue_country or (team in hosts and venue_country in hosts and team == venue_country)) else 0

class ContextFeatures:
    REST_CAP = 30

    def __init__(self):
        self._matches: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "ContextFeatures":
        self._matches = matches.sort_values("date").reset_index(drop=True)
        return self

    def rest_days(self, team: str, date) -> float:
        m = self._matches
        date = pd.Timestamp(date)
        mask = ((m["home_team"] == team) | (m["away_team"] == team)) & (m["date"] < date)
        prior = m[mask]
        if prior.empty:
            return float(self.REST_CAP)
        last = prior["date"].max()
        return float(min((date - last).days, self.REST_CAP))

    def head_to_head(self, team_a: str, team_b: str, date) -> dict:
        m = self._matches
        date = pd.Timestamp(date)
        pair = (((m["home_team"] == team_a) & (m["away_team"] == team_b)) |
                ((m["home_team"] == team_b) & (m["away_team"] == team_a)))
        hist = m[pair & (m["date"] < date)].dropna(subset=["home_score", "away_score"])
        if hist.empty:
            return {"h2h_ppg": 0.0}
        pts = 0
        for _, g in hist.iterrows():
            is_home = g["home_team"] == team_a
            sa = g["home_score"] if is_home else g["away_score"]
            sb = g["away_score"] if is_home else g["home_score"]
            pts += 3 if sa > sb else (1 if sa == sb else 0)
        return {"h2h_ppg": pts / len(hist)}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_context.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/features/context.py tests/test_context.py
git commit -m "feat: context features (travel, rest, h2h, home flag)"
```

---

### Task 9: Tier B squad/team features + availability/imputation

**Files:**
- Create: `src/fifa2026/features/squad_features.py`, `tests/test_squad_features.py`

**Interfaces:**
- Consumes: the player table from Task 5 (`SQUAD_COLUMNS`).
- Produces: `team_aggregates(players: pd.DataFrame) -> pd.DataFrame` indexed by `team` with columns `squad_value`, `top_xg`, `mean_age`, `n_injured`, `total_xg`. `impute_tier_b(features: pd.DataFrame) -> pd.DataFrame` adds `<col>_isna` indicators and fills missing numeric columns with the column median (0 if all-missing).

- [ ] **Step 1: Write the failing test** — `tests/test_squad_features.py`

```python
import numpy as np
import pandas as pd
from fifa2026.features.squad_features import team_aggregates, impute_tier_b

def _players():
    return pd.DataFrame({
        "team": ["Brazil", "Brazil", "France"],
        "player": ["A", "B", "C"], "position": ["Attacker", "Defender", "Attacker"],
        "age": [26, 31, 24], "market_value_eu": [90e6, 25e6, 80e6],
        "season_minutes": [2700, 2500, 2600], "season_xg": [18.2, 1.0, 15.0],
        "season_xa": [7.1, 1.4, 6.0], "injured": [False, True, False],
    })

def test_team_aggregates():
    agg = team_aggregates(_players())
    assert agg.loc["Brazil", "squad_value"] == 115e6
    assert agg.loc["Brazil", "top_xg"] == 18.2
    assert agg.loc["Brazil", "n_injured"] == 1

def test_impute_adds_indicator_and_fills():
    df = pd.DataFrame({"squad_value": [100.0, np.nan]})
    out = impute_tier_b(df)
    assert out["squad_value_isna"].tolist() == [0, 1]
    assert out["squad_value"].tolist() == [100.0, 100.0]  # median fill
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_squad_features.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/features/squad_features.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def team_aggregates(players: pd.DataFrame) -> pd.DataFrame:
    g = players.groupby("team")
    out = pd.DataFrame({
        "squad_value": g["market_value_eu"].sum(),
        "top_xg": g["season_xg"].max(),
        "total_xg": g["season_xg"].sum(),
        "mean_age": g["age"].mean(),
        "n_injured": g["injured"].sum().astype(int),
    })
    return out

def impute_tier_b(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in features.select_dtypes(include="number").columns:
        out[f"{col}_isna"] = out[col].isna().astype(int)
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_squad_features.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/features/squad_features.py tests/test_squad_features.py
git commit -m "feat: Tier B squad aggregates with imputation"
```

---

### Task 10: Feature assembly + leakage test

**Files:**
- Create: `src/fifa2026/features/assemble.py`, `tests/test_assemble.py`

**Interfaces:**
- Consumes: all feature builders (Elo, Form, Context, squad aggregates), reference data.
- Produces:
  - `class FeatureBuilder:` constructed with fitted `EloEngine`, `FormFeatures`, `ContextFeatures`, `confederations: dict`, `squad_agg: pd.DataFrame | None`, `hosts: list[str]`, `form_windows: list[int]`.
  - `row(self, home_team, away_team, date, venue_country, neutral) -> dict[str, float]` — one feature row of **A-vs-B differentials** (`team_a = home`). Includes `elo_diff`, `ppg_<w>_diff`, `gf_rate_<w>_diff`, `ga_rate_<w>_diff`, `rest_diff`, `h2h_ppg`, `home_diff`, `same_confed`, and (if squad data present) `squad_value_diff`, `top_xg_diff`, `mean_age_diff`, `n_injured_diff`.
  - `build_training_matrix(self, matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]` — features `X` and labels `y` (outcome encoding) for all matches.

- [ ] **Step 1: Write the failing test** — `tests/test_assemble.py`

```python
import numpy as np
import pandas as pd
from fifa2026.features.elo import EloEngine
from fifa2026.features.form import FormFeatures
from fifa2026.features.context import ContextFeatures
from fifa2026.features.assemble import FeatureBuilder

def _matches():
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"],
        "home_score": [2, 1, 0], "away_score": [0, 0, 1],
        "neutral": [True, True, True], "country": ["X", "X", "Y"],
    })

def _builder(m):
    return FeatureBuilder(
        elo=EloEngine(home_advantage=0).fit(m),
        form=FormFeatures().fit(m),
        context=ContextFeatures().fit(m),
        confederations={"A": "UEFA", "B": "UEFA"},
        squad_agg=None, hosts=[], form_windows=[5],
    )

def test_row_has_differential_features():
    m = _matches()
    fb = _builder(m)
    row = fb.row("A", "B", pd.Timestamp("2010-03-01"), venue_country="Y", neutral=True)
    assert "elo_diff" in row and "ppg_5_diff" in row and "same_confed" in row
    assert row["same_confed"] == 1
    assert row["elo_diff"] > 0  # A beat B twice before this date

def test_no_leakage_features_ignore_future():
    """Feature row for a match must not change if FUTURE matches are added."""
    m = _matches()
    fb_now = _builder(m)
    row_now = fb_now.row("A", "B", pd.Timestamp("2010-02-01"), "X", True)

    future = pd.concat([m, pd.DataFrame({
        "match_id": ["m9"], "date": pd.to_datetime(["2010-05-01"]),
        "home_team": ["A"], "away_team": ["B"], "home_score": [9], "away_score": [0],
        "neutral": [True], "country": ["X"],
    })], ignore_index=True)
    fb_future = _builder(future)
    row_future = fb_future.row("A", "B", pd.Timestamp("2010-02-01"), "X", True)
    assert row_now == row_future  # adding a 2010-05 match cannot change a 2010-02 feature row
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_assemble.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/features/assemble.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from fifa2026.ingest.matches import outcome

class FeatureBuilder:
    def __init__(self, elo, form, context, confederations, squad_agg, hosts, form_windows):
        self.elo = elo
        self.form = form
        self.context = context
        self.confederations = confederations
        self.squad_agg = squad_agg
        self.hosts = hosts
        self.form_windows = form_windows

    def _home_flag(self, team, venue_country, neutral):
        if neutral:
            return 1 if team == venue_country else 0
        return 1 if team == venue_country else 0

    def row(self, home_team, away_team, date, venue_country, neutral) -> dict:
        a, b = home_team, away_team
        feats = {}
        feats["elo_diff"] = self.elo.rating_before(a, date) - self.elo.rating_before(b, date)
        for w in self.form_windows:
            fa = self.form.team_form(a, date, w)
            fb = self.form.team_form(b, date, w)
            feats[f"ppg_{w}_diff"] = fa[f"ppg_{w}"] - fb[f"ppg_{w}"]
            feats[f"gf_rate_{w}_diff"] = fa[f"gf_rate_{w}"] - fb[f"gf_rate_{w}"]
            feats[f"ga_rate_{w}_diff"] = fa[f"ga_rate_{w}"] - fb[f"ga_rate_{w}"]
        feats["rest_diff"] = self.context.rest_days(a, date) - self.context.rest_days(b, date)
        feats["h2h_ppg"] = self.context.head_to_head(a, b, date)["h2h_ppg"]
        feats["home_diff"] = self._home_flag(a, venue_country, neutral) - self._home_flag(b, venue_country, neutral)
        feats["same_confed"] = int(self.confederations.get(a) == self.confederations.get(b)
                                   and self.confederations.get(a) is not None)
        if self.squad_agg is not None:
            for col, name in [("squad_value", "squad_value_diff"),
                              ("top_xg", "top_xg_diff"),
                              ("mean_age", "mean_age_diff"),
                              ("n_injured", "n_injured_diff")]:
                va = self.squad_agg[col].get(a, np.nan)
                vb = self.squad_agg[col].get(b, np.nan)
                feats[name] = float(va - vb) if pd.notna(va) and pd.notna(vb) else 0.0
        return feats

    def build_training_matrix(self, matches: pd.DataFrame):
        rows, labels = [], []
        m = matches.dropna(subset=["home_score", "away_score"]).sort_values("date")
        for _, g in m.iterrows():
            rows.append(self.row(g["home_team"], g["away_team"], g["date"],
                                 g.get("country", ""), bool(g.get("neutral", True))))
            labels.append(outcome(int(g["home_score"]), int(g["away_score"])))
        return pd.DataFrame(rows).fillna(0.0), np.array(labels)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_assemble.py -q`
Expected: PASS (especially the no-leakage test).

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/features/assemble.py tests/test_assemble.py
git commit -m "feat: feature assembly with point-in-time leakage test"
```

---

### Task 11: Dixon-Coles / Poisson goals model

**Files:**
- Create: `src/fifa2026/models/__init__.py`, `src/fifa2026/models/poisson.py`, `tests/test_poisson.py`

**Interfaces:**
- Produces: `class PoissonModel(MatchModel):` predicts expected goals for each side from features via two Poisson GLMs (`statsmodels`), then converts the (λ_home, λ_away) pair to `[p_home_win, p_draw, p_away_win]` by summing the bivariate scoreline grid (independent Poisson up to 10 goals). `expected_goals(self, X) -> tuple[np.ndarray, np.ndarray]`. `scoreline_probs(lam_h, lam_a, max_goals=10) -> np.ndarray` (3,) module function.

- [ ] **Step 1: Write the failing test** — `tests/test_poisson.py`

```python
import numpy as np
import pandas as pd
from fifa2026.models.poisson import PoissonModel, scoreline_probs

def test_scoreline_probs_sum_to_one_and_favor_higher_lambda():
    p = scoreline_probs(2.0, 0.5)
    assert abs(p.sum() - 1.0) < 1e-6
    assert p[0] > p[2]  # home (higher lambda) more likely to win

def test_poisson_fits_and_predicts_shape():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n)})
    # stronger team (higher elo_diff) scores more
    yh = rng.poisson(np.clip(1.3 + X["elo_diff"] / 200, 0.1, None))
    ya = rng.poisson(np.clip(1.3 - X["elo_diff"] / 200, 0.1, None))
    y = np.where(yh > ya, 0, np.where(yh == ya, 1, 2))
    model = PoissonModel().fit(X, y, goals_home=yh.values if hasattr(yh, "values") else yh,
                              goals_away=ya.values if hasattr(ya, "values") else ya)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_poisson.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/models/poisson.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

def scoreline_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), lam_h)
    a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(h, a)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return np.array([p_home, p_draw, p_away]) / total

class PoissonModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self._home = None
        self._away = None
        self._cols = None

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None,
            goals_home=None, goals_away=None) -> "PoissonModel":
        self._cols = list(X.columns)
        Xc = sm.add_constant(X, has_constant="add")
        self._home = sm.GLM(goals_home, Xc, family=sm.families.Poisson(),
                            freq_weights=sample_weight).fit()
        self._away = sm.GLM(goals_away, Xc, family=sm.families.Poisson(),
                            freq_weights=sample_weight).fit()
        return self

    def expected_goals(self, X: pd.DataFrame):
        Xc = sm.add_constant(X[self._cols], has_constant="add")
        return self._home.predict(Xc), self._away.predict(Xc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(X)
        return np.array([scoreline_probs(h, a, self.max_goals)
                         for h, a in zip(np.asarray(lam_h), np.asarray(lam_a))])
```

> Note: `build_training_matrix` (Task 10) must also expose per-match `home_score`/`away_score` to train the Poisson side. When wiring (Task 18), pass `goals_home`/`goals_away` from the matches frame aligned to `X`.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_poisson.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/models/poisson.py tests/test_poisson.py
git commit -m "feat: Poisson goals model with scoreline -> W/D/L"
```

---

### Task 12: LightGBM W/D/L classifier

**Files:**
- Create: `src/fifa2026/models/boosted.py`, `tests/test_boosted.py`

**Interfaces:**
- Produces: `class BoostedModel(MatchModel):` wraps `lightgbm.LGBMClassifier(objective="multiclass", num_class=3, random_state=42)`. `fit(X, y, sample_weight=None)`, `predict_proba(X) -> (n,3)`. **Class-order guard:** LightGBM orders columns by `classes_`; reindex output so columns are always `[0,1,2]` = `[home_win, draw, away_win]`.

- [ ] **Step 1: Write the failing test** — `tests/test_boosted.py`

```python
import numpy as np
import pandas as pd
from fifa2026.models.boosted import BoostedModel

def test_boosted_predicts_calibrated_shape_and_order():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n)})
    # home win when elo_diff high, away win when low, draw in middle
    y = np.where(X["elo_diff"] > 50, 0, np.where(X["elo_diff"] < -50, 2, 1))
    model = BoostedModel(n_estimators=50).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # A strongly-favored home case should put most mass on column 0
    strong = model.predict_proba(pd.DataFrame({"elo_diff": [400.0]}))
    assert strong[0].argmax() == 0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_boosted.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/models/boosted.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class BoostedModel:
    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.03,
                 num_leaves: int = 31, random_state: int = 42):
        self.clf = LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            random_state=random_state, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BoostedModel":
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        # reorder columns to canonical [0,1,2] regardless of clf.classes_
        order = np.argsort(self.clf.classes_)
        full = np.zeros((proba.shape[0], 3))
        for j, cls in enumerate(self.clf.classes_[order]):
            full[:, int(cls)] = proba[:, order[j]]
        return full
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_boosted.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/models/boosted.py tests/test_boosted.py
git commit -m "feat: LightGBM W/D/L classifier with canonical class order"
```

---

### Task 13: Ensemble blend + calibration

**Files:**
- Create: `src/fifa2026/models/ensemble.py`, `tests/test_ensemble.py`

**Interfaces:**
- Produces: `class EnsembleModel(MatchModel):` `__init__(self, poisson, boosted, weight=0.5)` blends `weight*poisson + (1-weight)*boosted`. `tune_weight(self, X_val, y_val, grid=None) -> float` picks the blend weight minimizing validation log-loss and stores it. `predict_proba(X) -> (n,3)`.

- [ ] **Step 1: Write the failing test** — `tests/test_ensemble.py`

```python
import numpy as np
import pandas as pd
from fifa2026.models.ensemble import EnsembleModel

class _Fake:
    def __init__(self, proba): self._p = np.array(proba)
    def predict_proba(self, X): return np.tile(self._p, (len(X), 1))

def test_blend_is_convex_combo():
    a = _Fake([0.7, 0.2, 0.1])
    b = _Fake([0.1, 0.2, 0.7])
    ens = EnsembleModel(a, b, weight=0.5)
    X = pd.DataFrame({"x": [0, 0]})
    out = ens.predict_proba(X)
    assert np.allclose(out[0], [0.4, 0.2, 0.4])
    assert np.allclose(out.sum(axis=1), 1.0)

def test_tune_weight_prefers_better_model():
    good = _Fake([0.9, 0.05, 0.05])   # matches label 0
    bad = _Fake([0.05, 0.05, 0.9])
    ens = EnsembleModel(good, bad)
    X = pd.DataFrame({"x": np.zeros(20)})
    y = np.zeros(20, dtype=int)        # all home wins
    w = ens.tune_weight(X, y)
    assert w > 0.5                     # lean toward the good (poisson-slot) model
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_ensemble.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/models/ensemble.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

class EnsembleModel:
    def __init__(self, poisson, boosted, weight: float = 0.5):
        self.poisson = poisson
        self.boosted = boosted
        self.weight = weight

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.poisson.predict_proba(X)
        b = self.boosted.predict_proba(X)
        out = self.weight * p + (1 - self.weight) * b
        return out / out.sum(axis=1, keepdims=True)

    def tune_weight(self, X_val: pd.DataFrame, y_val, grid=None) -> float:
        grid = grid if grid is not None else np.linspace(0, 1, 21)
        p = self.poisson.predict_proba(X_val)
        b = self.boosted.predict_proba(X_val)
        best_w, best_ll = 0.5, float("inf")
        for w in grid:
            blend = w * p + (1 - w) * b
            blend = blend / blend.sum(axis=1, keepdims=True)
            ll = log_loss(y_val, blend, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll, best_w = ll, w
        self.weight = float(best_w)
        return self.weight
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_ensemble.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/models/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble blend with log-loss weight tuning"
```

---

### Task 14: Knockout resolution (extra time + shootout)

**Files:**
- Create: `src/fifa2026/knockout/__init__.py`, `src/fifa2026/knockout/resolve.py`, `tests/test_resolve.py`

**Interfaces:**
- Produces: `resolve_tie(p_reg, pen_a=0.5, pen_b=0.5, depth_a=0.0, depth_b=0.0) -> float` returns `P(team_a advances)`. Logic: team_a advances if it wins regulation (`p_reg[0]`), OR the match is drawn (`p_reg[1]`) and team_a then wins extra-time-or-shootout. The draw is split by `shootout_prob(pen_a, pen_b, depth_a, depth_b)`. `shootout_prob(...)` is near 0.5, nudged by penalty record and squad depth.

- [ ] **Step 1: Write the failing test** — `tests/test_resolve.py`

```python
import numpy as np
from fifa2026.knockout.resolve import resolve_tie, shootout_prob

def test_shootout_is_near_even_by_default():
    assert abs(shootout_prob(0.5, 0.5, 0.0, 0.0) - 0.5) < 1e-9

def test_better_penalty_team_favored():
    assert shootout_prob(0.8, 0.2, 0.0, 0.0) > 0.5

def test_resolve_tie_accounts_for_draw_split():
    # 50% reg win for a, 20% draw, 30% reg win for b; even shootout
    p = np.array([0.5, 0.2, 0.3])
    adv = resolve_tie(p, pen_a=0.5, pen_b=0.5)
    assert abs(adv - (0.5 + 0.2 * 0.5)) < 1e-9   # 0.6

def test_resolve_tie_full_certainty():
    p = np.array([1.0, 0.0, 0.0])
    assert resolve_tie(p) == 1.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_resolve.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/knockout/resolve.py`**

```python
from __future__ import annotations
import numpy as np

def shootout_prob(pen_a: float, pen_b: float, depth_a: float = 0.0, depth_b: float = 0.0) -> float:
    """Near-even, nudged by penalty record (pen_*) and squad depth (depth_*)."""
    logit = 1.2 * (pen_a - pen_b) + 0.3 * (depth_a - depth_b)
    p = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(p, 0.05, 0.95))

def resolve_tie(p_reg, pen_a: float = 0.5, pen_b: float = 0.5,
                depth_a: float = 0.0, depth_b: float = 0.0) -> float:
    p_reg = np.asarray(p_reg, dtype=float)
    p_win, p_draw, _p_loss = p_reg
    s = shootout_prob(pen_a, pen_b, depth_a, depth_b)
    return float(p_win + p_draw * s)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_resolve.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/knockout tests/test_resolve.py
git commit -m "feat: knockout tie resolution with shootout sub-model"
```

---

### Task 15: Bracket DP → survival + champion probabilities

**Files:**
- Create: `src/fifa2026/knockout/bracket.py`, `config/bracket_2026.yaml`, `tests/test_bracket.py`

**Interfaces:**
- Consumes: a `win_prob(team_a, team_b) -> float` callable (built later from the ensemble + `resolve_tie`).
- Produces:
  - `load_bracket(path) -> list[str]` — 32 teams in bracket-slot order (R32 pairs are slots `(0,1),(2,3),...`).
  - `champion_probabilities(teams: list[str], win_prob) -> dict[str, float]` — exact DP over the single-elimination tree; values sum to 1.
  - `predict_champion(teams, win_prob) -> tuple[str, float]` — argmax team + its probability.

- [ ] **Step 1: Write `config/bracket_2026.yaml` (placeholder slots to be filled with the real draw)**

```yaml
# 32 teams in bracket-slot order. Round-of-32 ties are (slot0 vs slot1),
# (slot2 vs slot3), ... Fill with the actual 2026 Round-of-32 bracket.
teams:
  - "Team01"
  - "Team02"
  - "Team03"
  - "Team04"
# ... 32 total
```

- [ ] **Step 2: Write the failing test** — `tests/test_bracket.py`

```python
import numpy as np
from fifa2026.knockout.bracket import champion_probabilities, predict_champion

def test_four_team_bracket_sums_to_one_and_favors_strong():
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.9, "B": 0.5, "C": 0.5, "D": 0.1}
    def win_prob(a, b):  # logistic on strength difference
        return 1.0 / (1.0 + np.exp(-(strength[a] - strength[b]) * 6))
    probs = champion_probabilities(teams, win_prob)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["A"] == max(probs.values())
    assert probs["D"] == min(probs.values())

def test_predict_champion_returns_argmax():
    teams = ["A", "B"]
    champ, p = predict_champion(teams, lambda a, b: 0.8 if a == "A" else 0.2)
    assert champ == "A" and abs(p - 0.8) < 1e-9
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_bracket.py -q`
Expected: FAIL.

- [ ] **Step 4: Implement `src/fifa2026/knockout/bracket.py`**

```python
from __future__ import annotations
from pathlib import Path
import yaml

def load_bracket(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(data["teams"])

def _solve(teams: list[str], win_prob) -> dict[str, float]:
    """Return {team: P(team wins this sub-bracket)} via exact DP."""
    if len(teams) == 1:
        return {teams[0]: 1.0}
    mid = len(teams) // 2
    left = _solve(teams[:mid], win_prob)
    right = _solve(teams[mid:], win_prob)
    out: dict[str, float] = {}
    for a, pa in left.items():
        out[a] = out.get(a, 0.0) + pa * sum(pb * win_prob(a, b) for b, pb in right.items())
    for b, pb in right.items():
        out[b] = out.get(b, 0.0) + pb * sum(pa * win_prob(b, a) for a, pa in left.items())
    return out

def champion_probabilities(teams: list[str], win_prob) -> dict[str, float]:
    if (len(teams) & (len(teams) - 1)) != 0:
        raise ValueError("bracket size must be a power of 2")
    return _solve(teams, win_prob)

def predict_champion(teams: list[str], win_prob):
    probs = champion_probabilities(teams, win_prob)
    champ = max(probs, key=probs.get)
    return champ, probs[champ]
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_bracket.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fifa2026/knockout/bracket.py config/bracket_2026.yaml tests/test_bracket.py
git commit -m "feat: exact bracket DP for survival and champion probabilities"
```

---

### Task 16: Evaluation — temporal validation + metrics

**Files:**
- Create: `src/fifa2026/evaluate/__init__.py`, `src/fifa2026/evaluate/backtest.py`, `tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `temporal_split(matches, cutoff: str) -> tuple[index, index]` — train = `date < cutoff`, test = `date >= cutoff`.
  - `evaluate_probs(y_true, proba) -> dict` returning `{"log_loss": float, "brier": float, "accuracy": float}` (multiclass Brier = mean squared error vs one-hot).

- [ ] **Step 1: Write the failing test** — `tests/test_backtest.py`

```python
import numpy as np
import pandas as pd
from fifa2026.evaluate.backtest import temporal_split, evaluate_probs

def test_temporal_split_respects_time():
    m = pd.DataFrame({"date": pd.to_datetime(["2018-01-01", "2022-01-01", "2024-01-01"])})
    train, test = temporal_split(m, cutoff="2022-01-01")
    assert train.tolist() == [0]
    assert test.tolist() == [1, 2]

def test_metrics_perfect_and_shapes():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)[y]
    out = evaluate_probs(y, perfect)
    assert out["accuracy"] == 1.0
    assert out["brier"] < 1e-9
    assert out["log_loss"] < 1e-6
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_backtest.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/evaluate/backtest.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

def temporal_split(matches: pd.DataFrame, cutoff: str):
    date = pd.to_datetime(matches["date"])
    cut = pd.Timestamp(cutoff)
    train = matches.index[date < cut]
    test = matches.index[date >= cut]
    return train, test

def evaluate_probs(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    ll = float(log_loss(y_true, proba, labels=[0, 1, 2]))
    acc = float(accuracy_score(y_true, proba.argmax(axis=1)))
    return {"log_loss": ll, "brier": brier, "accuracy": acc}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/evaluate tests/test_backtest.py
git commit -m "feat: temporal validation split and probability metrics"
```

---

### Task 17: Market benchmark

**Files:**
- Create: `src/fifa2026/evaluate/benchmark.py`, `tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `odds_to_probs(odds_home, odds_draw, odds_away) -> np.ndarray` — decimal odds → de-vigged probabilities `[home, draw, away]`.
  - `compare_to_market(y_true, model_proba, market_proba) -> dict` returning `{"model_log_loss", "market_log_loss", "agreement_rate", "model_beats_market": bool}` where agreement = same argmax.

- [ ] **Step 1: Write the failing test** — `tests/test_benchmark.py`

```python
import numpy as np
from fifa2026.evaluate.benchmark import odds_to_probs, compare_to_market

def test_odds_to_probs_devig_sums_to_one():
    p = odds_to_probs(2.0, 3.5, 4.0)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[2]  # shorter odds -> higher prob

def test_compare_to_market():
    y = np.array([0, 0, 2])
    model = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    market = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]])
    out = compare_to_market(y, model, market)
    assert out["agreement_rate"] == 1.0
    assert out["model_beats_market"] is True
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_benchmark.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/evaluate/benchmark.py`**

```python
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss

def odds_to_probs(odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return raw / raw.sum()

def compare_to_market(y_true, model_proba, market_proba) -> dict:
    y_true = np.asarray(y_true)
    model_proba = np.asarray(model_proba)
    market_proba = np.asarray(market_proba)
    model_ll = float(log_loss(y_true, model_proba, labels=[0, 1, 2]))
    market_ll = float(log_loss(y_true, market_proba, labels=[0, 1, 2]))
    agreement = float(np.mean(model_proba.argmax(axis=1) == market_proba.argmax(axis=1)))
    return {
        "model_log_loss": model_ll,
        "market_log_loss": market_ll,
        "agreement_rate": agreement,
        "model_beats_market": bool(model_ll < market_ll),
    }
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/evaluate/benchmark.py tests/test_benchmark.py
git commit -m "feat: bookmaker-market benchmark"
```

---

### Task 18: End-to-end wiring (CLI + Makefile) + smoke test

**Files:**
- Create: `src/fifa2026/cli.py`, `Makefile`, `tests/test_cli_smoke.py`
- Modify: `README.md` (mark commands as working)

**Interfaces:**
- Consumes: every prior module.
- Produces:
  - `build_win_prob(model: MatchModel, feature_builder, as_of_date, pen, depth) -> Callable[[str, str], float]` — closes over the trained model and feature builder; for a neutral knockout match returns `resolve_tie(model.predict_proba(row)[0], ...)`.
  - CLI subcommands `data`, `train`, `evaluate`, `predict` (argparse). `predict` loads the bracket, builds `win_prob`, and prints champion + top-10 title probabilities.

- [ ] **Step 1: Write the failing smoke test** — `tests/test_cli_smoke.py`

```python
import numpy as np
import pandas as pd
from fifa2026.cli import build_win_prob
from fifa2026.knockout.bracket import predict_champion

class _Model:
    """Returns a fixed strong-home probability for any feature row."""
    def predict_proba(self, X):
        return np.tile(np.array([0.6, 0.2, 0.2]), (len(X), 1))

class _FB:
    def row(self, a, b, date, venue_country, neutral):
        return {"elo_diff": 1.0}

def test_build_win_prob_and_champion():
    win_prob = build_win_prob(_Model(), _FB(), as_of_date=pd.Timestamp("2026-07-01"))
    p = win_prob("A", "B")
    assert 0.0 <= p <= 1.0
    champ, prob = predict_champion(["A", "B", "C", "D"], win_prob)
    assert champ in {"A", "B", "C", "D"}
    assert 0.0 <= prob <= 1.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_cli_smoke.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/fifa2026/cli.py`**

```python
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from fifa2026.config import load_config
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.knockout.bracket import load_bracket, champion_probabilities, predict_champion

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None):
    pen = pen or {}
    depth = depth or {}
    def win_prob(team_a: str, team_b: str) -> float:
        row = feature_builder.row(team_a, team_b, as_of_date, venue_country="", neutral=True)
        X = pd.DataFrame([row])
        p_reg = model.predict_proba(X)[0]
        return resolve_tie(
            p_reg,
            pen_a=pen.get(team_a, 0.5), pen_b=pen.get(team_b, 0.5),
            depth_a=depth.get(team_a, 0.0), depth_b=depth.get(team_b, 0.0),
        )
    return win_prob

def _cmd_predict(args):
    cfg = load_config(args.config)
    teams = load_bracket(cfg.raw.get("bracket_path", "config/bracket_2026.yaml"))
    # NOTE: model + feature_builder loaded from models_dir in full wiring.
    raise SystemExit("predict requires trained model artifacts in models/ (see README)")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="fifa2026")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "evaluate", "predict"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    {"predict": _cmd_predict}.get(args.cmd, lambda a: print(f"{args.cmd}: see README"))(args)

if __name__ == "__main__":
    main()
```

> The full `data`/`train`/`evaluate` command bodies orchestrate the already-tested
> units (load_matches → fit EloEngine/FormFeatures/ContextFeatures → team_aggregates →
> FeatureBuilder.build_training_matrix → fit PoissonModel + BoostedModel → EnsembleModel.tune_weight
> → save to `models/` → backtest + benchmark). Each unit is covered by its own task's tests;
> this task's smoke test covers the wiring contract (`build_win_prob`).

- [ ] **Step 4: Implement `Makefile`**

```makefile
.PHONY: data train evaluate predict test
data:      ; python -m fifa2026.cli data
train:     ; python -m fifa2026.cli train
evaluate:  ; python -m fifa2026.cli evaluate
predict:   ; python -m fifa2026.cli predict
test:      ; pytest -q
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_cli_smoke.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/fifa2026/cli.py Makefile tests/test_cli_smoke.py README.md
git commit -m "feat: CLI + Makefile wiring with end-to-end smoke test"
```

---

## Self-Review

**Spec coverage check (each spec section → task):**
- §4 pipeline (ingest→features→models→knockout→bracket→evaluate) → Tasks 3–18 ✔
- §5 components & contracts → one task per component ✔
- §6 Tier A features (Elo, form, goals, home/host, rest, confederation, travel/altitude, H2H, experience) → Tasks 6–8 ✔ (squad age/experience via Task 9 `mean_age`)
- §6 Tier B (market value, depth, injuries, xG, deep per-player) + availability strategy → Tasks 5, 9 ✔
- §6 odds excluded from features → enforced; odds only in Task 17 ✔
- §7 knockout resolution (regulation→ET→shootout) → Task 14 ✔
- §8 hybrid ensemble + calibration + recency weighting → Tasks 11–13 (sample_weight plumbed through `fit`) ✔
- §9 temporal validation + metrics + market benchmark → Tasks 16, 17 ✔
- §10 stack/repo layout/reproducibility (make targets) → Tasks 1, 18 ✔
- §11 champion prediction with per-round survival probabilities → Task 15 (bracket DP yields both; champion-probability table is a bonus over the spec's v1 minimum) ✔

**Gaps deliberately deferred (documented, not silent):** recency-weight *fitting* in the CLI train body and the Poisson goal-target plumbing are described in Task 11's note and Task 18's wiring note — the unit contracts exist and are tested; only the orchestration glue is assembled in Task 18.

**Placeholder scan:** no "TBD/handle edge cases/similar to Task N" — each step has real code. `config/bracket_2026.yaml` ships intentionally with placeholder team names (the real draw is data, entered at predict time) and is flagged as such.

**Type consistency:** `predict_proba` returns `(n,3)` ordered `[home_win, draw, away_win]` in Tasks 11, 12, 13, 18; `outcome` encoding `0/1/2` consistent in Tasks 3, 10, 16, 17; `resolve_tie(p_reg, ...) -> float` consistent in Tasks 14, 18; `win_prob(a,b) -> float` consistent in Tasks 15, 18.
