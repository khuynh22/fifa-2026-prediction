# Availability-Adjusted Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let curated player-availability data (key absences) shift the 2026 champion forecast via an effective-Elo penalty, and remove the broken squad-features-via-API path that crashes `predict`.

**Architecture:** A new pure `squad_strength` module turns a maintained `injuries_2026.yaml` into `{team: -Δelo}`. `FeatureBuilder` swaps its `squad_agg` param for a `rating_adjustment` dict folded into `elo_diff` (symmetric → complementary win-prob preserved; feature count stays 14, fixing the crash). `run_predict` computes the adjustment and records it in `meta` for the app.

**Tech Stack:** Python 3.11+, pandas, numpy, pyyaml, plotly, streamlit, pytest.

## Global Constraints

- Python 3.11+, package `fifa2026`, `src/` layout. Use the project venv `.venv/Scripts/python.exe` for all commands.
- Availability is a **prediction-time** adjustment only; training stays unadjusted (no `rating_adjustment` for `run_train`/`run_evaluate`).
- The adjustment folds into `elo_diff` symmetrically (`adj[a] - adj[b]`), so `win_prob(a,b)+win_prob(b,a)` stays 1 and champion probabilities sum to 1.
- Penalty heuristic: `penalty_per_player = 10` Elo pts, `cap = 40`; both config-tunable.
- Removing the dead squad-API modules must leave the suite green (no dangling imports). Keep `ingest/api_client.py` + `cache.py`.
- Every task ends green (`.venv/Scripts/python.exe -m pytest -q`), pristine, and committed.

---

## File Structure

```
src/fifa2026/squad_strength.py          # NEW: load_injuries, availability_adjustment   [T1]
src/fifa2026/features/assemble.py        # MODIFY: squad_agg -> rating_adjustment         [T2]
src/fifa2026/pipeline.py                 # MODIFY: build_feature_builder + run_predict     [T3]
src/fifa2026/cli.py                      # MODIFY: _cmd_predict drops build_squad_agg      [T3]
config/default.yaml                      # MODIFY: injuries_path + availability knobs       [T3]
data/reference/injuries_2026.yaml        # NEW: curated availability (template)            [T3]
src/fifa2026/ingest/squads.py            # DELETE                                          [T4]
src/fifa2026/features/squad_features.py  # DELETE                                          [T4]
src/fifa2026/squad_enrich.py             # DELETE                                          [T4]
app.py                                   # MODIFY: "Availability impact" panel             [T5]
README.md                                # MODIFY: curated-injuries availability           [T5]
tests/...                                # squad_strength, assemble, pipeline_predict       [T1,T2,T3]; delete squad tests [T4]
```

---

### Task 1: `squad_strength` — load injuries + availability adjustment

**Files:**
- Create: `src/fifa2026/squad_strength.py`, `tests/test_squad_strength.py`

**Interfaces:**
- Produces: `load_injuries(path) -> dict[str, list[str]]` (parse the yaml's `injuries:` map; `{}` if file missing/empty). `availability_adjustment(injuries, penalty_per_player=10.0, cap=40.0) -> dict[str, float]` — per team with ≥1 listed player, `-min(n*penalty, cap)`; teams with no players omitted.

- [ ] **Step 1: Write the failing test** — `tests/test_squad_strength.py`:

```python
from fifa2026.squad_strength import load_injuries, availability_adjustment

def test_availability_adjustment_counts_and_caps():
    inj = {"France": ["A", "B"], "Spain": ["C"], "Brazil": [], "Italy": ["D","E","F","G","H"]}
    adj = availability_adjustment(inj, penalty_per_player=10.0, cap=40.0)
    assert adj["France"] == -20.0      # 2 * 10
    assert adj["Spain"] == -10.0       # 1 * 10
    assert "Brazil" not in adj         # empty list -> no adjustment
    assert adj["Italy"] == -40.0       # 5 * 10 capped at 40

def test_load_injuries_parses_and_missing_file(tmp_path):
    p = tmp_path / "inj.yaml"
    p.write_text("injuries:\n  France: [A, B]\n  Spain: [C]\n", encoding="utf-8")
    got = load_injuries(p)
    assert got == {"France": ["A", "B"], "Spain": ["C"]}
    assert load_injuries(tmp_path / "nope.yaml") == {}   # missing -> empty
```

- [ ] **Step 2: Run it, verify it fails** — `.venv/Scripts/python.exe -m pytest tests/test_squad_strength.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `src/fifa2026/squad_strength.py`**:

```python
from __future__ import annotations
from pathlib import Path
import yaml

def load_injuries(path) -> dict:
    """Parse the curated injuries file's `injuries:` map: {team: [players out]}.
    Returns {} if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    inj = data.get("injuries") or {}
    return {team: list(players or []) for team, players in inj.items()}

def availability_adjustment(injuries, penalty_per_player: float = 10.0,
                            cap: float = 40.0) -> dict:
    """Map team -> negative Elo delta from listed absences (capped). Teams with no
    listed players are omitted (no adjustment)."""
    adj = {}
    for team, players in injuries.items():
        n = len(players)
        if n <= 0:
            continue
        adj[team] = -min(n * penalty_per_player, cap)
    return adj
```

- [ ] **Step 4: Run it, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/squad_strength.py tests/test_squad_strength.py
git commit -m "feat: curated injuries loader and availability adjustment"
```

---

### Task 2: `FeatureBuilder` — `squad_agg` → `rating_adjustment` (fixes the crash)

**Files:**
- Modify: `src/fifa2026/features/assemble.py`
- Modify: `tests/test_assemble.py`

**Interfaces:**
- Produces: `FeatureBuilder(elo, form, context, confederations, hosts, form_windows, rating_adjustment=None)`. `row(...)` folds `rating_adjustment` into `elo_diff`: `(elo.rating_before(a,date)+adj[a]) - (elo.rating_before(b,date)+adj[b])`. The `squad_agg` param and its feature columns are removed; `build_training_matrix` no longer guards on `squad_agg`.

- [ ] **Step 1: Replace `src/fifa2026/features/assemble.py` entirely** with:

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from fifa2026.ingest.matches import outcome
from fifa2026.features.context import home_flag

class FeatureBuilder:
    def __init__(self, elo, form, context, confederations, hosts, form_windows,
                 rating_adjustment=None):
        """Builds per-match A-vs-B differential features.

        rating_adjustment : dict[str, float] | None
            Optional prediction-time Elo deltas per team (e.g. an availability
            penalty). Folded into ``elo_diff``. Leave None for training.
        """
        self.elo = elo
        self.form = form
        self.context = context
        self.confederations = confederations
        self.hosts = hosts
        self.form_windows = form_windows
        self.rating_adjustment = rating_adjustment or {}

    def row(self, home_team, away_team, date, venue_country, neutral) -> dict:
        a, b = home_team, away_team
        adj = self.rating_adjustment
        feats = {}
        feats["elo_diff"] = ((self.elo.rating_before(a, date) + adj.get(a, 0.0))
                             - (self.elo.rating_before(b, date) + adj.get(b, 0.0)))
        for w in self.form_windows:
            fa = self.form.team_form(a, date, w)
            fb = self.form.team_form(b, date, w)
            feats[f"ppg_{w}_diff"] = fa[f"ppg_{w}"] - fb[f"ppg_{w}"]
            feats[f"gf_rate_{w}_diff"] = fa[f"gf_rate_{w}"] - fb[f"gf_rate_{w}"]
            feats[f"ga_rate_{w}_diff"] = fa[f"ga_rate_{w}"] - fb[f"ga_rate_{w}"]
        feats["rest_diff"] = self.context.rest_days(a, date) - self.context.rest_days(b, date)
        feats["h2h_ppg"] = self.context.head_to_head(a, b, date)["h2h_ppg"]
        feats["home_diff"] = (home_flag(a, venue_country, self.hosts)
                              - home_flag(b, venue_country, self.hosts))
        feats["same_confed"] = int(self.confederations.get(a) == self.confederations.get(b)
                                   and self.confederations.get(a) is not None)
        return feats

    def build_training_matrix(self, matches: pd.DataFrame):
        rows, labels, goals_home, goals_away = [], [], [], []
        m = matches.dropna(subset=["home_score", "away_score"]).sort_values("date")
        for _, g in m.iterrows():
            rows.append(self.row(g["home_team"], g["away_team"], g["date"],
                                 g.get("country", ""), bool(g.get("neutral", True))))
            labels.append(outcome(int(g["home_score"]), int(g["away_score"])))
            goals_home.append(int(g["home_score"]))
            goals_away.append(int(g["away_score"]))
        X = pd.DataFrame(rows).fillna(0.0)
        return X, np.array(labels), np.array(goals_home), np.array(goals_away)
```

- [ ] **Step 2: Update `tests/test_assemble.py`** — change `_builder` to the new signature, delete the obsolete `squad_agg` rejection test, add a `rating_adjustment` test. Replace lines 18–25 (`_builder`) with:

```python
def _builder(m, rating_adjustment=None):
    return FeatureBuilder(
        elo=EloEngine(home_advantage=0).fit(m),
        form=FormFeatures().fit(m),
        context=ContextFeatures().fit(m),
        confederations={"A": "UEFA", "B": "UEFA"},
        hosts=[], form_windows=[5], rating_adjustment=rating_adjustment,
    )
```

Delete the whole `test_build_training_matrix_rejects_static_squad_agg` function (lines 47–55). Add this test:

```python
def test_rating_adjustment_shifts_elo_diff_symmetrically():
    m = _matches()
    base = _builder(m).row("A", "B", pd.Timestamp("2010-03-01"), "Y", True)
    adj = _builder(m, rating_adjustment={"A": -30.0}).row("A", "B", pd.Timestamp("2010-03-01"), "Y", True)
    # A penalized by 30 -> elo_diff drops by exactly 30; swapping teams flips the sign.
    assert abs((adj["elo_diff"] - base["elo_diff"]) - (-30.0)) < 1e-9
    swapped = _builder(m, rating_adjustment={"A": -30.0}).row("B", "A", pd.Timestamp("2010-03-01"), "Y", True)
    assert abs(adj["elo_diff"] + swapped["elo_diff"]) < 1e-9  # symmetric
```

Also remove the now-unused `import pytest` at line 1 (no test uses it after deleting the rejection test).

- [ ] **Step 3: Run the assemble tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_assemble.py -q`
Expected: PASS (leakage test still passes; new adjustment test passes).

- [ ] **Step 4: Run the full suite** — `.venv/Scripts/python.exe -m pytest -q`. NOTE: `pipeline.py`/`cli.py` still pass `squad_agg=` and will FAIL here; that's expected and fixed in Task 3. Confirm the only failures are the `squad_agg` TypeErrors in pipeline/cli tests, then proceed (do not commit a red suite — commit happens after Task 3 makes it green). If you prefer a green commit now, you may instead do Tasks 2 and 3 as one commit.

- [ ] **Step 5: Commit (assemble change; suite goes green after Task 3)**

```bash
git add src/fifa2026/features/assemble.py tests/test_assemble.py
git commit -m "refactor: FeatureBuilder rating_adjustment replaces squad_agg (fixes predict crash)"
```

---

### Task 3: Wire availability into pipeline + cli + config

**Files:**
- Modify: `src/fifa2026/pipeline.py`, `src/fifa2026/cli.py`, `config/default.yaml`
- Create: `data/reference/injuries_2026.yaml`
- Modify: `tests/test_pipeline_predict.py`

**Interfaces:**
- Consumes: `squad_strength.load_injuries`, `availability_adjustment`; `FeatureBuilder(... rating_adjustment=...)`.
- Produces: `build_feature_builder(cfg, matches, rating_adjustment=None)`; `run_predict(cfg, models=None, matches_csv=None, bracket_path=None, injuries_path=None)` (computes the adjustment, stores `meta["availability"] = {team: {"out": [...], "elo_penalty": Δ}}`).

- [ ] **Step 1: Edit `build_feature_builder` in `pipeline.py`** — replace its signature/body:

```python
def build_feature_builder(cfg, matches, rating_adjustment=None):
    elo = EloEngine(
        k=cfg.raw["elo"]["k"], home_advantage=cfg.raw["elo"]["home_advantage"],
        initial=cfg.raw["elo"]["initial"]).fit(matches)
    form = FormFeatures().fit(matches)
    context = ContextFeatures().fit(matches)
    confed = load_confederations(_ref_dir(cfg) / "confederations.csv")
    return FeatureBuilder(elo=elo, form=form, context=context, confederations=confed,
                          hosts=cfg.raw["hosts_2026"],
                          form_windows=cfg.raw["features"]["form_windows"],
                          rating_adjustment=rating_adjustment)
```

In `run_train` and `run_evaluate`, change `build_feature_builder(cfg, <matches>, squad_agg=None)` to `build_feature_builder(cfg, <matches>)` (two call sites — keep the same matches argument each used).

- [ ] **Step 2: Replace `run_predict` in `pipeline.py`** with (adds injuries → adjustment → meta):

```python
def run_predict(cfg, models=None, matches_csv=None, bracket_path=None, injuries_path=None) -> PredictionResult:
    from fifa2026.persistence import load_models
    from fifa2026.squad_strength import load_injuries, availability_adjustment
    csv = matches_csv or cfg.raw["sources"]["results_csv"]
    matches = load_matches(csv, train_start=cfg.train_start)
    ensemble = models if models is not None else load_models(cfg.models_dir)[0]
    bpath = bracket_path or cfg.raw.get("bracket_path", "config/bracket_2026.yaml")
    teams, decided, as_of = _load_bracket_cfg(bpath)
    as_of_date = pd.Timestamp(as_of) if as_of else matches["date"].max()
    matches = matches[matches["date"] <= as_of_date]
    ipath = injuries_path or cfg.raw.get("injuries_path", "data/reference/injuries_2026.yaml")
    injuries = load_injuries(ipath)
    av = cfg.raw.get("availability", {})
    rating_adjustment = availability_adjustment(
        injuries, penalty_per_player=av.get("penalty_per_player", 10.0),
        cap=av.get("cap", 40.0))
    fb = build_feature_builder(cfg, matches, rating_adjustment=rating_adjustment)
    win_prob = build_win_prob(ensemble, fb, as_of_date, decided=decided)
    champ = champion_probabilities(teams, win_prob)
    rounds = round_probabilities(teams, win_prob)
    ties = [{"home": teams[i], "away": teams[i + 1],
             "p_home": win_prob(teams[i], teams[i + 1])} for i in range(0, len(teams), 2)]
    availability_meta = {t: {"out": injuries.get(t, []), "elo_penalty": rating_adjustment[t]}
                         for t in rating_adjustment}
    result = PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties,
                              as_of=as_of, meta={"n_teams": len(teams),
                                                 "availability": availability_meta})
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "prediction.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 3: Edit `_cmd_predict` in `cli.py`** — remove the `build_squad_agg` import + call; predict now self-contains the adjustment:

```python
def _cmd_predict(args):
    from fifa2026.pipeline import run_predict
    cfg = load_config(args.config)
    res = run_predict(cfg)
    top = sorted(res.champion_probs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Champion probabilities (top 8):")
    for team, p in top:
        print(f"  {team:25s} {p:6.1%}")
```

(Remove the now-unused `from fifa2026.knockout.bracket import load_bracket` line inside `_cmd_predict` if present.)

- [ ] **Step 4: Add to `config/default.yaml`** (top level):

```yaml
injuries_path: data/reference/injuries_2026.yaml
availability:
  penalty_per_player: 10
  cap: 40
```

- [ ] **Step 5: Create `data/reference/injuries_2026.yaml`** (template, empty by default so the forecast is unchanged until filled):

```yaml
# Curated key absences for the 2026 field — update from team news.
# Each team maps to a list of players who are out. Teams not listed (or with an
# empty list) are treated as at full strength. Re-run `make predict` after editing.
injuries: {}
```

- [ ] **Step 6: Update `tests/test_pipeline_predict.py`** — make the existing test deterministic (pass an empty injuries file) and add the availability-shift test. The existing test calls `run_predict(cfg, matches_csv=csv, bracket_path=bracket)`; add `injuries_path` to it. Append a new test:

```python
def test_injuries_lower_champion_prob(tmp_path):
    from fifa2026.config import load_config
    from fifa2026.pipeline import run_train, run_predict
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
```

> Note: confirm `_synth_csv` in this test module yields the 8 team names above; if its existing helper uses different names, point the bracket at those names instead — the teams in the bracket must exist in the synthetic data.

- [ ] **Step 7: Run the full suite, verify green + pristine**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS (Task 2's pipeline failures are now resolved).

- [ ] **Step 8: Commit**

```bash
git add src/fifa2026/pipeline.py src/fifa2026/cli.py config/default.yaml data/reference/injuries_2026.yaml tests/test_pipeline_predict.py
git commit -m "feat: wire curated availability adjustment into predict"
```

---

### Task 4: Remove the dead squad-API modules

**Files:**
- Delete: `src/fifa2026/ingest/squads.py`, `src/fifa2026/features/squad_features.py`, `src/fifa2026/squad_enrich.py`
- Delete: `tests/test_ingest_squads.py`, `tests/test_squad_features.py`, `tests/test_squad_enrich.py`, `tests/fixtures/squad_sample.json`

**Interfaces:** none (removal only). Keep `src/fifa2026/ingest/api_client.py` and `src/fifa2026/cache.py`.

- [ ] **Step 1: Verify nothing still imports the doomed modules**

Run: `.venv/Scripts/python.exe -c "import subprocess; print('check refs')"` then grep:
`grep -rnE "squad_enrich|squad_features|ingest.squads|parse_squad|team_aggregates|build_squad_agg" src tests` 
Expected: no matches in `src/` or `tests/` (Task 3 removed the last `cli.py` reference). If any remain, stop and report — they must be cleared first.

- [ ] **Step 2: Delete the files**

```bash
git rm src/fifa2026/ingest/squads.py src/fifa2026/features/squad_features.py src/fifa2026/squad_enrich.py \
       tests/test_ingest_squads.py tests/test_squad_features.py tests/test_squad_enrich.py \
       tests/fixtures/squad_sample.json
```

- [ ] **Step 3: Run the full suite, verify green + pristine** (no collection/import errors)

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS, fewer tests than before, no import errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove dead squad-API feature path (data unreachable on free tier)"
```

---

### Task 5: App "Availability impact" panel + README

**Files:**
- Modify: `app.py`, `README.md`

**Interfaces:**
- Consumes: `prediction.json` `meta.availability` (`{team: {"out": [...], "elo_penalty": Δ}}`).

- [ ] **Step 1: Add an availability tab to `app.py`'s `main()`** — change the `st.tabs([...])` list to include "Availability" and add its block. Find the existing `tabs = st.tabs([...])` and replace with:

```python
    tabs = st.tabs(["Champion odds", "Bracket", "Survival", "Team explorer",
                    "Calibration", "Availability"])
```

Then before the final `if __name__` guard, inside `main()` after the calibration tab block, add:

```python
    with tabs[5]:
        st.subheader("Availability impact")
        avail = (prediction.get("meta") or {}).get("availability") or {}
        if not avail:
            st.info("No availability adjustments applied. Edit "
                    "data/reference/injuries_2026.yaml and re-run `make predict`.")
        else:
            import pandas as pd
            rows = [{"team": t, "players out": ", ".join(v.get("out", [])) or len(v.get("out", [])),
                     "elo penalty": v.get("elo_penalty")} for t, v in avail.items()]
            st.dataframe(pd.DataFrame(rows).sort_values("elo penalty"))
```

- [ ] **Step 2: Update `README.md`** — under the feature list / "What it does", add a bullet:

```markdown
- Adjusts the forecast for **player availability**: key absences listed in
  `data/reference/injuries_2026.yaml` apply an effective-Elo penalty (free,
  curated). Live-API enrichment is optional future work (the free API tier does
  not expose the 2026 season or national-team injuries).
```

- [ ] **Step 3: Run the app smoke test + full suite** (the smoke test builds figures from a fixture; the new tab uses Streamlit only inside `main()`, so `build_dashboard` is unaffected):

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py README.md
git commit -m "feat: availability-impact panel in the demo app"
```

---

### Task 6 (controller-run): Seed real absences + verify the forecast shift

**Files:** Modify `data/reference/injuries_2026.yaml` (real current absences).

**Interfaces:** none — runs the real pipeline and verifies the before/after shift.

- [ ] **Step 1: Capture the baseline** — `.venv/Scripts/python.exe -m fifa2026.cli predict` with the empty injuries file; record the top-8 champion table.

- [ ] **Step 2: Seed real absences** — populate `data/reference/injuries_2026.yaml` with current key absences for a few 2026 teams (from a news check), e.g.:

```yaml
injuries:
  France: ["<real out player>"]
```

- [ ] **Step 3: Re-run predict and diff** — `.venv/Scripts/python.exe -m fifa2026.cli predict`; confirm the listed teams' champion probabilities dropped vs. the baseline, champion probs still sum to 1, no NaN, pinned ties unchanged, and `reports/prediction.json` `meta.availability` lists the teams.

- [ ] **Step 4: App check** — confirm the "Availability" tab renders the adjustments (build the dashboard from the real `prediction.json`).

- [ ] **Step 5: Commit the seeded file**

```bash
git add data/reference/injuries_2026.yaml
git commit -m "data: seed current 2026 player absences"
```

---

## Self-Review

**Spec coverage:**
- §5 `squad_strength` (load_injuries, availability_adjustment) → T1 ✔
- §5 FeatureBuilder `rating_adjustment` + crash fix → T2 ✔
- §5 pipeline/cli/config wiring + meta.availability → T3 ✔
- §5 `injuries_2026.yaml` → T3 (template), T6 (seeded) ✔
- §5 app panel + §6 README → T5 ✔
- §6 remove dead modules (keep api_client+cache) → T4 ✔
- §7 tests (availability math, elo-diff shift symmetric, predict injury-shift, suite green post-removal) → T1, T2, T3, T4 ✔
- §7 remove obsolete squad_agg test → T2 ✔

**Placeholder scan:** none. The `injuries_2026.yaml` ships as an explicit empty template (`injuries: {}`); real values are seeded in T6 (flagged), not left as a vague gap. The README/yaml `<real out player>` is a controller-fill in T6, not code.

**Type consistency:** `rating_adjustment: dict[str,float]` consistent T1→T2→T3. `availability_adjustment(injuries, penalty_per_player, cap)` signature consistent T1/T3. `meta["availability"][team] = {"out": list, "elo_penalty": float}` consistent T3/T5. `build_feature_builder(cfg, matches, rating_adjustment=None)` consistent T3 (all call sites updated). `run_predict(..., injuries_path=None)` consistent T3 (pipeline) and T3 tests.
