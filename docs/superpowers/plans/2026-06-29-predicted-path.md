# Predicted Path to the Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project a most-likely-path knockout bracket (predicted winner of each tie advances to a predicted Final + champion) with per-match regulation W/D/L + shootout detail, shown in a new "Predicted path" tab, and predict all 16 Round-of-32 ties fresh.

**Architecture:** A new `knockout/walk.py` holds `match_breakdown` (symmetric per-match W/D/L + shootout), `build_win_prob` (moved here, derives its advance prob from `match_breakdown`), and `walk_bracket` (round-by-round projection). `run_predict` stores the walk in `meta["predicted_path"]`; the app renders it.

**Tech Stack:** Python 3.11+, pandas, numpy, plotly, streamlit, pyyaml, pytest.

## Global Constraints

- Python 3.11+, package `fifa2026`, `src/` layout. Use `.venv/Scripts/python.exe` for all commands.
- The per-match `p_a_advance` MUST equal the existing complementary win-prob `0.5*(resolve_tie(pab) + (1 - resolve_tie(pba)))` (enforced by test) — so the new tab cannot disagree with the champion-odds tab.
- Regulation probs `p_a_reg + p_draw + p_b_reg == 1`; `p_a_advance + p_b_advance == 1` (symmetric).
- Shootouts use the existing `shootout_prob` (≈0.5 with no penalty data) — shown explicitly, not hidden.
- The new "Predicted path" tab is ADDITIVE; existing tabs unchanged. `app.py` stays importable without Streamlit (all `st.*` inside `main()`).
- `build_win_prob` must remain importable from `fifa2026.cli` (re-export) so existing imports/tests keep working.
- Every task ends green (`.venv/Scripts/python.exe -m pytest -q`), pristine, and committed.

---

## File Structure

```
src/fifa2026/knockout/walk.py    # NEW: match_breakdown, build_win_prob, walk_bracket   [T1]
src/fifa2026/cli.py              # MODIFY: re-export build_win_prob from walk; drop local def [T1]
src/fifa2026/pipeline.py         # MODIFY: run_predict stores meta["predicted_path"]      [T2]
config/bracket_2026.yaml         # MODIFY: decided: []  (predict all 16 R32)              [T3]
app.py                           # MODIFY: "Predicted path" tab                           [T3]
README.md                        # MODIFY: note the predicted-path tab                    [T3]
tests/test_walk.py               # NEW                                                    [T1]
```

---

### Task 1: `knockout/walk.py` — match breakdown, win-prob, bracket walk

**Files:**
- Create: `src/fifa2026/knockout/walk.py`, `tests/test_walk.py`
- Modify: `src/fifa2026/cli.py`

**Interfaces:**
- Produces:
  - `match_breakdown(model, feature_builder, team_a, team_b, as_of_date, pen=None, depth=None, decided=None) -> dict` with keys `team_a, team_b, decided, winner, p_a_reg, p_draw, p_b_reg, p_a_shootout, p_a_advance, p_b_advance`.
  - `build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None)` returning a `win_prob(a,b) -> float` that equals `match_breakdown(...)["p_a_advance"]`.
  - `walk_bracket(teams, breakdown_fn) -> {"rounds": [{"round": str, "matches": [dict,...]}], "champion": str}` where `breakdown_fn(a,b) -> dict` must include `"winner"`.
  - `ROUND_NAMES` dict.

- [ ] **Step 1: Write the failing test** — `tests/test_walk.py`:

```python
import numpy as np
import pandas as pd
from fifa2026.knockout.resolve import resolve_tie
from fifa2026.knockout.walk import match_breakdown, build_win_prob, walk_bracket

class _FB:
    hosts = []
    def row(self, a, b, date, venue_country, neutral):
        # encodes which team is "home" so the two orderings differ
        return {"home_is_a": 1.0 if a == "A" else 0.0}

class _Model:
    def predict_proba(self, X):
        out = []
        for v in X["home_is_a"]:
            out.append([0.6, 0.2, 0.2] if v == 1.0 else [0.5, 0.2, 0.3])
        return np.array(out)

def test_breakdown_sums_and_consistency():
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"))
    assert abs(bd["p_a_reg"] + bd["p_draw"] + bd["p_b_reg"] - 1.0) < 1e-9
    assert abs(bd["p_a_advance"] + bd["p_b_advance"] - 1.0) < 1e-9
    # consistency with the old complementary win-prob formula
    pab = np.array([0.6, 0.2, 0.2]); pba = np.array([0.5, 0.2, 0.3])
    expected = 0.5 * (resolve_tie(pab) + (1 - resolve_tie(pba)))
    assert abs(bd["p_a_advance"] - expected) < 1e-9          # 0.55
    assert bd["winner"] == "A"                                # 0.55 >= 0.5

def test_build_win_prob_matches_breakdown():
    win_prob = build_win_prob(_Model(), _FB(), pd.Timestamp("2026-07-01"))
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"))
    assert abs(win_prob("A", "B") - bd["p_a_advance"]) < 1e-9

def test_decided_tie_locks():
    decided = {frozenset(("A", "B")): "A"}
    bd = match_breakdown(_Model(), _FB(), "A", "B", pd.Timestamp("2026-07-01"), decided=decided)
    assert bd["decided"] and bd["winner"] == "A"
    assert bd["p_a_advance"] == 1.0 and bd["p_b_advance"] == 0.0

def test_walk_bracket_structure():
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strength = {t: s for t, s in zip(teams, [8, 1, 7, 2, 6, 3, 5, 4])}
    def bd(a, b):
        wa = strength[a] > strength[b]
        return {"team_a": a, "team_b": b, "winner": a if wa else b,
                "p_a_advance": 1.0 if wa else 0.0}
    res = walk_bracket(teams, bd)
    assert [len(r["matches"]) for r in res["rounds"]] == [4, 2, 1]
    assert res["rounds"][0]["round"] == "Quarter-finals"  # 8 teams
    assert res["champion"] == "A"
```

- [ ] **Step 2: Run it, verify it fails** — `.venv/Scripts/python.exe -m pytest tests/test_walk.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `src/fifa2026/knockout/walk.py`**:

```python
from __future__ import annotations
import pandas as pd
from fifa2026.knockout.resolve import shootout_prob

ROUND_NAMES = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
               4: "Semi-finals", 2: "Final"}

def match_breakdown(model, feature_builder, team_a, team_b, as_of_date,
                    pen=None, depth=None, decided=None) -> dict:
    pen = pen or {}
    depth = depth or {}
    decided = decided or {}
    key = frozenset((team_a, team_b))
    if key in decided:
        a_adv = 1.0 if decided[key] == team_a else 0.0
        return {"team_a": team_a, "team_b": team_b, "decided": True,
                "winner": decided[key], "p_a_reg": a_adv, "p_draw": 0.0,
                "p_b_reg": 1.0 - a_adv, "p_a_shootout": a_adv,
                "p_a_advance": a_adv, "p_b_advance": 1.0 - a_adv}
    hosts = getattr(feature_builder, "hosts", []) or []
    venue = team_a if team_a in hosts else (team_b if team_b in hosts else "")
    row_ab = feature_builder.row(team_a, team_b, as_of_date, venue_country=venue, neutral=(venue == ""))
    row_ba = feature_builder.row(team_b, team_a, as_of_date, venue_country=venue, neutral=(venue == ""))
    pab = model.predict_proba(pd.DataFrame([row_ab]))[0]
    pba = model.predict_proba(pd.DataFrame([row_ba]))[0]
    p_a_reg = 0.5 * (pab[0] + pba[2])
    p_draw = 0.5 * (pab[1] + pba[1])
    p_b_reg = 0.5 * (pab[2] + pba[0])
    s_a = shootout_prob(pen.get(team_a, 0.5), pen.get(team_b, 0.5),
                        depth.get(team_a, 0.0), depth.get(team_b, 0.0))
    p_a_adv = float(p_a_reg + p_draw * s_a)
    winner = team_a if p_a_adv >= 0.5 else team_b
    return {"team_a": team_a, "team_b": team_b, "decided": False, "winner": winner,
            "p_a_reg": float(p_a_reg), "p_draw": float(p_draw), "p_b_reg": float(p_b_reg),
            "p_a_shootout": float(s_a), "p_a_advance": p_a_adv,
            "p_b_advance": float(1.0 - p_a_adv)}

def build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None):
    def win_prob(team_a: str, team_b: str) -> float:
        return match_breakdown(model, feature_builder, team_a, team_b, as_of_date,
                               pen=pen, depth=depth, decided=decided)["p_a_advance"]
    return win_prob

def walk_bracket(teams, breakdown_fn) -> dict:
    """Most-likely path: each tie's predicted winner advances to the next round."""
    rounds = []
    current = list(teams)
    while len(current) > 1:
        size = len(current)
        matches = [breakdown_fn(current[i], current[i + 1]) for i in range(0, size, 2)]
        rounds.append({"round": ROUND_NAMES.get(size, f"Round of {size}"), "matches": matches})
        current = [m["winner"] for m in matches]
    return {"rounds": rounds, "champion": current[0] if current else None}
```

- [ ] **Step 4: Edit `src/fifa2026/cli.py`** — remove the local `build_win_prob` definition (the `def build_win_prob...` block) and re-export it from `walk`. Specifically: delete the `import pandas as pd` line and the `from fifa2026.knockout.resolve import resolve_tie` line (both only used by the old `build_win_prob`), delete the entire `def build_win_prob(...)` function, and add this import near the other imports at the top:

```python
from fifa2026.knockout.walk import build_win_prob  # re-exported for callers
```

(Keep `load_dotenv`, `from fifa2026.config import load_config`, and all `_cmd_*` functions exactly as they are.)

- [ ] **Step 5: Run `test_walk.py` and `test_cli_smoke.py`, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_walk.py tests/test_cli_smoke.py -q`
Expected: PASS (the smoke test imports `build_win_prob` from `fifa2026.cli`, which now resolves to the re-export, and still satisfies its complementary/champion assertions).

- [ ] **Step 6: Run the full suite** — `.venv/Scripts/python.exe -m pytest -q` → all PASS, pristine.

- [ ] **Step 7: Commit**

```bash
git add src/fifa2026/knockout/walk.py src/fifa2026/cli.py tests/test_walk.py
git commit -m "feat: match breakdown + bracket walk; move build_win_prob to walk"
```

---

### Task 2: `run_predict` stores the predicted path

**Files:**
- Modify: `src/fifa2026/pipeline.py`
- Modify: `tests/test_pipeline_predict.py`

**Interfaces:**
- Consumes: `walk.match_breakdown`, `walk.walk_bracket`.
- Produces: `prediction.json` `meta["predicted_path"] = {"rounds": [...], "champion": str}`.

- [ ] **Step 1: Edit `run_predict` in `pipeline.py`** — after `win_prob = build_win_prob(...)` is built and `champ`/`rounds`/`ties` are computed, add the walk. Insert the import at the top of `run_predict` alongside the existing local imports, and build the path. Replace the block that constructs `availability_meta` + `result` with:

```python
    from fifa2026.knockout.walk import match_breakdown, walk_bracket
    def _breakdown(a, b):
        return match_breakdown(ensemble, fb, a, b, as_of_date, decided=decided)
    predicted_path = walk_bracket(teams, _breakdown)
    availability_meta = {t: {"out": injuries.get(t, []), "elo_penalty": rating_adjustment[t]}
                         for t in rating_adjustment}
    result = PredictionResult(champion_probs=champ, round_probs=rounds, tie_probs=ties,
                              as_of=as_of, meta={"n_teams": len(teams),
                                                 "availability": availability_meta,
                                                 "predicted_path": predicted_path})
```

(Leave the `reports/prediction.json` write and `return result` lines below it unchanged.)

- [ ] **Step 2: Add a test to `tests/test_pipeline_predict.py`** — in the existing `test_run_predict_sums_to_one_and_pins` (or a new test reusing its setup), after `res = run_predict(...)`, assert the path is present and well-formed. Append this new test (it reuses the module's `_synth_csv` 8-team helper and trains a model):

```python
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
```

- [ ] **Step 3: Run the predict tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_predict.py -q`
Expected: PASS.

- [ ] **Step 4: Run the full suite** — `.venv/Scripts/python.exe -m pytest -q` → all PASS, pristine.

- [ ] **Step 5: Commit**

```bash
git add src/fifa2026/pipeline.py tests/test_pipeline_predict.py
git commit -m "feat: store predicted path in prediction meta"
```

---

### Task 3: Predict all 16 fresh + app "Predicted path" tab + README

**Files:**
- Modify: `config/bracket_2026.yaml`, `app.py`, `README.md`

**Interfaces:**
- Consumes: `prediction.json` `meta["predicted_path"]`.

- [ ] **Step 1: Edit `config/bracket_2026.yaml`** — replace the four-entry `decided:` block with an empty list so all 16 R32 ties are predicted fresh:

```yaml
# Already-played knockout ties (pinned to certainty). Empty = predict all fresh.
# Add entries like {winner: France, loser: Sweden} as real results come in.
decided: []
```

(Leave the `teams:` list and `as_of:` exactly as they are.)

- [ ] **Step 2: Add the "Predicted path" tab to `app.py`'s `main()`** — extend the tabs list to 7 and add the block. Change the `tabs = st.tabs([...])` line to:

```python
    tabs = st.tabs(["Champion odds", "Bracket", "Survival", "Team explorer",
                    "Calibration", "Availability", "Predicted path"])
```

Then add this block after the `with tabs[5]:` (Availability) block, still inside `main()`:

```python
    with tabs[6]:
        st.subheader("Predicted path to the Final")
        st.caption("Single most-likely bracket — each tie's predicted winner advances. "
                   "Shootouts are modeled as ~50/50 (no penalty data).")
        path = (prediction.get("meta") or {}).get("predicted_path") or {}
        if not path.get("rounds"):
            st.info("No predicted path available. Run `make predict`.")
        else:
            import pandas as pd
            st.success(f"Predicted champion: {path.get('champion')}")
            for rnd in path["rounds"]:
                st.markdown(f"**{rnd['round']}**")
                rows = [{
                    "match": f"{m['team_a']} vs {m['team_b']}",
                    "P(A win)": round(m["p_a_reg"], 3),
                    "P(draw)": round(m["p_draw"], 3),
                    "P(B win)": round(m["p_b_reg"], 3),
                    "shootout A": round(m["p_a_shootout"], 2),
                    "P(adv A)": round(m["p_a_advance"], 3),
                    "winner": m["winner"],
                } for m in rnd["matches"]]
                st.dataframe(pd.DataFrame(rows))
```

- [ ] **Step 3: Update `README.md`** — add a bullet under the feature list:

```markdown
- Projects a **most-likely-path bracket**: the predicted winner of each tie
  advances to a predicted Final + champion, with per-match regulation
  win/draw/loss and (~50/50) shootout odds — shown in the "Predicted path" tab.
```

- [ ] **Step 4: Run the full suite, verify pass** (the app smoke test imports `app` and calls `build_dashboard`, which is unchanged; the new tab is inside `main()`):

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS, pristine.

- [ ] **Step 5: Commit**

```bash
git add config/bracket_2026.yaml app.py README.md
git commit -m "feat: predict all R32 fresh and add Predicted path tab"
```

---

### Task 4 (controller-run): Real run + present the predicted bracket

**Files:** none (runs the pipeline, verifies, reports).

**Interfaces:** none.

- [ ] **Step 1: Re-run predict on real data** — `.venv/Scripts/python.exe -m fifa2026.cli predict`. Confirm it writes `reports/prediction.json` with `meta.predicted_path` (5 rounds for 32 teams) and no error.

- [ ] **Step 2: Read out the predicted path** from `reports/prediction.json`: print, per round, each tie with `p_a_reg/p_draw/p_b_reg`, shootout, and winner, plus the predicted champion. Confirm champion probs still sum to 1, no NaN.

- [ ] **Step 3: Sanity-check** — favorites advance (e.g., Argentina, Spain, France deep), no minnow predicted as champion, the R32 winners are plausible (cross-check the three known results: Canada beat South Africa, Germany beat Paraguay — the model should at least not be wildly against known-strong sides). Note any tie the model calls a coin-flip (advance prob near 0.5).

- [ ] **Step 4: Launch the app** (`.venv/Scripts/streamlit run app.py`) and confirm the "Predicted path" tab renders the rounds + champion.

- [ ] **Step 5: Report** the predicted bracket to the user and set up the update loop (real result → add to `decided` → re-run).

---

## Self-Review

**Spec coverage:**
- §3 per-match breakdown (symmetric, consistency with win_prob) → T1 (`match_breakdown` + consistency test) ✔
- §4 bracket walk (most-likely path, rounds + champion) → T1 (`walk_bracket`) ✔
- §5 components: walk.py + cli re-export → T1; run_predict meta.predicted_path → T2; app tab → T3; bracket decided cleared → T3 ✔
- §2 decision 2 (all 16 fresh) → T3 (`decided: []`) ✔
- §6 update workflow → operational (T4 sets it up) ✔
- §7 testing (breakdown sums, consistency, decided lock, walk structure, predicted_path present) → T1, T2 ✔

**Placeholder scan:** none. T4 is a controller verification task (no code), explicitly so.

**Type consistency:** `match_breakdown(...)` dict keys (`team_a, team_b, decided, winner, p_a_reg, p_draw, p_b_reg, p_a_shootout, p_a_advance, p_b_advance`) consistent across T1/T2/T3. `walk_bracket(teams, breakdown_fn) -> {"rounds", "champion"}` consistent T1/T2/T3. `build_win_prob` signature unchanged (still importable from `fifa2026.cli`) T1. `meta["predicted_path"]` consistent T2/T3.
