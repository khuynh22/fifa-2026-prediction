# Availability-Adjusted Prediction — Design

**Date:** 2026-06-29
**Status:** Approved design (pending final user review)
**Builds on:** the end-to-end demo (`2026-06-28-fifa-2026-end-to-end-demo-design.md`)

## 1. Goal

Let current player **availability** (key absences) visibly influence the 2026
champion forecast, sourced from a small **maintained file** (not a live API),
folded into the model through an effective-Elo penalty. Also remove the broken,
now-unreachable squad-features-via-API path (which crashes `predict`).

## 2. Why curated, not the API (decision record)

A read-only probe of the user's real API-Football key established:
- The key authenticates (Free plan, 100 req/day) and team lookup works.
- **The free tier blocks the 2026 season** (`/injuries` for 2026 → "Free plans do
  not have access to this season, try from 2022 to 2024").
- **National-team injuries are empty even on an allowed season** (Argentina 2023
  → 0 results); API-Football injuries are club/league-based.

So current 2026 availability is not reachable via the free API. A **curated
`injuries_2026.yaml`** (maintained from team news) is free, reliable, and
user-controlled. The live-API path is retired here but the generic client stays
for a possible future paid-plan integration.

## 3. Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Availability source | Curated `data/reference/injuries_2026.yaml` (team → list of out players) |
| 2 | Blend mechanism | Effective-Elo penalty folded into `elo_diff` (symmetric → complementary win-prob still sums to 1) |
| 3 | Magnitude | `penalty_per_player = 10` Elo pts, `cap = 40`, both config-tunable; a transparent heuristic, not trained |
| 4 | Dead code | Remove `squad_enrich.py`, `squad_features.py`, `squads.py` + their tests; keep generic `api_client.py` + `cache.py` as future infra |
| 5 | Crash fix | Replace `FeatureBuilder.squad_agg` (adds columns → 14≠18 LightGBM crash) with `rating_adjustment` (folds into `elo_diff`, feature count stays 14) |

## 4. Data flow

```
data/reference/injuries_2026.yaml ─► load_injuries ─► {team: [players out]}
                                          │
                       availability_adjustment(penalty, cap)
                                          │
                       rating_adjustment {team: −Δelo}
                                          │
[predict] FeatureBuilder(rating_adjustment=...) folds into elo_diff
                                          │
          same complementary win_prob + bracket DP ─► forecast shifts
          (injured teams drop; meta records who/how much) ─► app panel
```

## 5. Components & contracts

- **`data/reference/injuries_2026.yaml`** (new, maintained):
  ```yaml
  # Key absences for the 2026 field (update from team news).
  # Teams not listed are treated as at full strength.
  injuries:
    France: ["Player A", "Player B"]
    Spain: ["Player C"]
  ```
  Ships with a clear template, seeded with 1–2 real current absences (fetched via
  a quick news check at build time) so the demo shows a real shift out of the box.

- **`src/fifa2026/squad_strength.py`** (new, pure functions):
  - `load_injuries(path) -> dict[str, list[str]]` — parse the yaml; `{}` if file
    absent or empty.
  - `availability_adjustment(injuries, penalty_per_player=10.0, cap=40.0) -> dict[str, float]`
    — for each listed team, `adj = -min(len(players) * penalty_per_player, cap)`;
    teams with no listed players are omitted (no adjustment).

- **`src/fifa2026/features/assemble.py` `FeatureBuilder`** (modify):
  - Signature: drop `squad_agg`; add `rating_adjustment: dict[str, float] | None = None`
    (stored as `{}` when None).
  - `row(...)`: `elo_diff = (elo.rating_before(a, date) + adj.get(a, 0.0))
    - (elo.rating_before(b, date) + adj.get(b, 0.0))`. Remove the `squad_agg`
    column block entirely. (The leakage guard in `build_training_matrix` is removed
    with the param; training simply uses no adjustment.)

- **`src/fifa2026/pipeline.py`** (modify):
  - `build_feature_builder(cfg, matches, rating_adjustment=None)`.
  - `run_train` / `run_evaluate`: build with no adjustment (training is unadjusted).
  - `run_predict`: load injuries (`cfg.raw["injuries_path"]`), compute
    `availability_adjustment(...)` with config knobs, pass as `rating_adjustment`,
    and store `{"availability": {team: {"out": [...], "elo_penalty": Δ}}}` in
    `PredictionResult.meta`.

- **`src/fifa2026/cli.py`** (modify): `_cmd_predict` loads injuries + computes the
  adjustment (replacing the removed `build_squad_agg` call), passes it to
  `run_predict`.

- **`config/default.yaml`** (modify): add
  `injuries_path: data/reference/injuries_2026.yaml` and
  `availability: {penalty_per_player: 10, cap: 40}`.

- **`app.py`** (modify): an **"Availability impact"** section listing adjusted
  teams (players out, Elo penalty) from `prediction.json` `meta.availability`.

## 6. Cleanup (removals)

Delete (dead now that the squad-feature path is gone and the API is unreachable):
`src/fifa2026/ingest/squads.py`, `src/fifa2026/features/squad_features.py`,
`src/fifa2026/squad_enrich.py`, and tests `tests/test_ingest_squads.py`,
`tests/test_squad_features.py`, `tests/test_squad_enrich.py`, plus fixture
`tests/fixtures/squad_sample.json`. Keep `ingest/api_client.py` + `cache.py`.
Update `README.md`: availability comes from the curated file; live-API enrichment
is documented as optional future (paid-plan) work.

## 7. Testing

- `squad_strength`: `availability_adjustment` math (count → capped negative delta;
  empty → omitted); `load_injuries` parses + handles a missing file.
- `FeatureBuilder`: a `rating_adjustment` makes `elo_diff` move by exactly
  `adj[a] - adj[b]`, and is symmetric under team swap.
- `run_predict` integration: a team listed with injuries has a **strictly lower**
  champion probability than with no injuries (same data); champion probs still sum
  to 1; no NaN; pinned ties unchanged.
- Suite stays green after the module removals (no dangling imports).
- Remove the now-obsolete `test_build_training_matrix_rejects_static_squad_agg`
  (and any other `squad_agg`-referencing assertions in `tests/test_assemble.py`),
  since the `squad_agg` param is gone; update the `test_pipeline_*` builders that
  pass `squad_agg=`/construct `FeatureBuilder` to the new signature.

## 8. Scope

- **In:** curated injuries file, availability adjustment math, FeatureBuilder
  refactor (fixing the crash), pipeline/cli/config wiring, app panel, dead-module
  removal, README update, a verified before/after forecast shift.
- **Out:** live-API availability (data-blocked); position/role weighting of
  injuries (YAGNI — simple per-player count); training-time player features
  (needs point-in-time historical data — still deferred).

## 9. Risks

- **Heuristic magnitude** — the penalty is not trained; mitigated by being small,
  capped, and config-tunable, and clearly labeled as a manual knob.
- **Curated-data freshness** — the forecast is only as current as the file; the
  app shows the `as_of` date and the adjustments applied, so it's transparent.
