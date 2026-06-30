# Predicted Path to the Final — Design

**Date:** 2026-06-29
**Status:** Approved design (pending final user review)
**Builds on:** the availability-adjusted demo (current `main`)

## 1. Goal

Add a **most-likely-path** projection: walk the knockout bracket round by round
(the model's predicted winner of each tie advances and forms the next round) all
the way to a predicted **Final and champion**, with each match annotated by its
**regulation win/draw/loss probabilities and shootout odds**. Surfaced as a new
"Predicted path" tab. Predict all 16 Round-of-32 ties fresh so the model can be
compared against real results as they come in.

## 2. Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Projection style | Most-likely path: each tie's predicted winner (argmax advance prob) advances to form the next round, to a single Final + champion |
| 2 | R32 state | Predict ALL 16 ties fresh — clear the `decided` list in `bracket_2026.yaml` |
| 3 | Per-match detail | Symmetric regulation W/D/L + shootout split + overall advance prob |
| 4 | Shootouts | Modeled ~50/50 (no penalty-record data); shown explicitly as a near-coin-flip |
| 5 | New tab | Additive — existing Champion-odds / Bracket / Survival / Team / Calibration / Availability tabs unchanged |
| 6 | Consistency | The per-match `p_a_advance` MUST equal the existing bracket-DP `win_prob` (enforced by test) |

## 3. The per-match breakdown (math)

For a tie A vs B, computed symmetrically across both home/away orderings so it is
unbiased and consistent with the existing complementary `win_prob`:

```
pab = model.predict_proba(row(A,B))   # [A_home_win, draw, B_away_win]
pba = model.predict_proba(row(B,A))   # [B_home_win, draw, A_away_win]
p_a_reg = 0.5*(pab[0] + pba[2])        #  these three
p_draw  = 0.5*(pab[1] + pba[1])        #  sum to 1
p_b_reg = 0.5*(pab[2] + pba[0])
s_a     = shootout_prob(pen[A], pen[B], depth[A], depth[B])   # ~0.5 with no pen data
p_a_advance = p_a_reg + p_draw * s_a
```

`p_a_advance` is algebraically identical to the current
`build_win_prob`'s `0.5*(resolve_tie(pab) + (1 - resolve_tie(pba)))` — so the new
tab cannot disagree with the champion-odds tab. A test asserts this equality.
Decided ties short-circuit to certainty (winner → advance 1.0, draw 0).

## 4. The bracket walk

```
ROUND_OF_32 (16 ties) → predicted winners
  → ROUND_OF_16 (8)   → predicted winners
    → QUARTER (4)      → predicted winners
      → SEMI (2)       → predicted winners
        → FINAL (1)    → CHAMPION
```

`walk_bracket(teams, breakdown_fn)` pairs consecutive slots, calls
`breakdown_fn` per tie, advances each `winner`, and records each round's matches.
Returns `{"rounds": [{"round": name, "matches": [breakdown,...]}...], "champion": str}`.
The most-likely-path champion may differ from the highest champion-*probability*
team (different question); both views coexist in different tabs.

## 5. Components

- **`src/fifa2026/knockout/walk.py`** (new):
  - `match_breakdown(model, feature_builder, team_a, team_b, as_of_date, pen=None, depth=None, decided=None) -> dict`
    with keys `team_a, team_b, decided(bool), winner, p_a_reg, p_draw, p_b_reg, p_a_shootout, p_a_advance, p_b_advance`.
  - `build_win_prob(model, feature_builder, as_of_date, pen=None, depth=None, decided=None)` — MOVED here from `cli.py`; its inner `win_prob(a,b)` now returns `match_breakdown(...)["p_a_advance"]` (single source of truth).
  - `walk_bracket(teams, breakdown_fn) -> dict` and `ROUND_NAMES`.
- **`src/fifa2026/cli.py`** (modify): re-export `build_win_prob` from `walk` (`from fifa2026.knockout.walk import build_win_prob`) so existing imports keep working; drop the local definition and the now-unused `resolve_tie` import.
- **`src/fifa2026/pipeline.py`** (modify `run_predict`): build a `breakdown_fn = lambda a,b: match_breakdown(ensemble, fb, a, b, as_of_date, decided=decided)`, call `walk_bracket(teams, breakdown_fn)`, store it as `meta["predicted_path"]`. (The existing champion/round probs are unchanged.)
- **`app.py`** (modify): new **"Predicted path"** tab — per round, a table: Match · P(A win) · P(draw) · P(B win) · shootout A% · P(A adv) · → winner; the predicted champion highlighted. Graceful message if `predicted_path` absent.
- **`config/bracket_2026.yaml`** (modify): set `decided: []` (predict all 16 R32 fresh). Real results get added back here as they happen.

## 6. The update workflow (operational, not code)

As each real R32 (then R16, …) result is known: add `{winner, loser}` to
`bracket_2026.yaml`'s `decided` list, re-run `make predict`. The "Predicted path"
tab locks that tie at 100% and re-projects the remaining bracket from the new
reality; we tally model-predicted vs actual.

## 7. Testing / correctness

- `match_breakdown`: `p_a_reg + p_draw + p_b_reg ≈ 1`; `p_a_advance + p_b_advance ≈ 1`;
  `p_a_advance == 0.5*(resolve_tie(pab) + (1 - resolve_tie(pba)))` (consistency with
  the old win-prob, using fake models); a decided tie returns certainty.
- `build_win_prob` still importable from `fifa2026.cli` and returns the same value
  as `match_breakdown[...]["p_a_advance"]` (existing `test_cli_smoke` must pass).
- `walk_bracket`: an 8-team bracket yields rounds of size 4/2/1 matches and one
  champion; the champion is the team that wins its projected ties.
- `run_predict`: `prediction.json` `meta["predicted_path"]` has `rounds` (5 for a
  32-team bracket) and a `champion`; champion probs still sum to 1.
- Controller sanity: the real predicted bracket has favorites advancing, no absurd
  upsets implied as certainties.

## 8. Scope

- **In:** the match breakdown + bracket walk module, `run_predict` wiring, the new
  app tab, clearing the R32 `decided` list, the predict-vs-actual update workflow.
- **Out:** a real penalty/shootout model (no data — stays ~50/50); changing the
  probabilistic champion-odds/survival tabs; simulating uncertainty in the path
  (this is the single most-likely path, by design).

## 9. Risks

- **Most-likely-path vs champion-odds confusion** — the path champion can differ
  from the top champion-probability team. Mitigated by labeling the tab clearly
  ("single most-likely bracket") and keeping the probabilistic tabs distinct.
- **Refactor of `build_win_prob`** — moving it risks breaking imports; mitigated by
  re-export from `cli.py` and the behavior-equality test.
