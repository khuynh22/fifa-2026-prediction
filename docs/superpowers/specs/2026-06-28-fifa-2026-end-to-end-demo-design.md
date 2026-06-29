# FIFA 2026 — End-to-End Real Prediction + Demo — Design

**Date:** 2026-06-28
**Status:** Approved design (pending final user review)
**Builds on:** `2026-06-28-fifa-2026-wc-prediction-design.md` (the tested library)

## 1. Goal

Turn the tested `fifa2026` library into a **real, runnable predictor with an
interactive demo**: ingest actual data, wire the train/evaluate/predict
pipeline end-to-end on real internationals, predict the **actual 2026 World Cup
champion from the current Round-of-32 state**, and present the result through an
interactive Streamlit app with charts and data a user can click through live.

## 2. Decisions (from brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Data sourcing | **Free data + football API.** Free is the reproducible backbone; API enriches the 2026 field. |
| 2 | API key | User HAS an API-Football key (in local `.env`). API path must **degrade gracefully** to team-strength-only if key/endpoint is absent. |
| 3 | Demo format | **Interactive Streamlit app** (Plotly charts). |
| 4 | 2026 bracket | **Real R32 bracket** fetched from Wikipedia, user-confirmed (see §5). |
| 5 | Decided results | **Pin** the four already-played R32 results so the forecast is "from the current state." |

## 3. Architecture (five layers + app)

```
[ingest] martj42 results.csv (download+cache) ─┐
         confederations.csv, venues_2026.csv   ├─► data/raw, data/processed
         API-Football squads (cache, optional) ─┘
            │
[train]  load 2010→now → Elo/Form/Context → build_training_matrix(squad_agg=None)
            → fit Poisson + LightGBM → tune ensemble → joblib → models/
            │
[evaluate] temporal backtest (log-loss/Brier/accuracy + calibration)
            + model-vs-market (fetched winner odds → implied probs) → reports/
            │
[predict] load models → host-aware complementary win_prob → pin decided R32 ties
            → bracket DP → champion probs + per-round survival → reports/prediction.json
            │
[app]    streamlit run app.py → champion odds, bracket, survival, team explorer, calibration
```

Each layer is a thin orchestration of already-tested units plus a small amount
of new glue (downloaders, persistence, the app). New code is kept in focused
modules; existing tested modules are not rewritten.

## 4. Data sources (concrete)

- **International results (training):** `martj42/international_results` —
  `results.csv` (1872→present, free), fetched from the raw GitHub URL into
  `data/raw/results.csv` and cached. This is the canonical `matches` source the
  existing `ingest/matches.load_matches` already parses.
- **Confederations:** a maintained `data/reference/confederations.csv` (team →
  confederation) covering at least the 2026 field + common opponents.
- **Venues:** `data/reference/venues_2026.csv` — the 2026 host cities with
  `lat,lon,altitude_m` (incl. Mexico City altitude, Denver, etc.).
- **Hosts:** `["United States", "Mexico", "Canada"]` (already in `default.yaml`).
- **Squads/players (enrichment):** API-Football, keyed by `FOOTBALL_API_KEY`
  from `.env`, fetched per-team for the 32 R32 teams and cached on disk. Used
  for the 2026 prediction only (prediction-time `squad_agg`, never in training —
  enforced by the existing leakage guard).
- **Market odds (benchmark):** current bookmaker "World Cup winner" odds,
  fetched once, converted to implied champion probabilities (de-vigged) and
  stored in `data/reference/market_odds_2026.csv` for the model-vs-market view.

If the API key is absent or a fetch fails, the pipeline logs a clear notice and
continues with team-strength features only — the demo never blocks.

## 5. Real 2026 bracket + result pinning

`config/bracket_2026.yaml` is rewritten with the **actual** R32 field in
bracket-slot order so the existing recursive `bracket._solve` reproduces the real
adjacency. Round-of-32 ties (Wikipedia, match numbers):

```
M73 Canada vs South Africa      M81 USA vs Bosnia & Herzegovina
M74 Germany vs Paraguay         M82 Belgium vs Senegal
M75 Netherlands vs Morocco      M83 Portugal vs Croatia
M76 Brazil vs Japan             M84 Spain vs Austria
M77 France vs Sweden            M85 Switzerland vs Algeria
M78 Ivory Coast vs Norway       M86 Argentina vs Cape Verde
M79 Mexico vs Ecuador           M87 Colombia vs Ghana
M80 England vs DR Congo         M88 Australia vs Egypt
```

Round-of-16 adjacency (drives the slot ordering): M89=W74·W77, M90=W73·W75,
M91=W76·W78, M92=W79·W80, M93=W83·W84, M94=W81·W82, M95=W86·W88, M96=W85·W87.
The complete QF/SF/Final adjacency will be fetched from the same Wikipedia source
during implementation to fix the full leaf ordering.

**Ordering validation:** a test reconstructs the implied R16 pairings from the
yaml slot order and asserts they equal the real pairings above — so an ordering
mistake fails loudly rather than silently mis-predicting.

**Decided results (pinned):** the config carries a `decided:` list of
`{winner, loser}` for already-played ties (Canada>South Africa, Germany>Paraguay,
Netherlands>Morocco, France>Sweden). The `win_prob` wrapper returns `1.0` when
the first team is a pinned winner and `0.0` when it is the pinned loser, before
any model call — so decided ties are certainties and the forecast is genuinely
"from the current state." New ad-hoc results are added by editing this list.

## 6. New / modified components

- **`ingest/download.py`** (new): `fetch_results_csv(url, dest, cache) -> Path`
  and small helpers; pure I/O with caching, tested against a tiny local stub.
- **`ingest/odds.py`** (new): `parse_winner_odds(rows) -> dict[str,float]` and
  `implied_champion_probs(odds) -> dict[str,float]` (de-vig across the field).
- **`persistence.py`** (new): `save_models(dir, ensemble, meta)` /
  `load_models(dir)` via joblib.
- **`pipeline.py`** (new): the orchestration functions the CLI calls —
  `run_train(cfg)`, `run_evaluate(cfg)`, `run_predict(cfg) -> PredictionResult`.
  `PredictionResult` holds champion probs, per-round survival, per-tie win
  probs, and metadata; serializable to `reports/prediction.json`.
- **`knockout/bracket.py`** (modify): add `round_probabilities(teams, win_prob)`
  for per-round survival (reach R16/QF/SF/Final/win), reusing the DP.
- **`cli.py`** (modify): flesh out `data`/`train`/`evaluate`/`predict` bodies to
  call `pipeline`/`download`; keep argparse surface.
- **`viz.py`** (new): pure chart builders returning Plotly figures from a
  `PredictionResult` (champion bar, bracket, survival heatmap, market compare,
  calibration). Separated from the app so charts are unit-testable.
- **`app.py`** (new, repo root): the Streamlit app — loads
  `reports/prediction.json` (+ backtest metrics) and renders the five sections
  via `viz`. Thin; no modeling logic.

## 7. Streamlit app sections

1. **🏆 Champion odds** — sorted bar chart of P(win cup) per team; toggle to
   overlay market-implied probabilities.
2. **🗺️ Bracket** — the R32→Final tree with each tie's win %, decided ties
   marked as locked.
3. **📊 Per-round survival** — heatmap/table: P(reach R16 / QF / SF / Final / win).
4. **🔎 Team explorer** — pick a team → its Elo, recent form, squad features
   (if API enriched), and path-to-final probabilities.
5. **📈 Calibration** — reliability curve + headline metrics (log-loss, Brier,
   accuracy) from the temporal backtest, with the model-vs-market summary.

## 8. Reproducibility

- `make data && make train && make evaluate && make predict && make app`
  (and `make all`). `.env.example` documents `FOOTBALL_API_KEY`. Deterministic
  seeds. Cached raw data + API responses so reruns are fast and offline.
- New deps: `streamlit`, `plotly`, `joblib` added to `pyproject.toml`.

## 9. Testing

- `download`/`odds`/`persistence`/`viz`/`pipeline` units get focused tests
  (I/O mocked against local stubs; viz tested for figure structure not pixels).
- Bracket ordering-validation test (§5).
- `run_predict` integration test on a tiny synthetic dataset asserting champion
  probabilities sum to 1 and pinned ties are certainties.
- The app itself is smoke-checked (imports, builds figures from a fixture
  `prediction.json`); not pixel-tested.

## 10. Scope

- **In:** real data ingestion, end-to-end train/evaluate/predict on real
  internationals, real 2026 bracket with pinned results, persisted models,
  Streamlit demo with five interactive views, model-vs-market.
- **Out (later):** auto-refreshing live results during the tournament,
  deployment/hosting of the app, deep historical player features, probability
  calibration layer (still a documented follow-up from the prior spec).

## 11. Risks

- **API-Football schema/limits** — mitigated by caching + graceful fallback.
- **Market-odds fetch fragility** — if unavailable, the market overlay is hidden
  and the rest of the demo is unaffected.
- **Bracket ordering correctness** — mitigated by the ordering-validation test.
- **Live results drift** — the tournament is in progress; `decided:` and the
  bracket are a maintained snapshot, clearly dated in the app.
