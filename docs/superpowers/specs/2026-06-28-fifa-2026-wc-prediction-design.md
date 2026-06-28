# FIFA 2026 World Cup Champion Prediction — Design

**Date:** 2026-06-28
**Status:** Approved design (pending final user review)
**Author:** Tim Huynh (with Claude)
**License:** MIT (open source)

## 1. Goal

Build an open-source, reproducible machine-learning engine that predicts the
**2026 FIFA World Cup champion** by modeling each knockout match from the
**Round of 32 through the Final**, with correct extra-time / penalty-shootout
handling, and benchmarked against the betting market.

The tournament is already past the group stage; the Round of 32 bracket is set
and group-stage results are known. The system therefore predicts forward from a
**known bracket state** rather than simulating the whole tournament from scratch.

## 2. Non-goals (v1)

- No full Monte Carlo trophy-odds table in v1 (the architecture is built *for*
  it — see §11 — but the headline deliverable is the match engine + a forward
  bracket prediction).
- No live/in-play prediction during matches.
- No web dashboard in v1 (future work).
- No use of bookmaker odds as a model **input** (used only as a **benchmark**).

## 3. Key decisions (decision log)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Prediction approach | Match-outcome engine applied across the knockout bracket | The match model is the core of every approach; chaining/simulating on top is cheap and additive. |
| 2 | Model family | **Hybrid ensemble**: Dixon-Coles/Poisson goals model **+** LightGBM classifier, probabilities blended | Poisson is football-native (scorelines, feeds shootout + future sim); LightGBM exploits rich tabular features. Blend beats either alone. |
| 3 | Knockout resolution | Regulation outcome → extra time → **shootout resolver** sub-model | Pure knockout has no draws; ignoring shootouts systematically over-rates favorites. |
| 4 | Data strategy | Free public datasets **+** a football API, with a caching layer | Free datasets give reproducible team-level history; API adds current squad/player richness. |
| 5 | Betting odds | **Excluded** from features; used as a **benchmark** | Keeps the prediction genuinely "ours"; the better open-source story is "can we match/beat the market?" |
| 6 | Player-data depth | **Deep per-player stats** aggregated to team level, where coverage exists | Maximizes signal for the 2026 prediction; partial in deep history (see §6 availability tiers). |
| 7 | Training window | **Modern only: 2010 → now**, all internationals (~16k matches) | Most relevant to today's game; still a healthy sample because it includes all internationals, not just World Cups. |
| 8 | Validation | **Temporal** (train past → test later tournaments), never random K-fold | Sports prediction must respect time ordering to avoid leakage. |
| 9 | User level | Comfortable with Python + ML | Lean explanations; focus on football-specific modeling and feature engineering. |

## 4. Architecture / data flow

```
[ingest]   free datasets + football API  ──►  data/raw/   (cached, point-in-time)
              │
[features] build per-match A-vs-B differential features, as-of match date
              │                              ──►  data/processed/features.parquet
              │
[train]    fit hybrid ensemble on all internationals 2010→now (optional recency weighting)
              │                              ──►  models/
              │
[evaluate] temporal validation + benchmark vs bookmaker closing odds
              │                              ──►  reports/
              │
[predict]  load actual R32 bracket → resolve each tie (reg→ET→shootout)
              │                              ──►  champion + per-round win probs
```

Each stage is an independently testable unit with a clear input/output contract,
wired together by a thin CLI / `make` targets.

## 5. Components & contracts

- **`ingest/`** — pulls raw data from each source into `data/raw/` with on-disk
  caching keyed by source + date range. Contract: idempotent; never refetches
  cached data; respects API rate limits. Output: tidy raw tables (matches,
  rankings, squads, players, venues).
- **`features/`** — given raw tables + a match (teams, date, venue, stage),
  produces a single feature row computed **strictly as-of before the match**.
  Contract: pure function of past data only (point-in-time correctness is the
  central invariant, enforced by tests).
- **`models/`** — `PoissonModel` (predict λ_A, λ_B → scoreline distribution →
  P(W/D/L)) and `BoostedModel` (LightGBM multiclass W/D/L), plus an `Ensemble`
  that blends their probabilities. Common interface: `predict_proba(features) →
  {p_home_win, p_draw, p_away_win}`.
- **`knockout/`** — `resolve_tie(team_a, team_b, features)` → single winner,
  using `predict_proba` for regulation, an extra-time adjustment, and a
  `shootout_resolver` for ties.
- **`bracket/`** — given the R32 bracket and the tie resolver, produces the
  forward prediction (most-likely path + per-round survival probabilities).
- **`evaluate/`** — temporal backtests, calibration, and the market benchmark.

## 6. Feature catalog

All features enter the model as **A-vs-B differentials** (or ratios) and are
computed **point-in-time** (only data available before the match date).

### Tier A — historical, full coverage (powers the trained model 2010→now)
- **Elo rating difference** (from eloratings.net or computed in-repo).
- **FIFA ranking / ranking-points difference.**
- **Recent form**, opponent-strength-adjusted: points-per-game / win rate over
  last 5, 10, 20 matches.
- **Attack / defense strength**: goals scored & conceded per game (recent +
  long-run) — also the natural inputs to the Poisson model.
- **Home / host / neutral**: 2026 has three co-hosts (USA, Mexico, Canada);
  host matches get a boost, everyone else is effectively neutral.
- **Rest & congestion**: days since last match; fixture density.
- **Confederation** (UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC).
- **Travel & altitude**: distance to venue; altitude (e.g. Mexico City ~2,240 m).
- **Head-to-head** history between the two teams.
- **Squad age / tournament experience**: average age, total caps, prior
  knockout experience, manager experience.

### Tier B — partial history, current-strong (used where coverage exists)
- **Squad market value** (Transfermarkt-style) — good historical coverage.
- **Squad quality depth**: count of players in top-5 leagues; total UCL/UEL
  minutes; number of "elite" players.
- **Key-player availability**: injuries / suspensions to top scorer / key
  creator.
- **xG / xGA** (team level) where the API provides it.
- **Deep per-player aggregates**: per-player xG, xA, minutes, club continental
  involvement, aggregated to team-level features. Requires the caching layer.

### Tier B availability strategy
Tier B features are **only valid where point-in-time coverage exists** (strong
for recent years and for the 2026 squads, sparse deep in the past). Handling:
1. Compute Tier B where available; **impute / fall back** otherwise (e.g.
   missing-indicator + median, or back off to Tier A proxies like market value).
2. Report Tier B **feature coverage by year** so we know what the model is
   actually learning from.
3. Validate Tier B contribution on the **recent window** where coverage is high.
   Expectation: Tier B contributes most to the **2026 prediction itself** and
   partially to deep historical training — this is acknowledged and acceptable.

### Excluded by decision
- **Bookmaker / market odds** — not a feature; reserved for benchmarking (§9).

## 7. Knockout resolution (format-specific)

Every tie must resolve to one winner:

```
P(win in regulation)  ──►  if not a draw: done
                            if draw: extra-time adjustment
                                     if still level: shootout_resolver
```

- **Extra time**: small continuation of the regulation edge (fitness/depth
  slightly favor the deeper squad).
- **Shootout resolver**: a lightweight sub-model near 50/50, nudged by team
  penalty record / historical shootout performance / a mild strength term.
  Trained on historical shootouts. This prevents the favorite-overrating bias
  that pure regulation models exhibit in knockouts.

## 8. Model & training

- **Target**: match result. Poisson side predicts goals (λ_A, λ_B) → scoreline
  distribution → W/D/L; LightGBM side predicts W/D/L directly. Ensemble blends
  the two probability vectors (blend weight tuned on validation).
- **Training data**: all FIFA-recognized internationals, **2010 → now**
  (friendlies, qualifiers, continental cups, World Cups), labeled point-in-time.
- **Recency weighting**: optional sample weights decaying with age inside the
  window (most-recent matches weighted highest).
- **Stage context**: a "knockout / competitive / friendly" feature lets the
  model learn that knockouts and friendlies behave differently.
- **Calibration**: probabilities calibrated (e.g. isotonic / Platt) and checked
  with reliability curves — calibrated probabilities matter more than accuracy
  for a champion-odds story.

## 9. Evaluation

- **Temporal validation**: train on earlier matches; test on **later
  tournaments** (e.g. 2018 & 2022 World Cups, 2021/2024 continental cups).
  Never random K-fold.
- **Metrics**: log-loss and Brier score (probability quality) + accuracy +
  calibration curves.
- **Market benchmark**: compare model probabilities to **bookmaker closing
  odds** — agreement rate, log-loss vs the market, and a highlight of matches
  where the model disagreed and was right ("beat Vegas on these upsets").

## 10. Tech stack & repo layout

- **Language**: Python 3.11+.
- **Libraries**: `pandas`/`polars`, `numpy`, `scikit-learn`, `lightgbm`,
  `statsmodels` (Poisson/Dixon-Coles GLM), `requests` + on-disk cache,
  `pytest`, `pyyaml`.
- **Reproducibility**: `make data && make train && make evaluate && make
  predict`; config in `config/*.yaml`; deterministic seeds; API key via env var
  (`.env.example` provided), with cached fixtures so the repo runs without a key
  for already-fetched data.

```
fifa-2026-prediction/
├── README.md                  # story, results, how-to-reproduce
├── LICENSE                    # MIT
├── Makefile
├── pyproject.toml
├── config/                    # data sources, feature flags, model params
├── data/{raw,processed}/      # cached (raw gitignored; small fixtures kept)
├── src/fifa2026/
│   ├── ingest/                # datasets + API, caching
│   ├── features/              # point-in-time feature builders
│   ├── models/                # poisson, boosted, ensemble
│   ├── knockout/              # tie + shootout resolution
│   ├── bracket/               # forward bracket prediction
│   └── evaluate/              # backtests + market benchmark
├── notebooks/                 # EDA, results, calibration plots
├── tests/                     # incl. point-in-time leakage tests
└── docs/superpowers/specs/    # this design doc
```

## 11. Scope

- **v1 (this spec):** ingest → point-in-time features → hybrid ensemble →
  temporal validation + market benchmark → forward prediction of the R32
  bracket → champion pick with per-round survival probabilities.
- **Designed-for, built later:**
  - **Monte Carlo wrapper** (~30 lines) turning match probabilities into full
    "Brazil 18% / France 16% …" trophy-odds across thousands of bracket
    simulations.
  - Small web dashboard.
  - "Odds-aware" model variant for an ablation study.

## 12. Risks & open questions

- **API coverage / rate limits** for deep per-player international data — biggest
  practical risk; mitigated by the caching layer and Tier B fallback strategy.
- **Data redistribution / ToS**: keep raw API data out of the repo; ship only
  derived features + small public fixtures.
- **Small knockout sample** for the shootout resolver — keep it deliberately
  simple to avoid overfitting.
- **2026 bracket data entry**: the current bracket/results may need a small
  manual/maintained file until the API exposes them reliably.
```
