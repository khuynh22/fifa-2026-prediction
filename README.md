# fifa-2026-prediction

Open-source machine-learning engine that predicts the **2026 FIFA World Cup
champion** by modeling each knockout match (Round of 32 → Final), with proper
extra-time / penalty-shootout handling, and benchmarked against the betting
market.

> Status: **core library complete and tested (35 tests).** The pipeline
> components — feature builders, the hybrid model, knockout/shootout
> resolution, and the exact bracket champion-probability DP — are implemented
> and unit-tested. The CLI orchestration and real-data wiring are follow-on
> work (see "What's left" below).
> Read the design first: [`docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md`](docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md)

## What it does

- Builds **point-in-time** features for international matches (no leakage).
- Predicts each match with a **hybrid ensemble** — a Dixon-Coles/Poisson goals
  model blended with a LightGBM classifier.
- Resolves knockout ties through regulation → extra time → a **shootout
  resolver**, then walks the actual bracket to a champion pick with per-round
  survival probabilities.
- Is **benchmarked against bookmaker odds** (odds are never a model input).

## Reproduce (intended interface)

```bash
make data       # ingest free datasets + football API (cached)
make train      # fit the hybrid ensemble on internationals 2010→now
make evaluate   # temporal backtest + market benchmark
make predict    # forward-predict the Round-of-32 bracket
```

> **Heads up:** these CLI commands are currently **scaffolded stubs** — the
> units they will orchestrate are built and tested, but the command bodies
> are not wired yet.

## What's left for a real 2026 prediction

- Wire the `data` / `train` / `evaluate` / `predict` command bodies to the
  (already-tested) pipeline units, plus model save/load.
- Supply real data: international `results.csv`, confederations, venues, and a
  football-API key for squads.
- Fill `config/bracket_2026.yaml` with the actual Round-of-32 draw (it ships
  with placeholder team names).
- Model the co-host (USA/Mexico/Canada) home advantage and feed each tie's
  real venue into the win-probability function.
- Make the pairwise win-probability complementary so the bracket
  champion-probability table sums to 1.
- Keep Tier-B squad features out of historical training (or make them
  point-in-time) to preserve the no-leakage invariant.

## License

[MIT](LICENSE)
