# fifa-2026-prediction

Open-source machine-learning engine that predicts the **2026 FIFA World Cup
champion** by modeling each knockout match (Round of 32 → Final), with proper
extra-time / penalty-shootout handling, and benchmarked against the betting
market.

> Status: **design complete, implementation starting.**
> Read the design first: [`docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md`](docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md)

## What it does (planned)

- Builds **point-in-time** features for international matches (no leakage).
- Predicts each match with a **hybrid ensemble** — a Dixon-Coles/Poisson goals
  model blended with a LightGBM classifier.
- Resolves knockout ties through regulation → extra time → a **shootout
  resolver**, then walks the actual bracket to a champion pick with per-round
  survival probabilities.
- Is **benchmarked against bookmaker odds** (odds are never a model input).

## Reproduce (planned)

```bash
make data       # ingest free datasets + football API (cached)
make train      # fit the hybrid ensemble on internationals 2010→now
make evaluate   # temporal backtest + market benchmark
make predict    # forward-predict the Round-of-32 bracket
```

## License

[MIT](LICENSE)
