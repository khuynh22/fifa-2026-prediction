# fifa-2026-prediction

Open-source machine-learning engine that predicts the **2026 FIFA World Cup
champion** by modeling each knockout match (Round of 32 → Final), with proper
extra-time / penalty-shootout handling, and benchmarked against the betting
market.

> Status: **end-to-end pipeline complete.** The feature builders, hybrid
> ensemble, knockout bracket DP, CLI orchestration, and Streamlit dashboard
> are all implemented and tested.
> Read the design first: [`docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md`](docs/superpowers/specs/2026-06-28-fifa-2026-wc-prediction-design.md)

## What it does

- Builds **point-in-time** features for international matches (no leakage).
- Predicts each match with a **hybrid ensemble** — a Dixon-Coles/Poisson goals
  model blended with a LightGBM classifier.
- Resolves knockout ties through regulation → extra time → a **shootout
  resolver**, then walks the actual bracket to a champion pick with per-round
  survival probabilities.
- Is **benchmarked against bookmaker odds** (odds are never a model input).
- Adjusts the forecast for **player availability**: key absences listed in
  `data/reference/injuries_2026.yaml` apply an effective-Elo penalty (a
  transparent, config-tunable heuristic — `config/default.yaml: availability` —
  not a trained signal). Live-API enrichment is optional future work (the free
  API tier does not expose the 2026 season or national-team injuries).
- Projects a **most-likely-path bracket**: the predicted winner of each tie
  advances to a predicted Final + champion, with per-match regulation
  win/draw/loss and (~50/50) shootout odds — shown in the "Predicted path" tab.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env          # optional: add FOOTBALL_API_KEY for squad data
make all                      # data → train → evaluate → predict
make app                      # launch the Streamlit dashboard
```

`make all` ingests free datasets, fits the hybrid ensemble on internationals
2010→present, runs a temporal backtest, and writes a dated champion-probability
snapshot to `reports/prediction.json`.  The forecast reflects the live
tournament state as of the run date — re-run `make predict` any time to update
it.

## Pipeline steps

```bash
make data       # ingest free datasets + football API (cached)
make train      # fit the hybrid ensemble on internationals 2010→now
make evaluate   # temporal backtest + market benchmark
make predict    # forward-predict the Round-of-32 bracket
make app        # open the Streamlit dashboard at http://localhost:8501
make test       # run the full test suite
```

## License

[MIT](LICENSE)
