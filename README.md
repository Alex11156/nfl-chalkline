# NFL Prediction Engine

Ask-anything NFL prediction system for fantasy football and betting analysis,
trained on 27 seasons (1999–2025) of nflverse play-by-play data.

## Launch

```bash
/opt/anaconda3/envs/nfl/bin/python -m streamlit run app/app.py
```

Optional: `export ANTHROPIC_API_KEY=...` before launching to enable the
Claude-powered free-form question mode (the local parser works without it).

## What it does

- **Game predictions** — win probability, projected margin/total/score for any
  matchup (LightGBM on opponent-adjusted EPA form, coaching, QB profiles,
  rest/travel/weather, Elo).
- **Player props** — P(over/under) for any line: passing/rushing/receiving
  yards, receptions, TDs, INTs (quantile-distribution models).
- **Fantasy** — weekly projections with floor/median/ceiling (PPR, half, standard),
  start/sit comparisons, top-20 boards.
- **Situational trends** — 20 years of history behind questions like "how do
  new head coaches do ATS?" (DuckDB over the full dataset).
- **Ask Anything** — chat interface; local pattern parser (free) or Claude
  tool-use mode (API key).

## Honest performance (walk-forward out-of-sample, 2010–2025)

| Metric | Engine (no market) | Engine (with market) | Benchmark |
|---|---|---|---|
| Straight-up winners | 65.4% | 66.1% | Vegas favorites 66.1% |
| Margin MAE | 10.29 | 10.18 | Vegas closing spread 10.05 |
| Against the spread | 50.9% | 51.4% | 52.4% = breakeven |
| Over/under | 51.9% | 52.2% | 52.4% = breakeven |
| Fantasy PPR MAE | 4.53 | — | naive rolling avg 4.69 |
| Passing-yards MAE | 61.6 (+10.1% vs naive) | — | naive 68.5 |

No model reliably beats closing lines. Use it for calibrated probabilities,
matchup magnitude, and floors/ceilings — not guaranteed edges.

## Website

The engine is also deployed as a static website ("Chalkline") with all model
outputs precomputed — every matchup, every player's projection and prop
distributions — so it runs entirely in the browser. To rebuild after a
retrain: `python scripts/export_web.py`, inline `web/data.json` into
`web/index.template.html` → `web/index.html`, and republish.

## Maintenance

```bash
# during the season: refresh data + retrain (~20 min)
/opt/anaconda3/envs/nfl/bin/python scripts/refresh_data.py
# re-run the question test suite
/opt/anaconda3/envs/nfl/bin/python scripts/test_questions.py
```

## Layout

- `nfl_engine/data` — nflverse download + DuckDB store (`data/nfl.duckdb`)
- `nfl_engine/features` — leak-free feature pipelines (game + player)
- `nfl_engine/models` — LightGBM models, walk-forward evaluation
- `nfl_engine/query` — serving layer, tool registry, NL parser, Claude agent
- `app/app.py` — Streamlit UI
- `reports/` — evaluation JSONs and logs
- `models_store/` — trained model artifacts
