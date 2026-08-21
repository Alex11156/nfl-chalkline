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

## Public website (Chalkline)

The engine deploys as a static site — all model outputs precomputed, all
probability math client-side — served from `docs/` via GitHub Pages.

One-time setup:
```bash
/opt/anaconda3/envs/nfl/bin/gh auth login     # browser login, ~1 min
bash scripts/deploy_github.sh                  # creates repo + enables Pages
# → https://<your-username>.github.io/nfl-chalkline/
```

## In-season weekly refresh

Run after each week's games (Tuesday morning is ideal — stats are final):

```bash
# pulls new data, rebuilds features, retrains + recalibrates all models,
# rebuilds the site, and pushes it live (~25 min)
/opt/anaconda3/envs/nfl/bin/python scripts/refresh_data.py --publish
```

Season boundaries are automatic: the current season is derived from the
date (`nfl_engine/config.py`), new-season play-by-play starts downloading
as soon as games are played, completed seasons stay cached, and the
walk-forward evaluation extends itself. Recency weighting (4-season
halflife) means new games immediately carry the most training influence.

Occasionally (once or twice a season) re-run the hyperparameter searches —
`scripts/tune_game.py` and `scripts/tune_player.py` — and update the
params in the model files if validation clearly improves.

## Layout

- `nfl_engine/data` — nflverse download + DuckDB store (`data/nfl.duckdb`)
- `nfl_engine/features` — leak-free feature pipelines (game + player)
- `nfl_engine/models` — LightGBM models, walk-forward evaluation
- `nfl_engine/query` — serving layer, tool registry, NL parser, Claude agent
- `app/app.py` — Streamlit UI
- `reports/` — evaluation JSONs and logs
- `models_store/` — trained model artifacts
