# research/ — one-off experiments

Nothing here runs in the weekly pipeline. Each script answered one question;
the verdict and its numbers are recorded in [MODEL.md](../MODEL.md) so the
question is not re-litigated. Outputs go to `reports/research/`.

Run any of them from the repo root: `python research/<script>.py`.

## Which kind of model?

| Script | Question | Verdict |
|---|---|---|
| `experiment_models.py`, `experiment_player.py` | Does another *family* of learner beat LightGBM? (XGBoost, CatBoost, forests, linear, MLP, ensembles) | No — LightGBM wins on games and players |
| `model_types_game.py` | Does a different *kind* of model fit games better? (additive models, Kalman-filter ratings, a direct classifier, a compact linear model, blends; and a margin distribution that knows about 3 and 7) | **Yes, in part.** LightGBM averaged with an additive model (EBM) is better for the no-market margin and total; everything else is rejected. The key-number margin distribution is adopted for the betting card |
| `model_types_props.py` | Does a different kind of model produce a better *range* for player props? (11 ways of building the distribution, then blends) | The five boosted quantile models stay. Adopted stat by stat: separate upper/lower tails for the lopsided stats, and an average with a ridge model for pass attempts and completions (passing yards narrowly missed) |

## Settings and inputs

| Script | Question | Verdict |
|---|---|---|
| `tune_game.py` | Best LightGBM hyperparameters for the game models? | Heavily regularised depth-3 trees; totals reuse the margin params |
| `tune_player.py` | Best hyperparameters for the player models? | Deeper trees (depth 6); shared across stats |
| `experiment_game2.py` | Feature selection, recency halflife, two-score targets? | All inside noise; nothing adopted |
| `experiment_player_recency.py` | Recency-weight the player models like the game models? | No — it hurts |
| `prop_model_search.py` | Do new inputs or other hyperparameters improve the prop models? Stages: `features`, `starter`, `depth_fix`, `params`, `params_heavy` | Game-script, volume and starting-QB inputs help quarterback stats; only the rushing-yards quantiles earn heavier settings |
| `feature_importance.py` | Which inputs matter most? (permutation importance on held-out seasons, plus the additive model's own curves) | Team quality and the quarterback for games; role and recent production for players |

## Is there a betting edge?

| Script | Question | Verdict |
|---|---|---|
| `edge_analysis.py`, `moneyline_history.py`, `reliability.py` | Do the game picks beat the closing line? Does a bigger disagreement win more? Is the model-market blend better than the market on held-out seasons? | No edge; edges do not concentrate; the blend is level with the market |
| `edge_search.py` | Can *any* strategy beat closing lines? (120 strategies, market-anchored models, injuries; frozen holdout) | No — every dev-period winner fell to break-even |
| `prop_edge_study.py` | Which prop types does the model know more about than a player's average? | QB volume props (+8–10 pts); others ~0–2 |
| `significance_test.py` | Paired bootstrap helper for comparing two game models | Tool |

**The bar for adopting anything:** write the test down first; the change must
clearly win on the seasons used to choose it *and* hold on seasons that were
not. A backtest that picks its own winners will always find some.
`model_types_props.py` (two fixed periods, then a season-by-season check of
any winner) is the current template.
