# Chalkline — NFL Prediction Engine

Predicts NFL games (winner, margin, total) and player stat lines (fantasy
points and props) from 27 seasons of public play-by-play data, answers
questions in plain English, publishes a static website, and grades itself
against real results and real sportsbook lines.

**New here? Read these in order:**
1. This file — what is where and how to run it.
2. [reports/Chalkline_Codebase_Guide.pdf](reports/Chalkline_Codebase_Guide.pdf) —
   a walkthrough of every part and *why* it is built that way: which kinds of
   model were tried, which inputs matter, and how the live system is checked
   against the tested one.
3. [MODEL.md](MODEL.md) — the full technical reference: every design decision,
   measured number and rejected idea.

## What it can and cannot do

Measured on games never seen in training (walk-forward, 2010 – 2026 week 10):

| Measure | Engine | Benchmark |
|---|---|---|
| Winners picked | 65.7% (66.2% given Vegas lines) | Vegas favourite 66.2% · Elo 64.3% |
| Margin error | 10.25 pts (10.17 given lines) | Vegas spread 10.04 |
| Against the spread | 50.8% | 52.4% needed to profit |
| Fantasy PPR error | 4.51 | player's own average 4.68 |
| Passing yards / attempts / completions (games the QB started) | +9.2% / +11.4% / +10.6% better than the player's average | — |
| Rushing / receiving yards | +3.1% / +1.7% better | — |

- **Trust it for:** calibrated win probabilities, expected scores, and player
  projections with realistic floors and ceilings.
- **Do not trust it for:** beating closing game lines (120 strategies tested;
  none survived fresh data) or touchdown props.
- **Open question:** quarterback volume props vs real sportsbooks — being
  graded weekly by `scripts/props.py`; nothing is proven yet.

## Where everything is

```
nfl_engine/            the engine (a Python package) — everything else calls it
  config.py              paths, team names, current season
  data/store.py          raw files -> DuckDB analytical tables
  features/              model inputs, built without peeking at the future
    elo.py                 team strength rating (baseline + input)
    game_features.py       one row per game: form, QB, coach, rest, weather
    player_features.py     one row per player-game: role, production, matchup
    stadiums.py            travel distance and time zones
  models/
    folds.py               the walk-forward backtest (with a cache of finished seasons)
    game_models.py         margin + total: LightGBM, averaged with an additive model
    player_models.py       14 player stats as ranges (quantiles / Poisson),
                           and the math that turns them into projections and odds
  calibration.py         honest win %s, the real pattern of final margins,
                         how much to trust model vs market
  query/                 predicting FUTURE games and answering questions
    serving.py             builds the inputs for upcoming games
    tools.py               the list of things the engine can be asked
    parser.py              free, offline plain-English question matcher
    claude_agent.py        optional Claude-powered answers (needs an API key)
  picks.py               weekly betting card: model vs market, EV, stake
  odds.py                odds arithmetic shared by everything that prices a bet

scripts/               the things you run
  refresh_data.py        THE weekly command — runs the whole pipeline
  download_data.py       pull nflverse data
  season_eval.py         grade this season so far
  build_site.py          export model outputs, build docs/index.html
  props.py               sportsbook prop tracker: `snapshot` and `grade`
  report_*.py            rebuild the PDF reports
  deploy_github.sh       one-time GitHub Pages setup

research/              one-off experiments (not part of the pipeline);
                       research/README.md lists each question and verdict
tests/
  test_questions.py      40 plain-English questions that must answer correctly
  test_parity.py         the live system must feed the models exactly what
                         they were trained on
app/app.py             local Streamlit app
web/  docs/            website template + data  /  the published site
reports/               PDFs, backtest results, saved sportsbook lines
  research/ logs/ figures/   experiment outputs, run logs, report charts
  prop_snapshots/ line_snapshots/   lines saved before kickoff, for grading
data/  models_store/   downloaded data and trained models (not in git)
Makefile               every routine command by name
```

## Running it

Run from this folder. `make` on its own lists the commands; each is one line of
Python, using the project's environment (`/opt/anaconda3/envs/nfl/bin/python`,
shown as `python` below) when it exists.

| When | Command | What it does |
|---|---|---|
| Tuesday | `make publish` · `python scripts/refresh_data.py --publish` | New data, backtest, retrain, grade last week, rebuild and push the site. `make refresh` keeps it local. |
| Saturday | `make props` · `python scripts/props.py snapshot` | Save sportsbook prop lines + the model's probabilities (free Odds API key in `~/.config/chalkline/odds_api_key`) |
| Any time | `make grade` · `python scripts/props.py grade` | The running prop record vs real books |
| Any time | `make app` · `python -m streamlit run app/app.py` | Local app (set `ANTHROPIC_API_KEY` for Claude-powered answers) |
| After code changes | `make test` | 40-question check, and train/serve parity for the last finished week |
| Any time | `make models` · `make site` | Retrain on the data already downloaded · rebuild the site |
| Any time | `make guide` | Rebuild the codebase guide PDF from the saved results |

First-time setup: `pip install -r requirements.txt`; for the public site,
`gh auth login`, then `bash scripts/deploy_github.sh` (serves `docs/` via
GitHub Pages).

## Changing the model

Any change to inputs, model type or settings must:

1. be tested by a rule written down first — win walk-forward on the seasons
   used to choose it **and** hold on later seasons that were not
   (`research/model_types_props.py` and `research/prop_model_search.py` are
   the templates);
2. be wired into training and serving together, with `make test` still
   passing (`tests/test_parity.py` fails if the live system would feed a model
   anything different from what it was trained on);
3. be recorded in [MODEL.md](MODEL.md) with its numbers — including when it
   fails.
