"""The weekly betting card.

For every scheduled game with posted odds, sets the engine's probability next
to the market's and scores each available bet by expected value, Sharpe ratio
and Kelly stake.

Three rules keep the card honest:

  * Bets are priced on a blend of model and market, not on the raw model. On
    held-out seasons the raw model's probability is worse than the market's
    on every bet type, and pricing on it inflated card EVs 5-10x (the 2026
    card promised a 60% hit rate and delivered 46%). The model's share of the
    blend is fitted walk-forward (calibration.fit_all): about a tenth.
  * Spreads and totals are priced on whole-number scores, where 3 and 7 are
    far more common than a bell curve says, so a line on a key number carries
    its real chance of a push (a refunded bet).
  * Every pick carries the backtested record of picks like it (the same size
    of disagreement with the line), recomputed at every retrain.
"""
import pandas as pd

from nfl_engine.calibration import (blend_with_market, cover_probs, key_factors,
                                    market_weights, pick_history, total_probs)
from nfl_engine.config import RAW, norm_team
from nfl_engine.io_utils import read_parquet
from nfl_engine.odds import american_to_decimal, bet_metrics, fair_prob
from nfl_engine.query.serving import ENGINE


def price(p_side: float, p_push: float, odds: float, other_odds: float, k: float) -> dict:
    """One side of a two-way bet. `p_side` is the model's chance that this
    side wins given the bet is decided (no push); the market's de-vigged price
    is a probability of the same kind, so the two are blended there."""
    fair = float(fair_prob(odds, other_odds))
    p_blend = float(blend_with_market(p_side, fair, k))
    m = bet_metrics(p_blend * (1 - p_push), american_to_decimal(odds), p_push)
    return m | dict(odds=float(odds), p_model=round(p_side, 4), p_blend=round(p_blend, 4),
                    p_market=round(fair, 4), p_push=round(p_push, 4), edge_prob=round(p_blend - fair, 4))


def build_picks(schedule: pd.DataFrame) -> list[dict]:
    """One row per available bet across the given games, best Sharpe first."""
    meta = ENGINE.game_models["pure"]["meta"]
    kw, factors = market_weights(), key_factors()
    picks = []

    for g in schedule.itertuples():
        home, away = norm_team(g.home_team), norm_team(g.away_team)
        try:
            # the scheduled game's real context: rest, venue, weekday, weather
            pred = ENGINE.predict_game(home, away, ctx=g)
        except Exception:
            continue
        margin, total, p_home = pred["pred_margin"], pred["pred_total"], pred["home_win_prob"]
        base = dict(game_id=g.game_id, week=int(g.week), gameday=str(g.gameday), home=home, away=away,
                    pred_margin=margin, pred_total=total, home_win_prob=p_home)

        def add(market, side, team, label, line, edge_pts, p_side, p_push, odds, other, hist_edge):
            picks.append(base | price(p_side, p_push, float(odds), float(other), kw[market]) | dict(
                market=market, side=side, team=team, label=label, line=line,
                edge_pts=edge_pts, history=pick_history(market, hist_edge)))

        # spread_line is the home team's expected margin (positive = home
        # favoured): the home side covers when its margin exceeds it, and its
        # betting line is shown as -spread_line
        if pd.notna(g.spread_line) and pd.notna(g.home_spread_odds) and pd.notna(g.away_spread_odds):
            line = float(g.spread_line)
            cover, push, lose = (float(x[0]) for x in cover_probs(margin, line, meta["sigma_margin"], factors))
            edge = round(margin - line, 2)
            add("spread", "home", home, f"{home} {-line:+.1f}", line, edge,
                cover / (cover + lose), push, g.home_spread_odds, g.away_spread_odds, edge)
            add("spread", "away", away, f"{away} {line:+.1f}", line, edge,
                lose / (cover + lose), push, g.away_spread_odds, g.home_spread_odds, edge)

        if pd.notna(g.total_line) and pd.notna(g.over_odds) and pd.notna(g.under_odds):
            line = float(g.total_line)
            over, push, under = (float(x[0]) for x in total_probs(total, line, meta["sigma_total"]))
            edge = round(total - line, 2)
            add("total", "over", None, f"OVER {line:.1f}", line, edge,
                over / (over + under), push, g.over_odds, g.under_odds, edge)
            add("total", "under", None, f"UNDER {line:.1f}", line, edge,
                under / (over + under), push, g.under_odds, g.over_odds, edge)

        if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            fair_home = float(fair_prob(g.home_moneyline, g.away_moneyline))
            add("moneyline", "home", home, f"{home} ML", None, None,
                p_home, 0.0, g.home_moneyline, g.away_moneyline, p_home - fair_home)
            add("moneyline", "away", away, f"{away} ML", None, None,
                1 - p_home, 0.0, g.away_moneyline, g.home_moneyline, p_home - fair_home)

    picks.sort(key=lambda x: -x["sharpe"])
    return picks


def load_schedule(season: int | None = None) -> pd.DataFrame:
    """The raw schedule: the DuckDB `games` view drops the odds columns, and
    the card needs the posted prices, not just the lines. Default season: the
    latest with results, or the next one once its lines are posted."""
    s = read_parquet(RAW / "schedules.parquet")
    if season is None:
        season = int(s.dropna(subset=["result"]).season.max())
        if len(s[(s.season == season + 1) & s.spread_line.notna()]):
            season += 1
    s = s[s.season == season].copy()
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s.sort_values(["week", "gameday"])
