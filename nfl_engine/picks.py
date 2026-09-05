"""Weekly betting-card construction.

For every scheduled game with posted odds, compares the model's probability
against the de-vigged market probability and scores each available bet by
expected value, Sharpe ratio, and Kelly stake.

Every pick also carries the *backtested* hit rate for its edge bucket, so the
historical track record of that kind of pick is visible next to the pick
itself rather than buried in a disclaimer.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_engine.config import norm_team
from nfl_engine.query.serving import ENGINE

BREAKEVEN = 0.5238


def american_to_decimal(odds: float) -> float:
    """American odds -> decimal (total return per 1 staked)."""
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def devig(p_a: float, p_b: float) -> tuple[float, float]:
    """Remove the bookmaker's margin from a two-way market (proportional)."""
    s = p_a + p_b
    return (p_a / s, p_b / s) if s > 0 else (p_a, p_b)


# Full Kelly assumes the stated probability is correct. Ours is not
# demonstrably better than the market (see scripts/edge_analysis.py), so the
# recommended stake is a quarter-Kelly capped at 2% of bankroll — the standard
# defence against staking on an overestimated edge.
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02


def bet_metrics(p_model: float, decimal_odds: float) -> dict:
    """EV, Sharpe and staking for a 1-unit binary bet."""
    b = decimal_odds - 1.0            # profit if it wins
    ev = p_model * b - (1 - p_model)  # expected profit per unit staked
    var = p_model * (b - ev) ** 2 + (1 - p_model) * (-1 - ev) ** 2
    sd = float(np.sqrt(var))
    kelly = max((p_model * decimal_odds - 1) / b, 0.0) if b > 0 else 0.0
    return {
        "ev": round(float(ev), 4),
        "sharpe": round(float(ev / sd), 4) if sd > 0 else 0.0,
        "kelly": round(float(kelly), 4),
        "stake": round(float(min(kelly * KELLY_FRACTION, MAX_STAKE)), 4),
        "decimal": round(float(decimal_odds), 3),
    }


# Backtested ATS / O-U hit rate by size of disagreement with the closing line
# (2010-2025 walk-forward; see scripts/edge_analysis.py). These are the honest
# track records attached to each pick.
HIST_ATS = [(0.5, 2, 0.4920, 1689), (2, 4, 0.5291, 1325), (4, 6, 0.5281, 462),
            (6, 9, 0.4436, 133), (9, 99, 0.5152, 33)]
HIST_TOT = [(0.5, 2, 0.5250, 1665), (2, 4, 0.5045, 1343), (4, 6, 0.5279, 501),
            (6, 99, 0.5230, 145)]
# Moneyline: hit rate and ROI by model-vs-market probability edge
# (scripts/moneyline_history.py). ROI is the honest number here — a low hit
# rate on big underdogs can still profit, and vice versa.
HIST_ML = [(0.00, 0.02, 0.485, 860, -0.0509), (0.02, 0.05, 0.502, 1177, 0.0124),
           (0.05, 0.10, 0.449, 1298, -0.0177), (0.10, 0.20, 0.440, 904, 0.0283),
           (0.20, 1.01, 0.475, 122, 0.2897)]


def _hist(table, edge_pts: float) -> dict:
    for row in table:
        lo, hi, rate, n = row[:4]
        if lo <= abs(edge_pts) < hi:
            out = {"rate": rate, "n": n}
            if len(row) > 4:
                out["roi"] = row[4]
            return out
    return {"rate": None, "n": 0}


def build_picks(schedule: pd.DataFrame) -> list[dict]:
    """One row per available bet across the given games."""
    gm = ENGINE.game_models["pure"]
    sigma_m = gm["meta"]["sigma_margin"]
    sigma_t = gm["meta"]["sigma_total"]
    picks = []

    for g in schedule.itertuples():
        home, away = norm_team(g.home_team), norm_team(g.away_team)
        try:
            pred = ENGINE.predict_game(home, away)
        except Exception:
            continue
        margin, total = pred["pred_margin"], pred["pred_total"]
        p_home_win = pred["home_win_prob"]

        base = dict(game_id=g.game_id, week=int(g.week), gameday=str(g.gameday),
                    home=home, away=away,
                    pred_margin=margin, pred_total=total,
                    home_win_prob=p_home_win)

        # --- spread ---
        if pd.notna(g.spread_line) and pd.notna(g.home_spread_odds):
            line = float(g.spread_line)
            # home covers when (home - away) margin exceeds the line
            p_home_cover = float(1 - norm.cdf((line - margin) / sigma_m))
            edge_pts = margin - line
            for side, p_m, odds in (("home", p_home_cover, g.home_spread_odds),
                                    ("away", 1 - p_home_cover, g.away_spread_odds)):
                d = american_to_decimal(float(odds))
                mkt = 1 / d
                other = 1 / american_to_decimal(
                    float(g.away_spread_odds if side == "home" else g.home_spread_odds))
                fair, _ = devig(mkt, other)
                m = bet_metrics(p_m, d)
                team = home if side == "home" else away
                shown = line if side == "home" else -line
                picks.append(base | m | dict(
                    market="spread", side=side, team=team,
                    label=f"{team} {shown:+.1f}", line=line, odds=float(odds),
                    p_model=round(p_m, 4), p_market=round(fair, 4),
                    edge_prob=round(p_m - fair, 4), edge_pts=round(edge_pts, 2),
                    history=_hist(HIST_ATS, edge_pts)))

        # --- total ---
        if pd.notna(g.total_line) and pd.notna(g.over_odds):
            tl = float(g.total_line)
            p_over = float(1 - norm.cdf((tl - total) / sigma_t))
            edge_pts = total - tl
            for side, p_m, odds in (("over", p_over, g.over_odds),
                                    ("under", 1 - p_over, g.under_odds)):
                d = american_to_decimal(float(odds))
                other = 1 / american_to_decimal(
                    float(g.under_odds if side == "over" else g.over_odds))
                fair, _ = devig(1 / d, other)
                m = bet_metrics(p_m, d)
                picks.append(base | m | dict(
                    market="total", side=side, team=None,
                    label=f"{side.upper()} {tl:.1f}", line=tl, odds=float(odds),
                    p_model=round(p_m, 4), p_market=round(fair, 4),
                    edge_prob=round(p_m - fair, 4), edge_pts=round(edge_pts, 2),
                    history=_hist(HIST_TOT, edge_pts)))

        # --- moneyline ---
        if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            for side, p_m, odds in (("home", p_home_win, g.home_moneyline),
                                    ("away", 1 - p_home_win, g.away_moneyline)):
                d = american_to_decimal(float(odds))
                other = 1 / american_to_decimal(
                    float(g.away_moneyline if side == "home" else g.home_moneyline))
                fair, _ = devig(1 / d, other)
                m = bet_metrics(p_m, d)
                team = home if side == "home" else away
                picks.append(base | m | dict(
                    market="moneyline", side=side, team=team,
                    label=f"{team} ML", line=None, odds=float(odds),
                    p_model=round(p_m, 4), p_market=round(fair, 4),
                    edge_prob=round(p_m - fair, 4), edge_pts=None,
                    history=_hist(HIST_ML, p_m - fair)))

    picks.sort(key=lambda x: -x["sharpe"])
    return picks


def load_schedule(season: int | None = None) -> pd.DataFrame:
    """Raw schedule: the DuckDB `games` view drops the odds columns, and the
    picks card needs the posted prices, not just the lines."""
    from nfl_engine.config import RAW
    from nfl_engine.io_utils import read_parquet
    s = read_parquet(RAW / "schedules.parquet")
    if season is None:
        played = s.dropna(subset=["result"])
        season = int(played.season.max())
        upcoming = s[(s.season == season + 1) & s.spread_line.notna()]
        if len(upcoming):
            season = season + 1
    s = s[s.season == season].copy()
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s.sort_values(["week", "gameday"])
