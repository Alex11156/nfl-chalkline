"""Betting-odds arithmetic, shared by the betting card, the prop tracker and
the evaluators."""
import numpy as np

BREAKEVEN = 0.5238      # win rate needed to break even at -110

# Full Kelly assumes the stated probability is correct. Ours is not
# demonstrably better than the market (see research/edge_search.py), so the
# recommended stake is a quarter-Kelly capped at 2% of bankroll: the standard
# defence against staking on an overestimated edge.
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02


def american_to_decimal(odds):
    """American odds -> decimal (total return per 1 staked). Scalar or array."""
    o = np.asarray(odds, dtype=float)
    d = np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / np.abs(o))
    return float(d) if d.ndim == 0 else d


def devig(p_a, p_b):
    """Remove the bookmaker's margin from a two-way market (proportional).
    Both sides of a real market sum to more than 100%; the excess is the fee."""
    s = p_a + p_b
    return p_a / s, p_b / s


def fair_prob(odds_a, odds_b):
    """De-vigged probability of side A from the two posted American prices."""
    return devig(1.0 / american_to_decimal(odds_a), 1.0 / american_to_decimal(odds_b))[0]


def bet_metrics(p_win: float, decimal_odds: float, p_push: float = 0.0) -> dict:
    """EV, Sharpe and staking for a 1-unit bet that wins with `p_win`, is
    refunded with `p_push` and loses otherwise."""
    b = decimal_odds - 1.0                       # profit if it wins
    p_lose = max(1.0 - p_win - p_push, 0.0)
    ev = p_win * b - p_lose                      # expected profit per unit staked
    var = p_win * (b - ev) ** 2 + p_lose * (-1 - ev) ** 2 + p_push * ev ** 2
    sd = float(np.sqrt(var))
    kelly = max((p_win * b - p_lose) / b, 0.0) if b > 0 else 0.0
    return {
        "ev": round(float(ev), 4),
        "sharpe": round(float(ev / sd), 4) if sd > 0 else 0.0,
        "kelly": round(float(kelly), 4),
        "stake": round(float(min(kelly * KELLY_FRACTION, MAX_STAKE)), 4),
        "decimal": round(float(decimal_odds), 3),
    }
