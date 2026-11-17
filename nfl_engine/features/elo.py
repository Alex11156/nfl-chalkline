"""Elo ratings with margin-of-victory multiplier (538-style).

Produces a pregame Elo for every game — both a baseline model and a
feature for the ML models.
"""
import numpy as np
import pandas as pd

MEAN = 1505.0
K = 20.0
HOME_ADV = 48.0
SEASON_REGRESS = 1 / 3  # regress toward mean between seasons


def elo_win_prob(elo_diff: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(elo_diff) / 400.0))


def elo_to_spread(elo_diff: float) -> float:
    return elo_diff / 25.0  # standard 538 conversion


def _run(games: pd.DataFrame) -> tuple[pd.DataFrame, list, list, dict]:
    """One pass over the schedule in date order. Returns the sorted games,
    each game's pregame ratings, and every team's rating after its last
    played game."""
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    home_pre, away_pre = [], []

    for row in games.itertuples():
        for team in (row.home_team, row.away_team):
            if team not in ratings:
                ratings[team] = MEAN
                last_season[team] = row.season
            elif last_season[team] != row.season:
                ratings[team] = MEAN + (ratings[team] - MEAN) * (1 - SEASON_REGRESS)
                last_season[team] = row.season

        h, a = ratings[row.home_team], ratings[row.away_team]
        home_pre.append(h)
        away_pre.append(a)

        if pd.isna(row.result):
            continue
        hfa = 0.0 if row.location == "Neutral" else HOME_ADV
        diff = h - a + hfa
        p_home = elo_win_prob(diff)
        margin = row.result  # home - away
        outcome = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        mov_mult = np.log(abs(margin) + 1) * (2.2 / (0.001 * diff * np.sign(margin or 1) + 2.2))
        shift = K * mov_mult * (outcome - p_home)
        ratings[row.home_team] = h + shift
        ratings[row.away_team] = a - shift
    return games, home_pre, away_pre, ratings


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """games: home_team, away_team, result, season, location, gameday.
    Returns games (date-sorted) with home_elo_pre, away_elo_pre, elo_diff
    (home perspective incl. home advantage) and elo_prob_home."""
    games, home_pre, away_pre, _ = _run(games)
    games["home_elo_pre"] = home_pre
    games["away_elo_pre"] = away_pre
    hfa_vec = np.where(games["location"] == "Neutral", 0.0, HOME_ADV)
    games["elo_diff"] = games["home_elo_pre"] - games["away_elo_pre"] + hfa_vec
    games["elo_prob_home"] = elo_win_prob(games["elo_diff"].values)
    return games


def current_ratings(games: pd.DataFrame) -> dict[str, float]:
    """Each team's rating after its last played game: the pregame rating of
    its next game in the same season."""
    return _run(games.dropna(subset=["result"]))[3]
