"""Game-level feature engineering.

For every game, builds leak-free features from prior games only:
  - opponent-adjusted rolling team form (EPA splits, success, explosiveness,
    sacks, turnovers) with exponential recency decay
  - situational: rest, travel/timezones, division game, roof/surface/weather
  - coaching: new-coach flags, tenure, career win%
  - QB: rolling efficiency/tendency profile of the starter, QB-change flags
  - Elo rating (also a standalone baseline)

Referee tendencies were dropped on 2026-11-12: the referee is not announced
before kickoff, so the live system could never use them, and removing them
changes the backtest by less than 0.02 points either way.

Outputs data/processed/game_features.parquet plus "current snapshot" tables
used to predict future (not yet played) matchups.
"""
import numpy as np
import pandas as pd

from nfl_engine.config import PROCESSED
from nfl_engine.data.store import connect
from nfl_engine.features.elo import compute_elo
from nfl_engine.features.stadiums import travel_km, tz_diff
from nfl_engine.io_utils import read_parquet as safe_read_parquet, to_parquet as safe_to_parquet

EWM_HALFLIFE = 6      # games; recency decay for team form
QB_HALFLIFE = 8       # games; QB profiles are stabler
MINP = 3

FORM_METRICS = ["epa_play", "pass_epa", "rush_epa", "success_rate",
                "explosive_rate", "sack_rate", "to_rate", "third_down_rate",
                "early_pass_epa", "early_rush_epa", "st_epa", "points_for",
                "plays", "pass_rate"]
QB_METRICS = ["epa_dropback", "cpoe", "sack_rate", "deep_rate",
              "scramble_rate", "adot"]


def known_before(d: pd.DataFrame, asof: tuple[int, int]) -> pd.Series:
    """Rows from games played before week `asof[1]` of season `asof[0]`."""
    return (d["season"] < asof[0]) | ((d["season"] == asof[0]) & (d["week"] < asof[1]))


def _load(asof: tuple[int, int] | None = None):
    con = connect()
    games = con.execute("SELECT * FROM games ORDER BY gameday, game_id").df()
    tg = con.execute("""
        SELECT t.*, f.fumbles, f.fumbles_lost
        FROM team_game t LEFT JOIN team_fumbles f
          ON t.game_id = f.game_id AND t.team = f.team
    """).df()
    qb = con.execute("SELECT * FROM qb_game").df()
    con.close()
    games["gameday"] = pd.to_datetime(games["gameday"])
    if asof is not None:
        # rebuild the world as it stood before that week: later games become
        # unplayed schedule rows and their play-by-play disappears
        future = ~known_before(games, asof)
        games.loc[future, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
        tg, qb = tg[known_before(tg, asof)], qb[known_before(qb, asof)]
    return games, tg, qb


def build_team_rows(games: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) with offensive and allowed (def) metrics."""
    tg = tg.copy()
    tg["to_rate"] = tg["turnovers"] / tg["plays"].clip(lower=1)
    # points scored per team-game from the schedule
    h = games[["game_id", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for"})
    a = games[["game_id", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for"})
    pts = pd.concat([h, a], ignore_index=True)
    tg = tg.merge(pts, on=["game_id", "team"], how="left")
    # fumble luck: lost fumbles above the ~45% league recovery expectation;
    # mean-reverting, so bad luck predicts improvement
    tg["fum_luck"] = tg["fumbles_lost"].fillna(0) - 0.45 * tg["fumbles"].fillna(0)
    off = tg[["game_id", "team", "opponent", "fum_luck"] + FORM_METRICS].copy()
    # What a team allowed = what its opponent produced.
    d = (off.drop(columns=["fum_luck"])
         .rename(columns={"team": "_o", "opponent": "team"})
         .rename(columns={"_o": "opponent"}))
    d = d.rename(columns={m: f"allowed_{m}" for m in FORM_METRICS})
    rows = off.merge(d, on=["game_id", "team", "opponent"], how="inner")
    meta = games[["game_id", "season", "week", "gameday"]]
    rows = rows.merge(meta, on="game_id", how="inner")
    return rows.sort_values(["team", "gameday"]).reset_index(drop=True)


def opponent_adjust(rows: pd.DataFrame) -> pd.DataFrame:
    """Adjust each game's raw metrics by opponent strength entering that game.

    Opponent strength = opponent's rolling allowed/produced metric (prior
    games, EWM) relative to the league rolling mean at that date.
    """
    rows = rows.sort_values(["team", "gameday"]).copy()
    g = rows.groupby("team", sort=False)

    for m in FORM_METRICS:
        rows[f"pre_allowed_{m}"] = g[f"allowed_{m}"].transform(
            lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=MINP).mean())
        rows[f"pre_off_{m}"] = g[m].transform(
            lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=MINP).mean())

    # League baseline per date: the mean over the 365 days up to and
    # including it. Same-day games are fair to use: the baseline only adjusts
    # stats that feed LATER games. It is aggregated per day so the result does
    # not depend on the order of same-day rows (a row-wise rolling mean did,
    # which made as-of rebuilds differ from the full build in the 4th decimal).
    daily = rows.groupby("gameday")[FORM_METRICS].agg(["sum", "count"]).sort_index()
    for m in FORM_METRICS:
        tot = daily[(m, "sum")].rolling("365D").sum()
        cnt = daily[(m, "count")].rolling("365D").sum()
        rows[f"lg_{m}"] = rows["gameday"].map((tot / cnt).where(cnt >= 50))

    # Join opponent's pregame strength onto each row.
    opp = rows[["game_id", "team"] + [f"pre_allowed_{m}" for m in FORM_METRICS]
               + [f"pre_off_{m}" for m in FORM_METRICS]].rename(columns={"team": "opponent"})
    opp = opp.rename(columns={f"pre_allowed_{m}": f"opp_allowed_{m}" for m in FORM_METRICS}
                     | {f"pre_off_{m}": f"opp_off_{m}" for m in FORM_METRICS})
    rows = rows.merge(opp, on=["game_id", "opponent"], how="left")

    # Adjusted per-game performance: raw minus how easy the opponent is.
    for m in FORM_METRICS:
        opp_def_edge = rows[f"opp_allowed_{m}"] - rows[f"lg_{m}"]
        rows[f"adj_off_{m}"] = rows[m] - opp_def_edge.fillna(0)
        opp_off_edge = rows[f"opp_off_{m}"] - rows[f"lg_{m}"]
        rows[f"adj_allowed_{m}"] = rows[f"allowed_{m}"] - opp_off_edge.fillna(0)
    return rows


def rolling_form(rows: pd.DataFrame) -> pd.DataFrame:
    """EWM/window rolling of adjusted metrics, shifted so features at each
    game use prior games only. Also emits unshifted 'current' values."""
    rows = rows.sort_values(["team", "gameday"]).copy()
    g = rows.groupby("team", sort=False)
    feat_cols = []
    for m in FORM_METRICS:
        for side in ("off", "allowed"):
            col = f"adj_{side}_{m}"
            f = f"{side}_{m}_ewm"
            rows[f] = g[col].transform(
                lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=MINP).mean())
            rows[f + "_cur"] = g[col].transform(
                lambda s: s.ewm(halflife=EWM_HALFLIFE, min_periods=MINP).mean())
            feat_cols.append(f)
    # Fast recency window on the headline metric (halflife 3)
    for side in ("off", "allowed"):
        col = f"adj_{side}_epa_play"
        f = f"{side}_epa_play_ewm3"
        rows[f] = g[col].transform(
            lambda s: s.shift(1).ewm(halflife=3, min_periods=MINP).mean())
        rows[f + "_cur"] = g[col].transform(
            lambda s: s.ewm(halflife=3, min_periods=MINP).mean())
        feat_cols.append(f)
    # Fumble luck (mean-reverting, unadjusted)
    rows["fum_luck_ewm"] = g["fum_luck"].transform(
        lambda s: s.shift(1).ewm(halflife=8, min_periods=MINP).mean())
    rows["fum_luck_ewm_cur"] = g["fum_luck"].transform(
        lambda s: s.ewm(halflife=8, min_periods=MINP).mean())
    feat_cols.append("fum_luck_ewm")
    # Short-window unadjusted momentum + season-to-date
    rows["off_epa_l4"] = g["epa_play"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    rows["allowed_epa_l4"] = g["allowed_epa_play"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    rows["off_epa_l4_cur"] = g["epa_play"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    rows["allowed_epa_l4_cur"] = g["allowed_epa_play"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    sg = rows.groupby(["team", "season"], sort=False)
    rows["games_played_season"] = sg.cumcount()
    rows["off_epa_std"] = sg["epa_play"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    rows["allowed_epa_std"] = sg["allowed_epa_play"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    rows["off_epa_std_cur"] = sg["epa_play"].transform(lambda s: s.expanding(min_periods=1).mean())
    rows["allowed_epa_std_cur"] = sg["allowed_epa_play"].transform(lambda s: s.expanding(min_periods=1).mean())
    feat_cols += ["off_epa_l4", "allowed_epa_l4", "games_played_season",
                  "off_epa_std", "allowed_epa_std"]
    return rows, feat_cols


def coach_features(games: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): coach tenure with team, career games, career
    win%, new-coach flags. Uses prior games only."""
    h = games[["game_id", "gameday", "season", "home_team", "home_coach", "result"]].rename(
        columns={"home_team": "team", "home_coach": "coach"})
    h["won"] = (h["result"] > 0).astype(float)
    a = games[["game_id", "gameday", "season", "away_team", "away_coach", "result"]].rename(
        columns={"away_team": "team", "away_coach": "coach"})
    a["won"] = (a["result"] < 0).astype(float)
    rows = pd.concat([h, a]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rows.loc[rows["result"].isna(), "won"] = np.nan

    rows["coach_tenure_games"] = rows.groupby(["team", "coach"]).cumcount()
    rows["coach_career_games"] = rows.groupby("coach").cumcount()
    cw = rows.groupby("coach", sort=False)["won"]
    rows["coach_career_winpct"] = cw.transform(
        lambda s: s.shift(1).expanding(min_periods=8).mean())
    rows["coach_new_team"] = (rows["coach_tenure_games"] < 17).astype(int)
    rows["coach_rookie"] = (rows["coach_career_games"] < 17).astype(int)
    return rows[["game_id", "team", "coach_tenure_games", "coach_career_games",
                 "coach_career_winpct", "coach_new_team", "coach_rookie"]]


def qb_profiles(qb: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-QB profile keyed by (game_id, qb_id), prior games only,
    plus a current snapshot per QB."""
    meta = games[["game_id", "gameday"]]
    qb = qb.merge(meta, on="game_id", how="inner")
    qb = qb[qb["dropbacks"] >= 10].sort_values(["qb_id", "gameday"]).reset_index(drop=True)
    g = qb.groupby("qb_id", sort=False)
    for m in QB_METRICS:
        qb[f"qb_{m}_ewm"] = g[m].transform(
            lambda s: s.shift(1).ewm(halflife=QB_HALFLIFE, min_periods=3).mean())
        qb[f"qb_{m}_ewm_cur"] = g[m].transform(
            lambda s: s.ewm(halflife=QB_HALFLIFE, min_periods=3).mean())
    qb["qb_career_dropbacks"] = g["dropbacks"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    qb["qb_games"] = g.cumcount()
    return qb


def starter_features(games: pd.DataFrame) -> pd.DataFrame:
    """QB-change flags per (game_id, team) from the scheduled starter ids."""
    h = games[["game_id", "gameday", "home_team", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id"})
    a = games[["game_id", "gameday", "away_team", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id"})
    rows = pd.concat([h, a]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rows["prev_qb_id"] = rows.groupby("team")["qb_id"].shift(1)
    rows["qb_new"] = ((rows["qb_id"] != rows["prev_qb_id"])
                      & rows["prev_qb_id"].notna()).astype(int)
    rows["qb_starts_team"] = rows.groupby(["team", "qb_id"]).cumcount()
    return rows[["game_id", "team", "qb_new", "qb_starts_team"]]


def venue_flags(games: pd.DataFrame) -> pd.DataFrame:
    """is_dome / is_turf per game. nflverse fills in `surface` (and for
    future games `roof`) weeks late; a blank takes the stadium's most recent
    known value, then the home team's. Reading a blank surface as "not grass"
    had marked every recent game as turf."""
    g = games.sort_values(["gameday", "game_id"])[
        ["game_id", "stadium", "home_team", "roof", "surface"]].copy()
    for c in ("roof", "surface"):
        v = g[c].replace("", np.nan)
        v = v.fillna(v.groupby(g["stadium"]).ffill())
        g[c] = v.fillna(v.groupby(g["home_team"]).ffill())
    g["is_dome"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["is_turf"] = (~g["surface"].fillna("grass").str.contains("grass")).astype(int)
    return g[["game_id", "is_dome", "is_turf"]]


def build(save: bool = True, asof: tuple[int, int] | None = None, out_dir=PROCESSED):
    """Build game features and the team/QB snapshots used to predict future
    games. With `asof=(season, week)` the build sees only games played before
    that week and writes to `out_dir` — the state the live system would have
    had then (used by tests/test_parity.py)."""
    games, tg, qb = _load(asof)

    rows = build_team_rows(games, tg)
    rows = opponent_adjust(rows)
    rows, form_cols = rolling_form(rows)

    # Rest days from actual schedule sequence
    rows["rest_days"] = rows.groupby("team")["gameday"].diff().dt.days.clip(upper=30)

    coach = coach_features(games)
    starter = starter_features(games)
    side_feats = (rows[["game_id", "team"] + form_cols + ["rest_days"]]
                  .merge(coach, on=["game_id", "team"], how="left")
                  .merge(starter, on=["game_id", "team"], how="left"))

    qbp = qb_profiles(qb, games)
    qb_cols = [f"qb_{m}_ewm" for m in QB_METRICS] + ["qb_career_dropbacks", "qb_games"]
    qb_by_game = qbp[["game_id", "qb_id"] + qb_cols]

    out = compute_elo(games.copy())
    for side in ("home", "away"):
        sf = side_feats.add_prefix(f"{side}_").rename(columns={f"{side}_game_id": "game_id"})
        out = out.merge(sf, left_on=["game_id", f"{side}_team"],
                        right_on=["game_id", f"{side}_team"], how="left")
        qf = qb_by_game.add_prefix(f"{side}_").rename(columns={f"{side}_game_id": "game_id"})
        out = out.merge(qf, left_on=["game_id", f"{side}_qb_id"],
                        right_on=["game_id", f"{side}_qb_id"], how="left")

    # Travel and timezone for the away side
    out["away_travel_km"] = [
        travel_km(at, ht, s) if loc != "Neutral" else 800.0
        for at, ht, s, loc in zip(out.away_team, out.home_team, out.season, out.location)]
    out["away_tz_diff"] = [
        tz_diff(at, ht, s) for at, ht, s in zip(out.away_team, out.home_team, out.season)]

    # Venue/schedule flags
    out["is_denver"] = (out["home_team"] == "DEN").astype(int)
    out["is_thursday"] = (out["weekday"] == "Thursday").astype(int)
    out["is_monday"] = (out["weekday"] == "Monday").astype(int)
    out["is_primetime"] = (out["gametime"].fillna("13:00") >= "20:00").astype(int)

    # Weather/venue (temp and wind stay missing when nflverse has not
    # published them: unknown, not "average")
    out = out.merge(venue_flags(games), on="game_id", how="left")
    out["temp_f"] = np.where(out["is_dome"] == 1, 70.0, out["temp"])
    out["wind_mph"] = np.where(out["is_dome"] == 1, 0.0, out["wind"])
    out["rest_diff"] = out["home_rest"] - out["away_rest"]

    # Differential features (home minus away)
    diff_bases = form_cols + ["rest_days", "coach_career_winpct", "qb_epa_dropback_ewm",
                              "qb_cpoe_ewm", "qb_career_dropbacks"]
    for b in diff_bases:
        hb, ab = f"home_{b}", f"away_{b}"
        if hb in out.columns and ab in out.columns:
            out[f"d_{b}"] = out[hb] - out[ab]

    out["home_win"] = np.where(out["result"].isna(), np.nan,
                               (out["result"] > 0).astype(float))
    out["is_playoff"] = (out["game_type"] != "REG").astype(int)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_to_parquet(out, out_dir / "game_features.parquet")
        # Current snapshots for predicting future matchups
        cur_cols = [c for c in rows.columns if c.endswith("_cur")]
        snap = (rows.sort_values("gameday").groupby("team")
                .tail(1)[["team", "season", "gameday"] + cur_cols])
        snap.columns = [c.replace("_cur", "") if c.endswith("_cur") else c
                        for c in snap.columns]
        safe_to_parquet(snap, out_dir / "team_current.parquet")
        qsnap_cols = ["qb_id", "qb_name", "team", "gameday", "dropbacks"] + \
                     [f"qb_{m}_ewm_cur" for m in QB_METRICS]
        qsnap = (qbp.sort_values("gameday").groupby("qb_id").tail(1)[qsnap_cols])
        qsnap["career_dropbacks"] = qbp.groupby("qb_id")["dropbacks"].sum().reindex(qsnap.qb_id).values
        # qualifying games so far = the qb_games value of his next game
        qsnap["qb_games"] = qbp.groupby("qb_id").size().reindex(qsnap.qb_id).values
        qsnap.columns = [c.replace("_ewm_cur", "") for c in qsnap.columns]
        safe_to_parquet(qsnap, out_dir / "qb_current.parquet")
        print(f"game_features: {out.shape}, team_current: {snap.shape}, qb_current: {qsnap.shape}")
    return out


if __name__ == "__main__":
    build()
