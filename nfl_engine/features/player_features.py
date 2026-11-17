"""Player-week feature engineering for fantasy and prop models.

For every player-week (QB/RB/WR/TE), builds leak-free features from prior
weeks only: rolling production/usage with recency decay, role (snaps, shares,
depth chart, starting QB), team offensive context, opponent defense vs. the
player's position, and Vegas implied totals.

Outputs (to data/processed/, or `out_dir` in as-of mode):
  player_features.parquet   one training row per player-game
  player_current.parquet    each player's latest state, for future games
  dvp_current.parquet       each defense's latest allowed-by-position form
  depth_current.parquet     the latest depth chart (who is expected to start)
"""
import numpy as np
import pandas as pd

from nfl_engine.config import PROCESSED, RAW, norm_team
from nfl_engine.data.store import connect
from nfl_engine.features.game_features import known_before
from nfl_engine.io_utils import read_parquet as safe_read_parquet, to_parquet as safe_to_parquet

HALFLIFE = 5
STAT_COLS = [
    "fantasy_points_ppr", "fantasy_points",
    "passing_yards", "passing_tds", "passing_interceptions", "completions",
    "attempts", "passing_epa", "passing_cpoe",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share", "wopr", "receiving_air_yards",
]
TEAM_CTX = ["off_epa_play_ewm", "off_pass_epa_ewm", "off_rush_epa_ewm",
            "allowed_epa_play_ewm", "allowed_pass_epa_ewm", "allowed_rush_epa_ewm",
            "off_plays_ewm", "off_pass_rate_ewm", "allowed_pass_rate_ewm"]
USAGE_COLS = ["rz_targets", "ez_targets", "rz_carries", "gl_carries"]
# per-game shares computed here (not in the stats feed); rolled like usage
SHARE_COLS = ["carry_share"]
NG_COLS = ["ng_ttt", "ng_aggr", "ng_pass_iay", "ng_sep", "ng_cushion",
           "ng_yacoe", "ng_box8", "ng_ryoe"]
DVP_COLS = ["fp_ppr_allowed", "rush_yds_allowed", "rec_yds_allowed",
            "rec_allowed", "pass_yds_allowed", "pass_att_allowed", "comp_allowed",
            "carries_allowed", "targets_allowed"]


def _team_long(gf: pd.DataFrame) -> pd.DataFrame:
    """Reshape game features to one row per (game_id, team) with team
    context + opponent context + implied totals."""
    outs = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = {f"{side}_{c}": c for c in TEAM_CTX}
        cols.update({f"{opp}_{c}": f"opp_{c}" for c in TEAM_CTX})
        cols[f"{side}_qb_epa_dropback_ewm"] = "team_qb_epa"
        cols[f"{side}_qb_new"] = "team_qb_new"   # starter differs from the team's last game
        d = gf[["game_id", "season", "week", "gameday", f"{side}_team",
                f"{opp}_team", "spread_line", "total_line", "is_dome",
                "wind_mph"] + list(cols)].rename(
            columns={f"{side}_team": "team", f"{opp}_team": "opponent", **cols})
        sign = 1.0 if side == "home" else -1.0
        d["implied_total"] = (d["total_line"] + sign * d["spread_line"]) / 2.0
        # game script: is the player's team at home, and by how much is it
        # favoured? (spread_line alone is home-perspective and cannot say)
        d["is_home"] = float(side == "home")
        d["team_spread"] = sign * d["spread_line"]
        d["team_qb_id"] = gf[f"{side}_qb_id"].values
        outs.append(d)
    return pd.concat(outs, ignore_index=True)



def current_roster() -> pd.DataFrame:
    """player_id -> team/status from the most recent published roster.

    A player's last *played game* is with their old team after an offseason
    move, so the serving snapshot must come from the roster, not game logs.
    """
    r = safe_read_parquet(RAW / "rosters.parquet")
    r = r[r.season == r.season.max()]
    # ACT = active, INA = on the 53 but inactive last game, RES = injured reserve
    r = r[r.status.isin(["ACT", "INA", "RES"])]
    out = (r[["gsis_id", "team", "status", "season"]]
           .rename(columns={"gsis_id": "player_id", "team": "roster_team",
                            "status": "roster_status", "season": "roster_season"})
           .dropna(subset=["player_id"]).drop_duplicates("player_id"))
    out["roster_team"] = out["roster_team"].map(norm_team)
    return out


def depth_charts(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Depth-chart rank per player from both nflverse formats.

    Through 2024: one chart per (season, week). From 2025 the source changed
    to daily ESPN-style snapshots with no season/week columns; ignoring them
    left every 2025+ row carrying the player's last 2024 rank.
    Returns (weekly, daily)."""
    weekly = con.execute("""
        SELECT gsis_id AS player_id, CAST(season AS INT) AS season,
               CAST(week AS INT) AS week,
               MIN(TRY_CAST(depth_team AS INT)) AS depth_rank
        FROM depth_charts
        WHERE gsis_id IS NOT NULL AND depth_team IS NOT NULL
          AND position IN ('QB','RB','WR','TE')
        GROUP BY 1, 2, 3
    """).df()
    daily = con.execute("""
        SELECT gsis_id AS player_id, CAST(SUBSTR(dt, 1, 10) AS DATE) AS day,
               MIN(CAST(pos_rank AS INT)) AS depth_rank,
               ANY_VALUE(team) AS team, ANY_VALUE(pos_abb) AS position
        FROM depth_charts
        WHERE season IS NULL AND gsis_id IS NOT NULL AND pos_rank IS NOT NULL
          AND pos_abb IN ('QB','RB','WR','TE')
        GROUP BY 1, 2
    """).df()
    daily["day"] = pd.to_datetime(daily["day"]).astype("datetime64[ns]")
    daily["team"] = daily["team"].map(norm_team)
    return weekly, daily.sort_values("day").reset_index(drop=True)


def build(save: bool = True, asof: tuple[int, int] | None = None, out_dir=PROCESSED):
    """Build player features and serving snapshots. With `asof=(season, week)`
    only games before that week are visible and outputs go to `out_dir` — the
    state the live system would have had then (tests/test_parity.py)."""
    con = connect()
    ps = con.execute("""
        SELECT player_id, player_display_name AS player_name, position,
               season, week, season_type, game_id, team, opponent_team,
               {cols}
        FROM player_stats
        WHERE position IN ('QB','RB','WR','TE')
          AND season_type IN ('REG','POST')
    """.format(cols=", ".join(STAT_COLS))).df()
    dvp = con.execute("SELECT * FROM def_vs_pos").df()
    snaps = con.execute("""
        SELECT s.game_id, p.gsis_id AS player_id, s.offense_pct
        FROM snap_counts s JOIN players p ON s.pfr_player_id = p.pfr_id
        WHERE p.gsis_id IS NOT NULL AND s.offense_snaps > 0
    """).df()
    usage = con.execute("SELECT * FROM player_pbp_usage").df()
    depth, depth_daily = depth_charts(con)
    con.close()

    # NextGen Stats (2016+), weekly rows only (week 0 = season aggregate)
    ng_frames = []
    for st, cols in [
        ("passing", {"avg_time_to_throw": "ng_ttt", "aggressiveness": "ng_aggr",
                     "avg_intended_air_yards": "ng_pass_iay"}),
        ("receiving", {"avg_separation": "ng_sep", "avg_cushion": "ng_cushion",
                       "avg_yac_above_expectation": "ng_yacoe"}),
        ("rushing", {"percent_attempts_gte_eight_defenders": "ng_box8",
                     "rush_yards_over_expected_per_att": "ng_ryoe"}),
    ]:
        f = safe_read_parquet(RAW / f"nextgen_{st}.parquet")
        f = f[f.week > 0][["player_gsis_id", "season", "week"] + list(cols)]
        f = f.rename(columns={"player_gsis_id": "player_id", **cols})
        ng_frames.append(f)
    ng = ng_frames[0]
    for f in ng_frames[1:]:
        ng = ng.merge(f, on=["player_id", "season", "week"], how="outer")

    # game rows before the cutoff are identical in the full build (every
    # feature is shifted), so as-of mode can read the main file
    gf = safe_read_parquet(PROCESSED / "game_features.parquet")
    tl = _team_long(gf)
    if asof is not None:
        ps, dvp = ps[known_before(ps, asof)], dvp[known_before(dvp, asof)]
        cutoff = gf.loc[(gf.season == asof[0]) & (gf.week == asof[1]), "gameday"].min()
        depth_daily = depth_daily[depth_daily.day < cutoff]

    ps["team"] = ps["team"].map(norm_team)
    ps["opponent_team"] = ps["opponent_team"].map(norm_team)
    # Starting quarterback. `is_starter`: did this player start this game —
    # announced before kickoff, and what a passing prop is conditional on (the
    # bet is void if he does not play). `was_starter_last`: the incumbent,
    # i.e. the team's starter in its previous game.
    tl = tl.sort_values(["team", "gameday"])
    tl["prev_qb_id"] = tl.groupby("team")["team_qb_id"].shift(1)
    ps = ps.merge(tl.drop(columns=["opponent", "season", "week"]),
                  left_on=["game_id", "team"], right_on=["game_id", "team"],
                  how="left")
    ps["is_starter"] = (ps["player_id"] == ps["team_qb_id"]).astype(int)
    ps["was_starter_last"] = (ps["player_id"] == ps["prev_qb_id"]).astype(int)
    ps = ps.drop(columns=["prev_qb_id", "team_qb_id"])
    team_carries = ps.groupby(["game_id", "team"])["carries"].transform("sum")
    ps["carry_share"] = ps["carries"].fillna(0) / team_carries.where(team_carries > 0)
    ps = ps.merge(snaps.drop_duplicates(["game_id", "player_id"]),
                  on=["game_id", "player_id"], how="left")
    ps = ps.merge(usage, on=["game_id", "player_id"], how="left")
    for c in USAGE_COLS:
        ps[c] = ps[c].fillna(0)  # played but no high-leverage touches
    ps = ps.merge(ng, on=["player_id", "season", "week"], how="left")
    ps = ps.merge(depth, on=["player_id", "season", "week"], how="left")
    ps = ps.sort_values(["player_id", "gameday"]).reset_index(drop=True)
    # 2025+: the latest daily chart published strictly before game day
    left = (ps[["player_id", "gameday"]].assign(_row=ps.index)
            .dropna(subset=["gameday"]).sort_values("gameday"))
    left["gameday"] = left["gameday"].astype("datetime64[ns]")
    asof_rank = pd.merge_asof(
        left, depth_daily[["player_id", "day", "depth_rank"]], left_on="gameday",
        right_on="day", by="player_id", allow_exact_matches=False,
        tolerance=pd.Timedelta(days=21)).set_index("_row")["depth_rank"]
    ps["depth_rank"] = ps["depth_rank"].fillna(asof_rank)
    ps["depth_rank"] = ps.groupby("player_id")["depth_rank"].ffill()

    # Every rolling stat comes in two forms: the shifted one is the training
    # feature (prior games only); the unshifted `_cur` twin is the same number
    # as it will stand before the player's NEXT game, kept for the serving
    # snapshot. Both use the same minimum sample, so they are interchangeable.
    g = ps.groupby("player_id", sort=False)
    ps["snap_pct_ewm"] = g["offense_pct"].transform(
        lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
    ps["snap_pct_ewm_cur"] = g["offense_pct"].transform(
        lambda s: s.ewm(halflife=HALFLIFE, min_periods=2).mean())
    ps["snap_pct_last"] = g["offense_pct"].shift(1)
    roll_cols = []
    for c in STAT_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=HALFLIFE, min_periods=2).mean())
        roll_cols.append(f)
    for c in ["fantasy_points_ppr", "targets", "carries"]:
        f = f"l4_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.rolling(4, min_periods=2).mean())
        roll_cols.append(f)
    for c in USAGE_COLS + SHARE_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=HALFLIFE, min_periods=2).mean())
        roll_cols.append(f)
    for c in NG_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=8, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=8, min_periods=2).mean())
        roll_cols.append(f)
    roll_cols.append("depth_rank")
    # Did the player change teams since their previous game? Learnable
    # historically, and true at serving time for offseason movers.
    ps["prev_team"] = g["team"].shift(1)
    ps["team_changed"] = ((ps["team"] != ps["prev_team"])
                          & ps["prev_team"].notna()).astype(int)
    roll_cols += ["team_changed", "was_starter_last", "is_starter", "team_qb_new"]
    ps["fp_ppr_sd"] = g["fantasy_points_ppr"].transform(
        lambda s: s.shift(1).ewm(halflife=8, min_periods=3).std())
    ps["fp_ppr_sd_cur"] = g["fantasy_points_ppr"].transform(
        lambda s: s.ewm(halflife=8, min_periods=3).std())
    ps["career_games"] = g.cumcount()
    ps["season_games"] = ps.groupby(["player_id", "season"], sort=False).cumcount()
    ps["days_since_game"] = g["gameday"].diff().dt.days.clip(upper=120)
    roll_cols += ["fp_ppr_sd", "career_games", "season_games", "days_since_game",
                  "snap_pct_ewm", "snap_pct_last"]

    # Vacated usage: recent targets/carries of teammates who played the
    # previous team game (same season) but are absent from this one.
    vac_rows = []
    tg_cols = ["team", "season", "game_id", "gameday", "player_id",
               "r_targets_cur", "r_carries_cur"]
    for (team, season), grp in ps[tg_cols].groupby(["team", "season"], sort=False):
        prev: dict[str, tuple] = {}
        for gid, gg in sorted(grp.groupby("game_id"),
                              key=lambda kv: kv[1].gameday.iloc[0]):
            cur_ids = set(gg.player_id)
            vac_t = sum(v[0] for pid, v in prev.items() if pid not in cur_ids)
            vac_c = sum(v[1] for pid, v in prev.items() if pid not in cur_ids)
            vac_rows.append((team, gid, vac_t, vac_c))
            prev = {r.player_id: (np.nan_to_num(r.r_targets_cur),
                                  np.nan_to_num(r.r_carries_cur))
                    for r in gg.itertuples()}
    vac = pd.DataFrame(vac_rows, columns=["team", "game_id",
                                          "vacated_targets", "vacated_carries"])
    ps = ps.merge(vac, on=["team", "game_id"], how="left")
    roll_cols += ["vacated_targets", "vacated_carries"]

    # Opponent defense vs position: rolling stats allowed (shifted)
    dvp = dvp.sort_values(["defense", "position", "season", "week"]).reset_index(drop=True)
    gd = dvp.groupby(["defense", "position"], sort=False)
    dvp_feats = []
    for c in DVP_COLS:
        f = "dvp_fp_allowed" if c == "fp_ppr_allowed" else f"dvp_{c}"
        dvp[f] = gd[c].transform(
            lambda s: s.shift(1).ewm(halflife=6, min_periods=3).mean())
        dvp[f + "_cur"] = gd[c].transform(
            lambda s: s.ewm(halflife=6, min_periods=3).mean())
        dvp_feats.append(f)
    ps = ps.merge(
        dvp[["season", "week", "defense", "position"] + dvp_feats],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense", "position"], how="left")
    roll_cols += dvp_feats

    ctx_cols = TEAM_CTX + [f"opp_{c}" for c in TEAM_CTX] + \
               ["implied_total", "spread_line", "total_line", "is_dome", "wind_mph",
                "is_home", "team_spread"]
    feature_cols = roll_cols + ctx_cols + ["week"]

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = list(dict.fromkeys(
            ["player_id", "player_name", "position", "season", "week",
             "season_type", "game_id", "team", "opponent_team", "gameday"]
            + feature_cols + STAT_COLS))
        safe_to_parquet(ps[keep], out_dir / "player_features.parquet")

        cur = (ps.sort_values("gameday").groupby("player_id").tail(1))
        cur_cols = {f"r_{c}_cur": f"r_{c}" for c in STAT_COLS}
        cur_cols.update({f"l4_{c}_cur": f"l4_{c}" for c in ["fantasy_points_ppr", "targets", "carries"]})
        cur_cols["snap_pct_ewm_cur"] = "snap_pct_ewm"
        cur_cols["offense_pct"] = "snap_pct_last"
        cur_cols["fp_ppr_sd_cur"] = "fp_ppr_sd"
        cur_cols.update({f"r_{c}_cur": f"r_{c}" for c in USAGE_COLS + SHARE_COLS + NG_COLS})
        snap = cur[["player_id", "player_name", "position", "team", "season",
                    "gameday", "career_games", "depth_rank"]
                   + list(cur_cols)].rename(columns=cur_cols)
        # games already played in the season of his last game (his next game's
        # season_games, if it is in the same season)
        snap["season_games_played"] = cur["season_games"].values + 1
        # the newest depth chart, not the one from his last game
        if len(depth_daily):
            recent = depth_daily[depth_daily.day >= depth_daily.day.max() - pd.Timedelta(days=21)]
            latest = recent.groupby("player_id")["depth_rank"].last()
            snap["depth_rank"] = snap["player_id"].map(latest).fillna(snap["depth_rank"])
        snap["prior_team"] = snap["team"]
        if asof is None:
            # Apply current-roster teams: after an offseason move the player's
            # last game log still shows the old team, which would attach the
            # wrong team context (pace, QB quality, implied total).
            snap = snap.merge(current_roster(), on="player_id", how="left")
            moved = snap["roster_team"].notna() & (snap["roster_team"] != snap["team"])
            snap.loc[snap["roster_team"].notna(), "team"] = snap.loc[
                snap["roster_team"].notna(), "roster_team"]
            snap["team_changed"] = moved.astype(int)
            snap["on_roster"] = snap["roster_team"].notna().astype(int)
            print(f"  roster sync: {int(moved.sum())} players moved teams, "
                  f"{int(snap.on_roster.sum())} on a current roster")
        else:   # historical rosters are not kept: everyone stays where he last played
            snap["roster_status"], snap["team_changed"], snap["on_roster"] = "ACT", 0, 1
        safe_to_parquet(snap, out_dir / "player_current.parquet")

        # latest depth chart per team: the expected starters
        if len(depth_daily):
            last_day = depth_daily.groupby("team")["day"].transform("max")
            dcur = depth_daily[depth_daily.day == last_day]
        else:
            dcur = depth_daily
        safe_to_parquet(dcur[["team", "player_id", "position", "depth_rank", "day"]],
                        out_dir / "depth_current.parquet")

        dsnap = (dvp.sort_values(["season", "week"]).groupby(["defense", "position"]).tail(1)
                 [["defense", "position"] + [f + "_cur" for f in dvp_feats]]
                 .rename(columns={f + "_cur": f for f in dvp_feats}))
        safe_to_parquet(dsnap, out_dir / "dvp_current.parquet")
        print(f"player_features: {ps.shape}, snapshot: {snap.shape}")
    return ps, feature_cols


if __name__ == "__main__":
    build()
