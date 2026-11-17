"""Serving layer: assemble feature rows for future games/players and run
the trained models.

The one rule here: a served row must be built exactly the way the training
row for that same game would be. Everything comes from "current snapshot"
tables written by the feature pipelines (data/processed/*_current.parquet)
plus the real scheduled game (venue, rest, posted lines), the latest depth
chart (who is expected to start) and what is known about availability
(injured reserve, the week's injury report). tests/test_parity.py rebuilds
the snapshots as of a past week and checks served rows against training rows.
"""
import json
from functools import cached_property

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_engine.config import CURRENT_SEASON, MODELS_STORE, PROCESSED, RAW, TEAMS, norm_team
from nfl_engine.data.store import connect
from nfl_engine.features.elo import HOME_ADV, MEAN, SEASON_REGRESS, current_ratings, elo_win_prob
from nfl_engine.features.game_features import QB_METRICS, known_before, venue_flags
from nfl_engine.features.player_features import TEAM_CTX
from nfl_engine.features.stadiums import travel_km, tz_diff
from nfl_engine.io_utils import read_parquet as safe_read_parquet
from nfl_engine.models import player_models as PM

RESULT_COLS = ["home_score", "away_score", "result", "total", "overtime"]


class Engine:
    """Lazy-loading bundle of snapshots + models for predicting future games.

    `Engine()` serves the live state. `Engine(processed=dir, asof=(season,
    week))` serves from an as-of rebuild: results from that week on are
    hidden, so it predicts that week as the live system would have.
    """

    def __init__(self, processed=PROCESSED, asof: tuple[int, int] | None = None):
        self.processed = processed
        self.asof = asof
        self.season = asof[0] if asof else CURRENT_SEASON
        self._cache: dict = {}

    def _hide_future(self, g: pd.DataFrame) -> pd.DataFrame:
        if self.asof is not None:
            g.loc[~known_before(g, self.asof), [c for c in RESULT_COLS if c in g.columns]] = np.nan
        return g

    # ---------------- snapshots ----------------

    @cached_property
    def games(self) -> pd.DataFrame:
        con = connect()
        g = con.execute("SELECT * FROM games ORDER BY gameday, game_id").df()
        con.close()
        g["gameday"] = pd.to_datetime(g["gameday"])
        return self._hide_future(g)

    @cached_property
    def team_current(self) -> pd.DataFrame:
        return safe_read_parquet(self.processed / "team_current.parquet").set_index("team")

    @cached_property
    def qb_current(self) -> pd.DataFrame:
        return safe_read_parquet(self.processed / "qb_current.parquet").set_index("qb_id")

    @cached_property
    def player_current(self) -> pd.DataFrame:
        pc = safe_read_parquet(self.processed / "player_current.parquet")
        pc["gameday"] = pd.to_datetime(pc["gameday"])
        return pc

    @cached_property
    def dvp_current(self) -> pd.DataFrame:
        return safe_read_parquet(self.processed / "dvp_current.parquet")

    @cached_property
    def depth_current(self) -> pd.DataFrame:
        return safe_read_parquet(self.processed / "depth_current.parquet")

    @cached_property
    def elo_ratings(self) -> dict:
        return current_ratings(self.games)

    def elo(self, team: str, season: int) -> float:
        """Pregame rating for a game in `season`: the rating after the team's
        last game, pulled a third of the way to average if that game was in an
        earlier season (as the training ratings are)."""
        r = self.elo_ratings.get(team, MEAN)
        g = self.games
        last = g.loc[((g.home_team == team) | (g.away_team == team)) & g.result.notna(), "season"].max()
        return MEAN + (r - MEAN) * (1 - SEASON_REGRESS) if pd.notna(last) and last != season else r

    @cached_property
    def venue(self) -> pd.DataFrame:
        """is_dome / is_turf per scheduled game, blanks filled as in training."""
        return venue_flags(self.games).set_index("game_id")

    @cached_property
    def _coach_games(self) -> pd.DataFrame:
        g = self.games.dropna(subset=["result"])
        h = g[["gameday", "home_team", "home_coach", "result"]].rename(
            columns={"home_team": "team", "home_coach": "coach"})
        h["won"] = (h.result > 0).astype(float)
        a = g[["gameday", "away_team", "away_coach", "result"]].rename(
            columns={"away_team": "team", "away_coach": "coach"})
        a["won"] = (a.result < 0).astype(float)
        return pd.concat([h, a]).sort_values("gameday")

    def coach_context(self, team: str, coach: str | None = None) -> dict:
        """Head-coach inputs. `coach` is the name on the schedule (so a new
        hire counts from his first game); default: the team's last coach."""
        rows = self._coach_games
        if coach is None or pd.isna(coach):
            mine = rows[rows.team == team]
            if mine.empty:
                return {}
            coach = mine.coach.iloc[-1]
        cg = rows[rows.coach == coach]
        tenure = int((cg.team == team).sum())
        return dict(coach_tenure_games=float(tenure), coach_career_games=float(len(cg)),
                    # training needs 8 prior games before it trusts a win rate
                    coach_career_winpct=float(cg.won.mean()) if len(cg) >= 8 else np.nan,
                    coach_new_team=float(tenure < 17), coach_rookie=float(len(cg) < 17))

    @cached_property
    def schedule(self) -> pd.DataFrame:
        """This season's schedule with posted lines, for context lookup."""
        s = safe_read_parquet(RAW / "schedules.parquet")
        s = s[s.season == self.season].copy()
        s["home_team"] = s.home_team.map(norm_team)
        s["away_team"] = s.away_team.map(norm_team)
        s["gameday"] = pd.to_datetime(s.gameday)
        return self._hide_future(s)

    @cached_property
    def _sched_index(self) -> dict:
        """(team, week) -> (schedule row, is_home)."""
        idx = {}
        for r in self.schedule.itertuples():
            idx[(r.home_team, int(r.week))] = (r, True)
            idx[(r.away_team, int(r.week))] = (r, False)
        return idx

    def game_context(self, team: str, week: int | None = None,
                     opponent: str | None = None):
        """The scheduled game for `team` in `week` (or its next unplayed game
        against `opponent`). Returns (schedule row, team_is_home) or None."""
        team = norm_team(team)
        if week is not None:
            hit = self._sched_index.get((team, int(week)))
            if hit and (opponent is None or norm_team(opponent) in
                        (hit[0].home_team, hit[0].away_team)):
                return hit
            return None
        if opponent is None:
            return None
        opp = norm_team(opponent)
        s = self.schedule
        m = s[(((s.home_team == team) & (s.away_team == opp))
               | ((s.home_team == opp) & (s.away_team == team))) & s.result.isna()]
        if m.empty:
            return None
        r = next(m.sort_values("gameday").itertuples())
        return r, r.home_team == team

    # ---------------- who plays ----------------

    @cached_property
    def _qb_starts(self) -> pd.DataFrame:
        """One row per (team, played game) with that game's starting QB."""
        g = self.games.dropna(subset=["result"])
        h = g[["gameday", "home_team", "home_qb_id"]].rename(columns={"home_team": "team", "home_qb_id": "qb_id"})
        a = g[["gameday", "away_team", "away_qb_id"]].rename(columns={"away_team": "team", "away_qb_id": "qb_id"})
        return pd.concat([h, a]).sort_values("gameday")

    @cached_property
    def last_starters(self) -> dict:
        """team -> the QB who started its most recent game (the incumbent)."""
        return self._qb_starts.groupby("team").tail(1).set_index("team")["qb_id"].to_dict()

    def ruled_out(self, week: int | None) -> set:
        """Players listed Out or Doubtful on that week's injury report (empty
        until the report is published, usually Wednesday)."""
        if week is None:
            return set()
        if "_inj" not in self._cache:
            i = safe_read_parquet(RAW / "injuries.parquet")
            i = i[(i.season == self.season) & i.report_status.isin(["Out", "Doubtful"])]
            self._cache["_inj"] = i.groupby("week")["gsis_id"].agg(set).to_dict()
        return self._cache["_inj"].get(int(week), set())

    def unavailable(self, week: int | None = None) -> set:
        """Everyone known to be out: injured reserve / off the roster, plus the
        week's injury report."""
        if "_gone" not in self._cache:
            pc = self.player_current
            self._cache["_gone"] = set(pc.loc[(pc.on_roster == 0) | (pc.roster_status == "RES"), "player_id"])
        return self._cache["_gone"] | self.ruled_out(week)

    def expected_starters(self, week: int | None = None) -> dict:
        """team -> the QB expected to start: the first available QB on the
        latest depth chart, else the incumbent. Read the day before a game the
        chart named the starter 91% of the time in 2025-26 (the incumbent rule:
        87%); when the two disagreed the chart was right 63% to 31%."""
        key = ("exp", week)
        if key not in self._cache:
            gone = self.unavailable(week)
            qbs = self.depth_current
            qbs = qbs[qbs.position == "QB"].sort_values("depth_rank")
            out = {}
            for team in TEAMS:
                listed = [p for p in qbs.loc[qbs.team == team, "player_id"] if p not in gone]
                out[team] = listed[0] if listed else self.last_starters.get(team)
            self._cache[key] = out
        return self._cache[key]

    def qb_context(self, team: str, week: int | None = None, starter_id: str | None = None) -> dict:
        """The starting-QB inputs for `team`, as training builds them: the
        starter's rolling profile, his starts with this team, his career games,
        and whether he differs from last game's starter. `starter_id` overrides
        the expected starter (e.g. a book has posted his passing line)."""
        qid = starter_id or self.expected_starters(week).get(team)
        ctx = dict(qb_id=qid, qb_new=float(qid != self.last_starters.get(team)),
                   qb_career_dropbacks=0.0, qb_games=0.0)
        st = self._qb_starts
        ctx["qb_starts_team"] = float(((st.team == team) & (st.qb_id == qid)).sum())
        if qid in self.qb_current.index:
            q = self.qb_current.loc[qid]
            ctx.update({f"qb_{m}_ewm": float(q[f"qb_{m}"]) for m in QB_METRICS if pd.notna(q[f"qb_{m}"])})
            ctx["qb_career_dropbacks"] = float(q["career_dropbacks"])
            ctx["qb_games"] = float(q["qb_games"])
        return ctx

    def vacated(self, team: str, week: int | None = None) -> tuple[float, float]:
        """(targets, carries) left behind by players who played the team's
        last game this season and are now known to be out — training's
        definition of vacated usage, with "absent" replaced by what can be
        known beforehand."""
        key = ("vac", team, week)
        if key not in self._cache:
            g = self.games
            mine = g[((g.home_team == team) | (g.away_team == team))
                     & (g.season == self.season) & g.result.notna()]
            out = (0.0, 0.0)
            if len(mine):
                pc = self.player_current
                played = pc[(pc.prior_team == team) & (pc.gameday == mine.gameday.max())]
                gone = played[played.player_id.isin(self.unavailable(week)) | (played.team != team)]
                out = (float(gone.r_targets.fillna(0).sum()), float(gone.r_carries.fillna(0).sum()))
            self._cache[key] = out
        return self._cache[key]

    def _games_before(self, team: str, when: pd.Timestamp, season: int) -> int:
        """Team games already played this season before `when` (training's
        games_played_season is the same within-season count)."""
        g = self.games
        m = (g.season == season) & (g.gameday < when) & \
            ((g.home_team == team) | (g.away_team == team))
        return int(m.sum())

    def _game_meta(self, tag: str) -> dict:
        return json.loads((MODELS_STORE / f"game_meta_{tag}.json").read_text())

    @cached_property
    def game_models(self) -> dict:
        out = {}
        for tag in ("pure", "mkt"):
            cal_path = MODELS_STORE / f"game_cal_{tag}.joblib"
            out[tag] = dict(
                margin=joblib.load(MODELS_STORE / f"game_margin_{tag}.joblib"),
                total=joblib.load(MODELS_STORE / f"game_total_{tag}.joblib"),
                cal=joblib.load(cal_path) if cal_path.exists() else None,
                meta=self._game_meta(tag))
        return out

    @cached_property
    def player_models(self) -> dict:
        store = MODELS_STORE / "player"
        meta = json.loads((store / "meta.json").read_text())
        return {"meta": meta,
                "models": {t: joblib.load(store / f"{t}.joblib")
                           for t in meta["targets"]}}

    def win_prob(self, margin, tag: str = "pure"):
        """Predicted home margin -> calibrated home win probability."""
        gm = self.game_models[tag]
        p = norm.cdf(np.asarray(margin, dtype=float) / gm["meta"]["sigma_margin"])
        return gm["cal"].predict(p) if gm["cal"] is not None else p

    # ---------------- feature assembly ----------------

    def game_feature_row(self, home: str, away: str, neutral: bool = False,
                         ctx=None) -> dict:
        """Feature row for a future game. `ctx` is the scheduled game's
        schedule row; without it (a hypothetical matchup) the game is treated
        as each team's next game with typical rest and conditions."""
        home, away = norm_team(home), norm_team(away)
        tc = self.team_current
        if ctx is not None:
            neutral = ctx.location == "Neutral"
            when, season, week = ctx.gameday, int(ctx.season), int(ctx.week)
        else:
            when, season, week = pd.Timestamp.now().normalize(), self.season, None
        row: dict = {}
        for side, team in (("home", home), ("away", away)):
            if team in tc.index:
                for c, v in tc.loc[team].items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        row[f"{side}_{c}"] = float(v)
                if int(tc.loc[team]["season"]) != season:
                    # season-to-date form does not exist before a season's first game
                    row[f"{side}_off_epa_std"] = row[f"{side}_allowed_epa_std"] = np.nan
            coach = getattr(ctx, f"{side}_coach", None) if ctx is not None else None
            for c, v in self.coach_context(team, coach).items():
                row[f"{side}_{c}"] = v
            for c, v in self.qb_context(team, week).items():
                if c != "qb_id":
                    row[f"{side}_{c}"] = v
            # rest = days since the team's previous game (scheduled or played),
            # clipped at 30 exactly as in training
            g = self.games
            prev = g[((g.home_team == team) | (g.away_team == team))
                     & (g.gameday < when)].gameday.max()
            rest = 7.0 if pd.isna(prev) or ctx is None else float(min((when - prev).days, 30))
            row[f"{side}_rest_days"] = rest
            row[f"{side}_games_played_season"] = float(self._games_before(team, when, season))
        row["home_elo_pre"] = self.elo(home, season)
        row["away_elo_pre"] = self.elo(away, season)
        hfa = 0.0 if neutral else HOME_ADV
        row["elo_diff"] = row["home_elo_pre"] - row["away_elo_pre"] + hfa
        row["away_travel_km"] = travel_km(away, home, season) if not neutral else 800.0
        row["away_tz_diff"] = tz_diff(away, home, season)
        row["is_denver"] = float(home == "DEN" and not neutral)
        if ctx is not None:
            row["home_rest"] = float(ctx.home_rest) if pd.notna(ctx.home_rest) else 7.0
            row["away_rest"] = float(ctx.away_rest) if pd.notna(ctx.away_rest) else 7.0
            row["div_game"] = float(ctx.div_game) if pd.notna(ctx.div_game) else 0.0
            row["is_playoff"] = float(ctx.game_type != "REG")
            row["is_thursday"] = float(ctx.weekday == "Thursday")
            row["is_monday"] = float(ctx.weekday == "Monday")
            row["is_primetime"] = float((ctx.gametime or "13:00") >= "20:00")
        else:
            row["home_rest"] = row["away_rest"] = 7.0
            row["div_game"] = row["is_playoff"] = 0.0
            row["is_thursday"] = row["is_monday"] = row["is_primetime"] = 0.0
        row["rest_diff"] = row["home_rest"] - row["away_rest"]
        row.update(self.venue_row(home, ctx))

        meta = self.game_models["pure"]["meta"]
        for f in meta["features"]:
            if f.startswith("d_"):
                b = f[2:]
                hv, av = row.get(f"home_{b}"), row.get(f"away_{b}")
                row[f] = (hv - av) if hv is not None and av is not None else np.nan
        return row

    def venue_row(self, home: str, ctx=None) -> dict:
        """Roof, surface and weather for a game at `home`. Weather that has
        not been published stays missing — training has the same gaps, and
        the models treat "unknown" as its own case rather than as an average
        day."""
        if ctx is not None and ctx.game_id in self.venue.index:
            v = self.venue.loc[ctx.game_id]
        else:                                  # hypothetical: the team's latest home game
            g = self.games
            ids = g.loc[g.home_team == home, "game_id"]
            v = self.venue.loc[ids.iloc[-1]] if len(ids) else pd.Series({"is_dome": 0, "is_turf": 0})
        dome = float(v["is_dome"])
        temp = float(ctx.temp) if ctx is not None and pd.notna(ctx.temp) else np.nan
        wind = float(ctx.wind) if ctx is not None and pd.notna(ctx.wind) else np.nan
        return dict(is_dome=dome, is_turf=float(v["is_turf"]),
                    temp_f=70.0 if dome else temp, wind_mph=0.0 if dome else wind)

    def predict_game(self, home: str, away: str, neutral: bool = False,
                     ctx=None, week: int | None = None) -> dict:
        """Pure-model prediction. Uses the scheduled game's context when the
        matchup is on the schedule (`ctx`, or looked up by `week`)."""
        home, away = norm_team(home), norm_team(away)
        if ctx is None and not neutral:
            hit = self.game_context(home, week, away)
            if hit and hit[1]:
                ctx = hit[0]
        key = ("game", home, away, neutral, getattr(ctx, "game_id", None))
        if key in self._cache:
            return self._cache[key]
        row = self.game_feature_row(home, away, neutral, ctx)
        gm = self.game_models["pure"]
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in gm["meta"]["features"]}])
        margin = float(gm["margin"].predict(X)[0])
        total = float(gm["total"].predict(X)[0])
        p_home = float(self.win_prob([margin])[0])
        elo_p = float(elo_win_prob(row["elo_diff"]))
        self._cache[key] = dict(
            home=home, away=away, neutral=neutral,
            pred_margin=round(margin, 1), pred_total=round(total, 1),
            home_win_prob=round(p_home, 3), away_win_prob=round(1 - p_home, 3),
            home_score=round((total + margin) / 2, 1),
            away_score=round((total - margin) / 2, 1),
            elo_home=round(row["home_elo_pre"]), elo_away=round(row["away_elo_pre"]),
            elo_win_prob=round(elo_p, 3),
            sigma_margin=round(gm["meta"]["sigma_margin"], 1),
        )
        return self._cache[key]

    def player_feature_row(self, player_row: pd.Series, opponent: str | None,
                           week: int = 8, home: bool | None = None,
                           starter: bool | None = None) -> dict:
        """Feature row for one player's future game, mirroring how training
        rows are built. `starter` forces the starting-QB flag for this player
        (True when a book has posted his passing line); by default it follows
        the expected starter from the depth chart."""
        team, pid = player_row.get("team"), player_row.get("player_id")
        row = {c: float(v) for c, v in player_row.items()
               if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v)}
        ctx = None
        if opponent:
            hit = (self.game_context(team, week, opponent)
                   or self.game_context(team, None, opponent))
            if hit is not None:
                ctx, home = hit
                week = int(ctx.week)
        season = int(ctx.season) if ctx is not None else self.season
        row["week"] = float(week)
        # the snapshot's career_games counts games before the last one played
        row["career_games"] = float(player_row.get("career_games", 0) or 0) + 1
        last = player_row.get("gameday")
        if ctx is not None and pd.notna(last):
            row["days_since_game"] = float(min((ctx.gameday - pd.Timestamp(last)).days, 120))
        else:
            row["days_since_game"] = 7.0
        # games he will have played this season before this one: those already
        # played plus the team's games still to come before it
        played = (float(player_row.get("season_games_played", 0) or 0)
                  if int(player_row.get("season", -1)) == season else 0.0)
        if ctx is not None:
            g = self.games
            done = int((((g.home_team == team) | (g.away_team == team))
                        & (g.season == season) & g.result.notna()).sum())
            row["season_games"] = played + max(self._games_before(team, ctx.gameday, season) - done, 0)
        else:
            row["season_games"] = played
        for p in ["QB", "RB", "WR", "TE"]:
            row[f"pos_{p}"] = float(player_row["position"] == p)
        tc = self.team_current
        if team in tc.index:
            for c in TEAM_CTX:
                if c in tc.columns:
                    row[c] = float(tc.loc[team][c])
        # starting quarterback: his own flags, and the QB his teammates play with
        is_qb = player_row["position"] == "QB"
        qb = self.qb_context(team, week, pid if (starter and is_qb) else None)
        if "qb_epa_dropback_ewm" in qb:
            row["team_qb_epa"] = qb["qb_epa_dropback_ewm"]
        row["team_qb_new"] = qb["qb_new"]
        row["was_starter_last"] = float(pid == self.last_starters.get(team))
        row["is_starter"] = float(pid == qb["qb_id"]) if starter is None or not is_qb else float(starter)
        if season == self.season:
            row["vacated_targets"], row["vacated_carries"] = self.vacated(team, week)
        else:
            row["vacated_targets"] = row["vacated_carries"] = 0.0
        if opponent:
            opponent = norm_team(opponent)
            if opponent in tc.index:
                for c in TEAM_CTX:
                    if c in tc.columns:
                        row[f"opp_{c}"] = float(tc.loc[opponent][c])
            dvp = self.dvp_current
            hit = dvp[(dvp.defense == opponent) & (dvp.position == player_row["position"])]
            if not hit.empty:
                for c in hit.columns:
                    if c.startswith("dvp_"):
                        row[c] = float(hit.iloc[0][c])
            # Market context, as in training: spread_line is the HOME team's
            # expected margin and total_line the game total. Posted lines are
            # used when the book has them; otherwise the model's own game
            # prediction stands in.
            if home is None:
                home = True  # hypothetical matchup with unknown venue
            h, a = (team, opponent) if home else (opponent, team)
            g = self.predict_game(h, a, ctx=ctx)
            spread = (float(ctx.spread_line) if ctx is not None and pd.notna(ctx.spread_line)
                      else g["pred_margin"])
            total = (float(ctx.total_line) if ctx is not None and pd.notna(ctx.total_line)
                     else g["pred_total"])
            row["spread_line"] = spread
            row["total_line"] = total
            row["implied_total"] = (total + (spread if home else -spread)) / 2
            row["is_home"] = float(home)
            row["team_spread"] = spread if home else -spread
            venue = self.venue_row(h, ctx)
            row["is_dome"], row["wind_mph"] = venue["is_dome"], venue["wind_mph"]
        return row

    def resolve_player(self, name: str) -> pd.Series | None:
        pc = self.player_current
        q = name.lower().strip()
        exact = pc[pc.player_name.str.lower() == q]
        pool = exact if not exact.empty else pc[pc.player_name.str.lower().str.contains(q, regex=False)]
        if pool.empty:
            import difflib
            names = pc.player_name.str.lower().tolist()
            close = difflib.get_close_matches(q, names, n=1, cutoff=0.75)
            if close:
                pool = pc[pc.player_name.str.lower() == close[0]]
        if pool.empty:
            return None
        return pool.sort_values(["season", "gameday"]).iloc[-1]

    def _player_X(self, row: dict) -> pd.DataFrame:
        return pd.DataFrame([{f: row.get(f, np.nan) for f in self.player_models["meta"]["features"]}])

    def project_player(self, name: str, opponent: str | None = None,
                       week: int = 8) -> dict | None:
        """Fantasy projection (mean, floor, median, ceiling) in each scoring."""
        p = self.resolve_player(name)
        if p is None:
            return None
        models = self.player_models["models"]
        X = self._player_X(self.player_feature_row(p, opponent, week))
        preds = {t: PM.predict(models[t], X, mean_only=True) for t in PM.PPR_WEIGHTS}
        out = dict(player=p.player_name, position=p.position, team=p.team,
                   last_seen_season=int(p.season), opponent=opponent)
        for target, ppr in (("fantasy_points_ppr", True), ("fantasy_points", False)):
            preds[target] = PM.predict(models[target], X)
            q = preds[target]["q"][0]
            out[target] = dict(mean=round(float(PM.fantasy_mean(preds, [p.position], ppr)[0]), 2),
                               floor=round(float(q[0]), 1), median=round(float(q[2]), 1),
                               ceiling=round(float(q[4]), 1))
        out["fantasy_points_half"] = dict(
            mean=round((out["fantasy_points_ppr"]["mean"] + out["fantasy_points"]["mean"]) / 2, 2))
        return out

    def prop_probability(self, name: str, stat: str, line: float,
                         opponent: str | None = None, week: int = 8,
                         starter: bool | None = None) -> dict | None:
        """P(over / under `line`) for one player and stat, with the
        distribution it came from."""
        p = self.resolve_player(name)
        if p is None:
            return None
        pm = self.player_models
        if stat not in pm["models"]:
            return {"error": f"unsupported stat '{stat}'",
                    "supported": list(pm["models"])}
        if p.position not in pm["meta"]["targets"][stat]["positions"]:
            return {"error": f"{p.player_name} is a {p.position}; "
                            f"'{stat}' props cover {pm['meta']['targets'][stat]['positions']}"}
        pred = PM.predict(pm["models"][stat], self._player_X(self.player_feature_row(p, opponent, week, starter=starter)))
        prob = float(PM.prob_over(pred, line, stat in PM.SPLIT_TAILS)[0])
        if "rate" in pred:
            dist = {"mean": float(pred["rate"][0])}
        else:
            dist = {"mean": float(pred["mean"][0])} | {
                f"p{int(q * 100)}": float(v) for q, v in zip(PM.QUANTILES, pred["q"][0])}
        return dict(player=p.player_name, position=p.position, team=p.team,
                    stat=stat, line=line, opponent=opponent,
                    prob_over=round(prob, 3), prob_under=round(1 - prob, 3),
                    distribution={k: round(v, 2) for k, v in dist.items()})


ENGINE = Engine()
