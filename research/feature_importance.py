"""Which inputs matter most? Permutation importance on held-out seasons.

For each model, train on seasons before HOLD_FROM, then on the held-out
seasons shuffle one *group* of related features at a time and measure how much
the error rises. Shuffling whole groups matters: the home, away and
home-minus-away versions of one signal are near-duplicates, so shuffling one
alone lets the model lean on the others and understates the signal.

This is more honest than LightGBM's built-in "gain" (which is computed on the
training data and flatters noisy, high-cardinality features); gain is reported
alongside for comparison.

The game model served without Vegas lines is half an additive model (an
Explainable Boosting Machine), whose workings can be read off directly: one
curve per input, in points of margin. Its view is saved too ("additive"): how
many points each group typically moves a prediction, and the curves themselves.

Usage: python research/feature_importance.py
Writes reports/research/feature_importance.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nfl_engine.config import RESEARCH
from nfl_engine.models import game_models as GM
from nfl_engine.models import player_models as PM

HOLD_FROM = 2021
REPEATS = 5
PLAYER_TARGETS = ["fantasy_points_ppr", "passing_yards", "rushing_yards", "receiving_yards", "receptions"]


def game_group(f: str) -> str:
    if "elo" in f: return "Elo team rating"
    if "qb_" in f: return "Starting quarterback (efficiency, experience, change)"
    if f in ("spread_line", "total_line"): return "Vegas lines"
    if "points_for" in f: return "Points scored / allowed (opponent-adjusted)"
    if any(k in f for k in ("epa_play", "epa_l4", "epa_std")): return "Overall efficiency (EPA per play)"
    if any(k in f for k in ("pass_epa", "rush_epa")): return "Pass / rush efficiency splits"
    if "success" in f or "explosive" in f or "third_down" in f: return "Success, explosive-play and 3rd-down rates"
    if any(k in f for k in ("sack", "to_rate", "fum_luck")): return "Sacks, turnovers, fumble luck"
    if "st_epa" in f or "plays" in f: return "Special teams and pace"
    if "coach" in f: return "Coaching (tenure, record)"
    if any(k in f for k in ("rest", "travel", "tz_", "games_played", "div_game", "is_thursday", "is_monday",
                            "is_primetime", "is_playoff")): return "Rest, travel and schedule spot"
    if any(k in f for k in ("temp", "wind", "is_dome", "is_turf", "is_denver")): return "Venue and weather"
    return "Other"


def player_group(f: str) -> str:
    if f in ("implied_total", "spread_line", "total_line", "team_spread", "is_home"): return "Vegas lines and game script"
    if f.startswith("dvp_") or f.startswith("opp_"): return "Opponent defense"
    if f.startswith("r_ng_"): return "NextGen tracking stats"
    if f in ("r_rz_targets", "r_ez_targets", "r_rz_carries", "r_gl_carries"): return "Red-zone and goal-line usage"
    if f in ("snap_pct_ewm", "snap_pct_last", "depth_rank", "was_starter_last", "is_starter", "r_target_share",
             "r_air_yards_share", "r_wopr", "r_carry_share", "r_targets", "r_carries", "l4_targets", "l4_carries",
             "r_attempts", "r_receiving_air_yards"): return "Role and volume (snaps, shares, depth chart)"
    if f.startswith("vacated_") or f == "team_changed": return "Vacated usage and team change"
    if f.startswith(("r_", "l4_")) or f == "fp_ppr_sd": return "Player's recent production"
    if f.startswith(("off_", "allowed_")) or f in ("team_qb_epa", "team_qb_new"): return "Own team's offense and QB"
    if f.startswith("pos_"): return "Position"
    if f in ("week", "season_games", "career_games", "days_since_game", "is_dome", "wind_mph"): return "Experience, schedule, venue"
    return "Other"


def permutation(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], seed=7) -> dict:
    base = float(np.abs(model.predict(X) - y).mean())
    rng = np.random.default_rng(seed)
    out = {}
    for g, cols in groups.items():
        rises = []
        for _ in range(REPEATS):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]          # shuffle the group together
            rises.append(float(np.abs(model.predict(Xp) - y).mean()) - base)
        out[g] = dict(mae_rise=float(np.mean(rises)), pct=float(100 * np.mean(rises) / base), n_features=len(cols))
    return dict(base_mae=base, groups=dict(sorted(out.items(), key=lambda kv: -kv[1]["mae_rise"])))


def grouped(cols, fn):
    g: dict[str, list[str]] = {}
    for c in cols:
        g.setdefault(fn(c), []).append(c)
    return g


def gain_share(model, cols, fn) -> dict:
    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    s = (imp / imp.sum()).groupby(imp.index.map(fn)).sum().sort_values(ascending=False)
    return {k: float(v) for k, v in s.items()}


def top_single(model, gbm, X, y, cols, k=12, seed=7) -> list:
    """Largest single-feature permutation rises among the top-gain features."""
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    base = float(np.abs(model.predict(X) - y).mean())
    rng = np.random.default_rng(seed)
    rows = []
    for c in imp.index[:30]:
        r = []
        for _ in range(3):
            Xp = X.copy(); Xp[c] = X[c].values[rng.permutation(len(X))]
            r.append(float(np.abs(model.predict(Xp) - y).mean()) - base)
        rows.append((c, float(np.mean(r)), float(imp[c] / imp.sum())))
    rows.sort(key=lambda t: -t[1])
    return [dict(feature=c, mae_rise=m, gain_share=g) for c, m, g in rows[:k]]


def additive_view(ebm, X: pd.DataFrame, fn, n_curves: int = 6) -> dict:
    """Read the additive model directly. `group_points`: how far each group
    of inputs typically moves a held-out prediction (mean absolute points);
    `top_points`: the same for single inputs; `curves`: the fitted curve of
    the most influential inputs (bin edges and the points each bin adds)."""
    contrib = pd.DataFrame(ebm.eval_terms(X), columns=ebm.term_names_)
    contrib = contrib - contrib.mean()
    groups = contrib.T.groupby(contrib.columns.map(fn)).sum().T.abs().mean().sort_values(ascending=False)
    single = contrib.abs().mean().sort_values(ascending=False)
    curves = {}
    for name in single.index[:n_curves]:
        i = ebm.term_names_.index(name)
        f = ebm.term_features_[i][0]
        lo, hi = ebm.feature_bounds_[f]
        curves[name] = dict(edges=[float(lo), *map(float, ebm.bins_[f][0]), float(hi)],
                            points=[float(v) for v in ebm.term_scores_[i][1:-1]],
                            p05=float(X[name].quantile(0.05)), p95=float(X[name].quantile(0.95)))
    return dict(group_points={k: float(v) for k, v in groups.items()},
                top_points={k: float(v) for k, v in single.head(12).items()}, curves=curves)


def main():
    report = {"hold_from": HOLD_FROM}

    df = GM.load_features()
    played = df.dropna(subset=["result"])
    for tag, market in (("game_margin", False), ("game_margin_with_lines", True), ("game_total", False)):
        target = "total" if tag == "game_total" else "result"
        cols = GM.feature_cols(df, market)
        tr, te = played[played.season < HOLD_FROM], played[played.season >= HOLD_FROM]
        # the model as served: LightGBM, blended with the additive model when there are no lines
        m = GM.fit(tr, cols, target, HOLD_FROM - 1, GM.EBM_WEIGHT["mkt" if market else "pure"])
        gbm, ebm = (m.models if isinstance(m, GM.Blend) else (m, None))
        y = te[target].to_numpy(float)
        r = permutation(m, te[cols], y, grouped(cols, game_group))
        r["gain"] = gain_share(gbm, cols, game_group)
        r["top_features"] = top_single(m, gbm, te[cols], y, cols)
        if ebm is not None:
            r["additive"] = additive_view(ebm, te[cols], game_group)
        r["n_test"] = int(len(te))
        assert "Other" not in r["groups"], [c for c in cols if game_group(c) == "Other"]
        report[tag] = r
        print(tag, {k: round(v["pct"], 2) for k, v in list(r["groups"].items())[:6]}, flush=True)

    pdf, pcols = PM.load()
    for t in PLAYER_TARGETS:
        d = PM.subset(pdf, t)
        tr, te = d[d.season < HOLD_FROM], d[(d.season >= HOLD_FROM) & (d.season_type == "REG")]
        m = PM.fit_one(tr, pcols, t)
        y = te[t].to_numpy(float)
        r = permutation(m, te[pcols], y, grouped(pcols, player_group))
        r["gain"] = gain_share(m, pcols, player_group)
        r["top_features"] = top_single(m, m, te[pcols], y, pcols)
        assert "Other" not in r["groups"], [c for c in pcols if player_group(c) == "Other"]
        r["n_test"] = int(len(te))
        report[t] = r
        print(t, {k: round(v["pct"], 2) for k, v in list(r["groups"].items())[:6]}, flush=True)

    RESEARCH.mkdir(exist_ok=True)
    (RESEARCH / "feature_importance.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
