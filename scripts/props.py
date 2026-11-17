"""Player-prop tracker: record sportsbook prop lines next to the model's
probability before kickoff, then grade them against what happened.

  python scripts/props.py snapshot   run once a week before the Sunday slate
                                     (ideally Saturday, when lines are posted)
  python scripts/props.py grade      settle every saved snapshot; the weekly
                                     refresh runs this automatically

snapshot — The Odds API (free plan, 500 credits/month; 6 prop markets x ~15
games ~ 90 credits a run). The key is read from $ODDS_API_KEY or
~/.config/chalkline/odds_api_key and is never printed or written elsewhere.
Writes to reports/prop_snapshots/:
  lines_<date>.csv        every book's line and price
  model_<date>.csv        consensus line, de-vigged market probability, model
                          probability, best price across books, EV
  candidates_<date>.csv   the primary strategy's bets for the week
  --reprice               re-run the model on today's saved lines (no credits)

grade — matches each saved prop to the player's stat line and settles the
model's side at the best price recorded before kickoff, into graded.csv.
Two records are kept apart:
  primary      the strategy fixed in advance (PRIMARY_MARKETS: QB volume
               props, model side, +EV at best price). The only record that
               counts as evidence. Stopping rule, also fixed in advance: judge
               once at 250 graded bets or season end, whichever comes first;
               an edge is claimed only if the hit rate beats the 52.4%
               break-even at one-sided p < 0.05 AND ROI at the recorded prices
               is positive.
  exploratory  every other market the model leaned on. For ideas, never for
               claims: testing many subsets finds winners by luck.

Until the primary record passes that test, no prop edge is proven.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nfl_engine.config import RAW, norm_team
from nfl_engine.io_utils import read_parquet
from nfl_engine.odds import BREAKEVEN, american_to_decimal as dec
from nfl_engine.query.tools import resolve_team

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = {"player_pass_yds": "passing_yards", "player_pass_completions": "completions",
           "player_pass_attempts": "attempts", "player_rush_yds": "rushing_yards",
           "player_reception_yds": "receiving_yards", "player_receptions": "receptions"}
# Anytime-TD props were captured in the first snapshot (2026-10-20) and are
# still graded if present, but no longer fetched: the TD models cannot beat
# the TD-prop margin (research/prop_edge_study.py), and the credits are better
# spent on the QB volume props that tested best.
ALL_MARKETS = dict(MARKETS, player_anytime_td="anytime_td")

# The one strategy that will be judged, fixed before any props were graded:
# QB volume props, the model's side, only when it has positive expected value
# at the best price across books. Anything else found later is exploratory.
PRIMARY_MARKETS = {"player_pass_yds", "player_pass_completions", "player_pass_attempts"}
# A book only posts a passing line for the quarterback it expects to start, so
# these stats are priced with the starter flag set whatever the depth chart says.
STARTER_STATS = {"passing_yards", "completions", "attempts"}
OUT = ROOT / "reports" / "prop_snapshots"


def api_key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if not k:
        f = Path.home() / ".config" / "chalkline" / "odds_api_key"
        k = f.read_text().strip() if f.exists() else ""
    if not k:
        sys.exit("no Odds API key: set ODDS_API_KEY or ~/.config/chalkline/odds_api_key")
    return k


def get(url: str) -> tuple[object, dict]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read()), {k.lower(): v for k, v in r.headers.items()}


def fetch(days: int, min_credits: int) -> pd.DataFrame:
    key = api_key()
    events, h = get(f"{API}/events?apiKey={key}")
    now = dt.datetime.now(dt.timezone.utc)
    soon = [e for e in events
            if 0 < (dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) - now).days < days]
    remaining = int(h.get("x-requests-remaining", 0))
    need = len(soon) * len(MARKETS)
    print(f"{len(soon)} games in the next {days} days, ~{need} credits needed, {remaining} remaining")
    if remaining - need < min_credits:
        sys.exit(f"stopping: would leave fewer than {min_credits} credits this month")
    rows = []
    for e in soon:
        d, h = get(f"{API}/events/{e['id']}/odds?apiKey={key}&regions=us&oddsFormat=american"
                   f"&markets={','.join(MARKETS)}")
        for b in d.get("bookmakers", []):
            for m in b["markets"]:
                for o in m["outcomes"]:
                    rows.append(dict(event_id=e["id"], commence=e["commence_time"], home=e["home_team"],
                                     away=e["away_team"], book=b["key"], market=m["key"],
                                     player=o.get("description"), side=o["name"], point=o.get("point"),
                                     price=o["price"]))
    print(f"credits remaining after run: {h.get('x-requests-remaining')}")
    return pd.DataFrame(rows)


def consensus(lines: pd.DataFrame) -> pd.DataFrame:
    """Per player-market: the most-quoted line, the market's de-vigged
    probability at it (averaged over books quoting both sides), and the best
    available price for each side across books at that line."""
    out = []
    for (ev, mkt, pl), g in lines.groupby(["event_id", "market", "player"]):
        base = g.iloc[0]
        if mkt == "player_anytime_td":
            yes = g[g.side == "Yes"]
            if yes.empty:
                continue
            # no "No" side is quoted: strip a typical ~7% anytime-TD margin
            implied = (1 / dec(yes.price)).mean() / 1.07
            best = yes.price.max()
            out.append(dict(event_id=ev, home=base.home, away=base.away, commence=base.commence, market=mkt,
                            player=pl, line=0.5, p_market_over=float(implied), best_over=float(best),
                            best_under=np.nan, books=int(yes.book.nunique())))
            continue
        line = g.point.mode().iloc[0]
        at = g[g.point == line]
        both = at.pivot_table(index="book", columns="side", values="price", aggfunc="first").dropna()
        if both.empty or not {"Over", "Under"} <= set(both.columns):
            continue
        io, iu = 1 / dec(both.Over), 1 / dec(both.Under)
        out.append(dict(event_id=ev, home=base.home, away=base.away, commence=base.commence, market=mkt,
                        player=pl, line=float(line), p_market_over=float((io / (io + iu)).mean()),
                        best_over=float(at[at.side == "Over"].price.max()),
                        best_under=float(at[at.side == "Under"].price.max()), books=int(len(both))))
    return pd.DataFrame(out)


def _norm(name: str) -> str:
    import re
    n = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).split())


def add_model(c: pd.DataFrame) -> pd.DataFrame:
    """Model P(over) for each prop. A book's player is matched only on an exact
    (normalised) name for a player whose current team is in that game —
    fuzzy matching attached other players' projections to unknown names."""
    from nfl_engine.query.serving import ENGINE
    pc = ENGINE.player_current
    pc = pc[pc.on_roster == 1].assign(_n=lambda d: d.player_name.map(_norm))
    probs, ids, teams = [], [], []
    for r in c.itertuples():
        home, away = resolve_team(r.home), resolve_team(r.away)
        hit = pc[(pc._n == _norm(r.player)) & pc.team.isin([home, away])]
        if len(hit) != 1:
            probs.append(np.nan); ids.append(None); teams.append(None); continue
        p = hit.iloc[0]
        opp = away if p.team == home else home
        stat = ALL_MARKETS[r.market]
        name = p.player_name
        if stat == "anytime_td":
            miss = 1.0
            for s in ("rushing_tds", "receiving_tds"):
                q = ENGINE.prop_probability(name, s, 0.5, opp)
                if q and "prob_over" in q:
                    miss *= 1 - q["prob_over"]
            pr = 1 - miss if miss < 1 else np.nan
        else:
            q = ENGINE.prop_probability(name, stat, r.line, opp, starter=True if stat in STARTER_STATS else None)
            pr = q["prob_over"] if q and "prob_over" in q else np.nan
        probs.append(pr); ids.append(p.player_id); teams.append(p.team)
    c = c.assign(player_id=ids, team=teams, p_model_over=probs)
    c["edge_over"] = c.p_model_over - c.p_market_over
    c["ev_over"] = c.p_model_over * (dec(c.best_over) - 1) - (1 - c.p_model_over)
    c["ev_under"] = (1 - c.p_model_over) * (dec(c.best_under.fillna(-1e9)) - 1) - c.p_model_over
    c.loc[c.best_under.isna(), "ev_under"] = np.nan
    return c


def snapshot(argv):
    ap = argparse.ArgumentParser(prog="props.py snapshot")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--min-credits", type=int, default=60)
    ap.add_argument("--reprice", action="store_true",
                    help="re-run the model on today's saved lines without spending credits")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    if a.reprice:
        lines = pd.read_csv(OUT / f"lines_{today}.csv")
    else:
        lines = fetch(a.days, a.min_credits)
        lines.assign(captured_at=dt.datetime.now().isoformat(timespec="minutes")).to_csv(OUT / f"lines_{today}.csv", index=False)
    c = add_model(consensus(lines))
    c["captured_at"] = dt.datetime.now().isoformat(timespec="minutes")
    c.to_csv(OUT / f"model_{today}.csv", index=False)
    ok = c.dropna(subset=["p_model_over"])
    print(f"{len(lines)} book quotes, {len(c)} player props, model priced {len(ok)}")
    top = pd.concat([ok.assign(side="over", ev=ok.ev_over, p=ok.p_model_over, mk=ok.p_market_over),
                     ok.assign(side="under", ev=ok.ev_under, p=1 - ok.p_model_over, mk=1 - ok.p_market_over)])
    cands = candidates(c)
    cands.to_csv(OUT / f"candidates_{today}.csv", index=False)
    print(f"\n{len(cands)} candidates under the primary strategy (QB volume props, +EV at best price):")
    print(cands[["player", "market", "line", "side", "best_price", "p_model", "p_market", "ev", "books"]]
          .round(3).to_string(index=False) if len(cands) else "  none this week")


def candidates(c: pd.DataFrame) -> pd.DataFrame:
    """The primary strategy's bets: one side per prop, the side with positive
    EV at the best available price."""
    q = c[c.market.isin(PRIMARY_MARKETS)].dropna(subset=["p_model_over"])
    rows = []
    for r in q.itertuples():
        for side, ev, price, p, mk in (("over", r.ev_over, r.best_over, r.p_model_over, r.p_market_over),
                                       ("under", r.ev_under, r.best_under, 1 - r.p_model_over, 1 - r.p_market_over)):
            if pd.notna(ev) and ev > 0:
                rows.append(dict(event_id=r.event_id, commence=r.commence, home=r.home, away=r.away,
                                 market=r.market, player=r.player, player_id=r.player_id, team=r.team,
                                 line=r.line, side=side, best_price=price, p_model=p, p_market=mk,
                                 ev=ev, books=r.books))
    out = pd.DataFrame(rows)
    return out.sort_values("ev", ascending=False) if len(out) else out


# ---------------------------------------------------------------- grading

def actuals() -> pd.DataFrame:
    ps = read_parquet(RAW / "player_stats.parquet")
    ps = ps[ps.season_type == "REG"].copy()
    ps["anytime_td"] = ps.rushing_tds.fillna(0) + ps.receiving_tds.fillna(0)
    sched = read_parquet(RAW / "schedules.parquet")[["season", "week", "gameday", "home_team", "away_team"]]
    sched = sched.assign(home_team=sched.home_team.map(norm_team), away_team=sched.away_team.map(norm_team))
    return ps, sched


def week_of(r, sched) -> tuple:
    """Season/week of the event, from the kickoff date in US Eastern time."""
    d = pd.Timestamp(r.commence).tz_convert("America/New_York").strftime("%Y-%m-%d")
    h, a = resolve_team(r.home), resolve_team(r.away)
    m = sched[(sched.home_team == h) & (sched.away_team == a) & (sched.gameday == d)]
    return (int(m.season.iloc[0]), int(m.week.iloc[0])) if len(m) else (None, None)


def strength(hit: float, n: int) -> str:
    """How sure we can be that the hit rate beats the 52.4% break-even."""
    if n == 0:
        return "no bets graded yet"
    from scipy.stats import binomtest
    p = binomtest(int(round(hit * n)), n, BREAKEVEN, alternative="greater").pvalue
    if n < 100:
        return f"too few bets to tell ({n} graded; a real edge needs a few hundred)"
    if p < 0.001:
        return "very strong evidence of a real edge"
    if p < 0.05:
        return "fairly strong evidence — unlikely to be luck"
    if p < 0.2:
        return "promising, but could still be luck"
    return "no evidence of an edge yet"


def settle(bets: pd.DataFrame, ps: pd.DataFrame, sched) -> pd.DataFrame:
    rows = []
    for r in bets.itertuples():
        season, week = week_of(r, sched)
        if season is None or pd.isna(r.player_id):
            continue
        row = ps[(ps.player_id == r.player_id) & (ps.season == season) & (ps.week == week)]
        if row.empty:
            continue  # not played yet, data not refreshed, or player inactive (voided)
        actual = float(row[ALL_MARKETS[r.market]].iloc[0])
        diff = actual - r.line
        push = diff == 0
        won = (diff > 0) if r.side == "over" else (diff < 0)
        profit = 0.0 if push else (float(dec(r.best_price)) - 1 if won else -1.0)
        rows.append(dict(r._asdict(), season=season, week=week, actual=actual, won=bool(won),
                         push=bool(push), profit=profit))
    return pd.DataFrame(rows)


def exploratory_bets(c: pd.DataFrame) -> pd.DataFrame:
    """Every other prop's model side, at the best price (for ideas only)."""
    q = c[~c.market.isin(PRIMARY_MARKETS)].dropna(subset=["p_model_over"])
    rows = []
    for r in q.itertuples():
        over = r.p_model_over >= r.p_market_over
        price = r.best_over if over else r.best_under
        if pd.isna(price):
            continue
        rows.append(dict(event_id=r.event_id, commence=r.commence, home=r.home, away=r.away, market=r.market,
                         player=r.player, player_id=r.player_id, line=r.line, side="over" if over else "under",
                         best_price=price, p_model=r.p_model_over if over else 1 - r.p_model_over,
                         p_market=r.p_market_over if over else 1 - r.p_market_over))
    return pd.DataFrame(rows)


def report(g: pd.DataFrame, title: str):
    if g.empty:
        print(f"\n{title}: nothing graded yet (games not played, or data not refreshed since)")
        return
    d = g[~g.push]
    n, hit = len(d), (d.won.mean() if len(d) else 0.0)
    units = g.profit.sum()
    print(f"\n{title}: {int(d.won.sum())}-{int((~d.won).sum())}" + (f"-{int(g.push.sum())}" if g.push.any() else "") +
          f"  hit {hit:.1%}  ·  {units:+.2f} units on {len(g)} bets ({units/len(g):+.1%} per bet)" if len(g) else f"\n{title}: none graded")
    if len(g):
        print(f"  {strength(hit, n)}")
        for m, x in g.groupby("market"):
            x2 = x[~x.push]
            print(f"    {m:24s} {int(x2.won.sum())}-{int((~x2.won).sum())}  {x.profit.sum():+.2f} units")


def grade():
    ps, sched = actuals()
    prim, expl = [], []
    for f in sorted(OUT.glob("model_*.csv")):
        c = pd.read_csv(f)
        prim.append(settle(candidates(c), ps, sched).assign(snapshot=f.stem))
        expl.append(settle(exploratory_bets(c), ps, sched).assign(snapshot=f.stem))
    # a game captured in more than one snapshot counts once: the latest one
    key = ["event_id", "market", "player"]
    def combine(frames):
        frames = [x for x in frames if len(x)]      # snapshots with no games played yet
        if not frames:
            return pd.DataFrame()
        g = pd.concat(frames, ignore_index=True).sort_values("snapshot").drop_duplicates(key, keep="last")
        return g.astype({"won": bool, "push": bool})
    prim, expl = combine(prim), combine(expl)
    pd.concat([prim.assign(record="primary"), expl.assign(record="exploratory")]).to_csv(OUT / "graded.csv", index=False)
    report(prim, "PRIMARY (QB volume props, +EV at best price) — the record that counts")
    report(expl, "Exploratory (all other props, model side) — ideas only")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "snapshot":
        snapshot(sys.argv[2:])
    elif cmd == "grade":
        grade()
    else:
        sys.exit("usage: python scripts/props.py snapshot [--days 8] [--min-credits 60] [--reprice] | grade")
