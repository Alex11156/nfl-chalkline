"""Local natural-language parser: pattern-matches a question to a tool call.

Free/offline fallback for the Claude tool-use mode. Recognizes:
game predictions, fantasy projections, player comparisons, prop lines,
situational trends, power ratings, head-to-head, and stat lookups.
"""
import re

from nfl_engine.config import TEAM_NAMES
from nfl_engine.query.tools import _match_stat, call_tool


def find_teams(text: str) -> list[str]:
    t = " " + text.lower() + " "
    hits = []
    for code, aliases in TEAM_NAMES.items():
        for a in aliases + [code.lower()]:
            pos = t.find(f" {a} ") if len(a) <= 3 else t.find(a)
            if pos >= 0:
                hits.append((pos, code))
                break
    hits.sort()
    seen, ordered = set(), []
    for _, c in hits:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def find_players(text: str, engine) -> list[str]:
    """Match capitalized name-like spans against the player dictionary."""
    pc = engine.player_current
    names = pc.player_name.tolist()
    lower_map = {n.lower(): n for n in names}
    t = text.lower()
    found = []
    for ln, orig in lower_map.items():
        if ln in t and orig not in found:
            found.append((t.find(ln), orig, len(ln)))
    # prefer longer matches at distinct positions (avoid substring dupes)
    found.sort(key=lambda x: (x[0], -x[2]))
    out, used = [], []
    for pos, name, ln in found:
        if any(abs(pos - p) < 5 for p in used):
            continue
        out.append(name)
        used.append(pos)
    if out:
        return out
    # last-name fallback
    last_map = {}
    for n in names:
        last = n.split()[-1].lower()
        last_map.setdefault(last, []).append(n)
    for w in re.findall(r"[a-z']+", t):
        if w in last_map and len(w) > 3:
            cands = last_map[w]
            pc_sub = pc[pc.player_name.isin(cands)].sort_values("gameday")
            out.append(pc_sub.iloc[-1].player_name)
    return list(dict.fromkeys(out))


NUM = r"(\d+(?:\.\d+)?)"


def parse(question: str, engine) -> tuple[str, dict]:
    """Returns (tool_name, args)."""
    q = question.strip()
    ql = q.lower()
    teams = find_teams(q)

    # situational trends
    for topic, pats in {
        "new_coach": [r"new (head )?coach", r"first.year coach", r"coaching change"],
        "rest_advantage": [r"rest advantage", r"extra rest", r"off a bye", r"short week"],
        "division_underdogs": [r"division(al)? (under)?dog", r"divisional.*spread"],
        "wind_totals": [r"wind", r"weather.*(total|under|over)"],
        "primetime_favorites": [r"prime.?time", r"night game"],
    }.items():
        if any(re.search(p, ql) for p in pats):
            return "situational_query", {"topic": topic}

    # power ratings / rankings
    if re.search(r"(power )?ranking|best team|elo|ratings", ql):
        return "team_ratings", {}

    # head to head history
    if len(teams) >= 2 and re.search(r"head.to.head|history|record against|last \d+ (games|meetings)", ql):
        return "head_to_head", {"team_a": teams[0], "team_b": teams[1]}

    players = find_players(q, engine)

    # prop: player + stat + number
    mnum = re.search(NUM, ql)
    stat = _match_stat(ql)
    if players and stat and mnum and re.search(r"over|under|prop|line|hit|o/u", ql):
        opp = teams[0] if teams else None
        return "prop_probability", {"name": players[0], "stat": stat,
                                    "line": float(mnum.group(1)),
                                    "opponent": opp}

    # compare / start-sit
    if len(players) >= 2 and re.search(r" or |compare|vs\.?|versus|start|sit|better", ql):
        return "compare_players", {"names": players[:4]}

    # single-player projection or stats lookup
    if players:
        myear = re.search(r"\b(19\d\d|20\d\d)\b", ql)
        if re.search(r"stats|numbers|how did|season totals|career", ql):
            args = {"name": players[0]}
            if myear:
                args["season"] = int(myear.group(1))
            return "player_stats_lookup", args
        opp = teams[0] if teams else None
        return "project_player", {"name": players[0], "opponent": opp}

    # game prediction: two teams
    if len(teams) >= 2:
        # "X at Y" => Y home; otherwise first-mentioned = home unless 'at'
        m = re.search(r"\bat\b|@", ql)
        if m:
            home, away = teams[1], teams[0]
        else:
            home, away = teams[0], teams[1]
        return "predict_game", {"home": home, "away": away}

    if len(teams) == 1 and re.search(r"rating|good|strength", ql):
        return "team_ratings", {}

    return "help", {}


HELP = ("I couldn't match that question. Try things like:\n"
        "- 'Chiefs vs Bills — who wins?'\n"
        "- 'Project Bijan Robinson against the Saints'\n"
        "- 'Start Puka Nacua or CeeDee Lamb?'\n"
        "- 'Josh Allen over 250.5 passing yards vs Miami?'\n"
        "- 'How do teams with new head coaches do against the spread?'\n"
        "- 'Power rankings' / 'Cowboys vs Eagles head to head'")


def answer(question: str, engine) -> dict:
    tool, args = parse(question, engine)
    if tool == "help":
        return {"tool": "help", "args": {}, "result": None, "text": HELP}
    result = call_tool(tool, args)
    return {"tool": tool, "args": args, "result": result,
            "text": compose(tool, args, result)}


def _fmt_pct(x) -> str:
    return f"{100*x:.0f}%"


def compose(tool: str, args: dict, r: dict) -> str:
    if r is None:
        return "No result."
    if "error" in r:
        extra = ""
        if "supported" in r:
            extra = f" Supported: {', '.join(r['supported'])}"
        return f"⚠️ {r['error']}.{extra}"
    if tool == "predict_game":
        fav = r["home"] if r["pred_margin"] > 0 else r["away"]
        return (f"**{r['away']} @ {r['home']}** — model favors **{fav}**.\n"
                f"- Win probability: {r['home']} {_fmt_pct(r['home_win_prob'])} / "
                f"{r['away']} {_fmt_pct(r['away_win_prob'])}\n"
                f"- Projected score: {r['home']} {r['home_score']} – {r['away']} {r['away_score']} "
                f"(margin {r['pred_margin']:+.1f}, total {r['pred_total']:.1f})\n"
                f"- Elo check: {r['home']} {r['elo_home']} vs {r['away']} {r['elo_away']} "
                f"(Elo home win prob {_fmt_pct(r['elo_win_prob'])})\n"
                f"- Uncertainty: ±{r['sigma_margin']} pts on the margin — treat close calls as coin flips.")
    if tool == "project_player":
        f = r["fantasy_points_ppr"]
        vs = f" vs {r['opponent']}" if r.get("opponent") else ""
        note = ("" if r["last_seen_season"] >= 2025 else
                f"\n⚠️ Last data from {r['last_seen_season']} — treat with caution.")
        return (f"**{r['player']}** ({r['position']}, {r['team']}){vs}\n"
                f"- PPR: **{f['mean']:.1f}** (floor {f['floor']}, median {f['median']}, ceiling {f['ceiling']})\n"
                f"- Standard: {r['fantasy_points']['mean']:.1f} | Half-PPR: {r['fantasy_points_half']['mean']:.1f}"
                + note)
    if tool == "compare_players":
        lines = []
        for p in r["projections"]:
            if "error" in p:
                lines.append(f"- ⚠️ {p['error']}")
            else:
                f = p["fantasy_points_ppr"]
                lines.append(f"- **{p['player']}** ({p['position']}, {p['team']}): "
                             f"{f['mean']:.1f} PPR (floor {f['floor']}, ceiling {f['ceiling']})")
        rec = f"\n➡️ **Start {r['recommend_ppr']}** (higher PPR mean)." if r.get("recommend_ppr") else ""
        return "\n".join(lines) + rec
    if tool == "prop_probability":
        d = r["distribution"]
        dist = (f"mean {d.get('mean'):.1f}" +
                (f", p10–p90 {d.get('p10')}–{d.get('p90')}" if "p10" in d else ""))
        lean = "OVER" if r["prob_over"] > 0.55 else ("UNDER" if r["prob_over"] < 0.45 else "no strong lean")
        return (f"**{r['player']}** — {r['stat'].replace('_', ' ')} over {r['line']}"
                + (f" vs {r['opponent']}" if r.get("opponent") else "") + "\n"
                f"- P(over) **{_fmt_pct(r['prob_over'])}** / P(under) {_fmt_pct(r['prob_under'])} ({dist})\n"
                f"- Lean: **{lean}**. Note: model edge only counts if it beats the book's implied odds.")
    if tool == "team_ratings":
        rows = r["ratings"][:10]
        body = "\n".join(f"{i+1}. **{x['team']}** — Elo {x['elo']:.0f}, "
                         f"off EPA {x['off_epa']:+.3f}, def EPA allowed {x['def_epa_allowed']:+.3f}"
                         for i, x in enumerate(rows))
        return "**Current power ratings (top 10)**\n" + body
    if tool == "player_stats_lookup":
        lines = [f"**{r['player']}** ({r['position']}, {r['team']}) — recent seasons:"]
        for s in r["seasons"][:6]:
            bits = [f"{s['games']} gms"]
            if s.get("pass_yds"):
                bits.append(f"{s['pass_yds']:.0f} pass yds, {s['pass_tds']:.0f} TD")
            if s.get("rush_yds"):
                bits.append(f"{s['rush_yds']:.0f} rush yds")
            if s.get("rec"):
                bits.append(f"{s['rec']:.0f} rec, {s['rec_yds']:.0f} yds")
            bits.append(f"{s['fp_ppr_pg']} PPR/gm")
            lines.append(f"- {s['season']}: " + ", ".join(bits))
        return "\n".join(lines)
    if tool == "head_to_head":
        a, b = r["teams"]
        lines = [f"**{a} vs {b}** — last {r['last_n']} meetings: "
                 f"{a} {r[f'{a}_wins']}, {b} {r[f'{b}_wins']}" +
                 (f", ties {r['ties']}" if r.get("ties") else "")]
        for g in r["games"][-5:]:
            lines.append(f"- {g['season']} wk{g['week']}: {g['away_team']} "
                         f"{g['away_score']:.0f} @ {g['home_team']} {g['home_score']:.0f}")
        return "\n".join(lines)
    if tool == "situational_query":
        lines = [f"**Situational trend: {r['topic'].replace('_',' ')}** (since {r['since']})"]
        for row in r["result"]:
            lines.append("- " + ", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(lines)
    return str(r)
