"""NFL Prediction Engine — Streamlit UI.

Run:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nfl_engine.config import PROP_STATS, REPORTS, TEAMS

# Validated categorical palette (fixed assignment order)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK_MUTED = "#52514e"

st.set_page_config(page_title="NFL Prediction Engine", page_icon="🏈",
                   layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading models and data …")
def get_engine():
    from nfl_engine.query.serving import ENGINE
    # assignment (not a bare expression) so Streamlit magic doesn't render it
    _warm = (ENGINE.game_models, ENGINE.player_models)
    return ENGINE


def chart_layout(fig, title=""):
    fig.update_layout(
        title=title, template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=40), showlegend=True,
        font=dict(color="#0b0b0b"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#eeeeec", zeroline=False)
    fig.update_yaxes(gridcolor="#eeeeec", zeroline=False)
    return fig


engine = get_engine()

tab_ask, tab_games, tab_props, tab_fantasy, tab_report = st.tabs(
    ["💬 Ask Anything", "🎯 Game Predictions", "📊 Player Props",
     "🏆 Fantasy", "📋 Model Report Card"])

# ---------------- Ask Anything ----------------
with tab_ask:
    from nfl_engine.query import claude_agent
    from nfl_engine.query.parser import answer as local_answer

    claude_ok = claude_agent.available()
    with st.sidebar:
        st.markdown("### 🏈 NFL Prediction Engine")
        mode = st.radio("Answer mode", ["Local parser (free)", "Claude API"],
                        index=1 if claude_ok else 0,
                        help="Claude mode needs an Anthropic API key "
                             "(export ANTHROPIC_API_KEY before launching).")
        if mode == "Claude API" and not claude_ok:
            st.warning("No Anthropic credentials — using the local parser.")
            mode = "Local parser (free)"
        st.caption("1999–2025 play-by-play · walk-forward validated · "
                   "see Model Report Card for honest accuracy")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if not st.session_state.chat:
        st.markdown("#### Ask about any game, player, prop, or matchup")
        st.caption("Try: *“Chiefs vs Bills, who wins?”* · *“Josh Allen over 250.5 "
                   "passing yards vs Miami?”* · *“Start Puka Nacua or CeeDee Lamb?”* · "
                   "*“How do new head coaches do against the spread?”*")
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    q = st.chat_input("Ask about a game, player, prop, or trend …")
    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    if mode == "Claude API":
                        r = claude_agent.answer(q)
                        text = r["text"]
                        if r["tool_calls"]:
                            tools_used = ", ".join(sorted({t["tool"] for t in r["tool_calls"]}))
                            text += f"\n\n*<sub>tools used: {tools_used}</sub>*"
                    else:
                        text = local_answer(q, engine)["text"]
                except Exception as e:
                    text = f"⚠️ Error: {e}"
            st.markdown(text)
        st.session_state.chat.append(("assistant", text))

# ---------------- Game Predictions ----------------
with tab_games:
    c1, c2, c3 = st.columns([2, 2, 1])
    away = c1.selectbox("Away team", TEAMS, index=TEAMS.index("KC"))
    home = c2.selectbox("Home team", TEAMS, index=TEAMS.index("BUF"))
    neutral = c3.checkbox("Neutral site")
    if st.button("Predict game", type="primary"):
        if home == away:
            st.error("Pick two different teams.")
        else:
            r = engine.predict_game(home, away, neutral)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{r['home']} win probability", f"{r['home_win_prob']:.0%}")
            m2.metric(f"{r['away']} win probability", f"{r['away_win_prob']:.0%}")
            m3.metric("Projected margin", f"{r['home']} {r['pred_margin']:+.1f}")
            m4.metric("Projected total", f"{r['pred_total']:.1f}")
            st.markdown(
                f"**Projected score: {r['home']} {r['home_score']:.0f} – "
                f"{r['away']} {r['away_score']:.0f}**  ·  Elo: {r['home']} "
                f"{r['elo_home']} vs {r['away']} {r['elo_away']} "
                f"(Elo agrees at {r['elo_win_prob']:.0%} home)")
            st.info(f"Margin uncertainty is ±{r['sigma_margin']} points (1σ). "
                    "A 60/40 edge is real but far from certain.")

            fig = go.Figure()
            fig.add_bar(x=[r["home_win_prob"]], y=["win prob"], orientation="h",
                        marker_color=C1, name=r["home"], text=f"{r['home']} {r['home_win_prob']:.0%}",
                        textposition="inside")
            fig.add_bar(x=[r["away_win_prob"]], y=["win prob"], orientation="h",
                        marker_color=C2, name=r["away"], text=f"{r['away']} {r['away_win_prob']:.0%}",
                        textposition="inside")
            fig.update_layout(barmode="stack", height=140, showlegend=False,
                              template="plotly_white", margin=dict(l=20, r=20, t=10, b=10),
                              xaxis=dict(visible=False), yaxis=dict(visible=False),
                              plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

# ---------------- Player Props ----------------
with tab_props:
    c1, c2, c3, c4 = st.columns([3, 2, 1.5, 2])
    pname = c1.text_input("Player", "Josh Allen")
    stat_label = c2.selectbox("Stat", list(PROP_STATS.values()))
    stat = [k for k, v in PROP_STATS.items() if v == stat_label][0]
    line = c3.number_input("Line", value=250.5, step=0.5)
    opp = c4.selectbox("Opponent (optional)", ["—"] + TEAMS)
    if st.button("Evaluate prop", type="primary"):
        r = engine.prop_probability(pname, stat, line,
                                    None if opp == "—" else opp)
        if r is None or "error" in (r or {}):
            st.error((r or {}).get("error", f"Player not found: {pname}"))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("P(over)", f"{r['prob_over']:.0%}")
            m2.metric("P(under)", f"{r['prob_under']:.0%}")
            m3.metric("Model mean", f"{r['distribution'].get('mean', float('nan')):.1f}")
            d = r["distribution"]
            if "p10" in d:
                qs = [d["p10"], d["p25"], d["p50"], d["p75"], d["p90"]]
                fig = go.Figure()
                fig.add_scatter(x=qs, y=[10, 25, 50, 75, 90], mode="lines+markers",
                                line=dict(color=C1, width=2),
                                marker=dict(size=8, color=C1),
                                name="model distribution",
                                hovertemplate="%{x:.1f} → p%{y}<extra></extra>")
                fig.add_vline(x=line, line_width=2, line_color=C2)
                fig.add_annotation(x=line, y=95, text=f"line {line}",
                                   showarrow=False, font=dict(color=C2))
                chart_layout(fig, f"{r['player']} — {stat_label} distribution")
                fig.update_layout(showlegend=False)
                fig.update_yaxes(title="percentile")
                fig.update_xaxes(title=stat_label)
                st.plotly_chart(fig, width="stretch")
            st.caption("An edge only pays if it beats the sportsbook's implied "
                       "probability (≈52.4% at −110). Small edges are noise.")

# ---------------- Fantasy ----------------
with tab_fantasy:
    st.subheader("Head-to-head start/sit")
    c1, c2 = st.columns(2)
    p1 = c1.text_input("Player A", "Puka Nacua")
    p2 = c2.text_input("Player B", "CeeDee Lamb")
    if st.button("Compare", type="primary"):
        projections = [engine.project_player(p) for p in (p1, p2)]
        ok = [p for p in projections if p]
        missing = [n for n, p in zip((p1, p2), projections) if not p]
        for n in missing:
            st.error(f"Player not found: {n}")
        if len(ok) == 2:
            fig = go.Figure()
            colors = [C1, C2]
            for i, pr in enumerate(ok):
                f = pr["fantasy_points_ppr"]
                fig.add_scatter(
                    x=[f["floor"], f["ceiling"]], y=[pr["player"]] * 2,
                    mode="lines", line=dict(color=colors[i], width=8),
                    name=pr["player"], showlegend=False,
                    hovertemplate=f"floor {f['floor']} – ceiling {f['ceiling']}<extra></extra>")
                fig.add_scatter(
                    x=[f["mean"]], y=[pr["player"]], mode="markers",
                    marker=dict(color="#0b0b0b", size=12, symbol="line-ns-open"),
                    name="mean", showlegend=False,
                    hovertemplate=f"mean {f['mean']}<extra></extra>")
            chart_layout(fig, "Projected PPR range (floor → ceiling, tick = mean)")
            fig.update_layout(height=220)
            st.plotly_chart(fig, width="stretch")
            best = max(ok, key=lambda p: p["fantasy_points_ppr"]["mean"])
            st.success(f"**Start {best['player']}** — higher projected PPR mean "
                       f"({best['fantasy_points_ppr']['mean']:.1f}).")
            rows = [{"player": p["player"], "pos": p["position"], "team": p["team"],
                     "PPR mean": p["fantasy_points_ppr"]["mean"],
                     "floor (p10)": p["fantasy_points_ppr"]["floor"],
                     "median": p["fantasy_points_ppr"]["median"],
                     "ceiling (p90)": p["fantasy_points_ppr"]["ceiling"],
                     "standard": p["fantasy_points"]["mean"],
                     "half-PPR": p["fantasy_points_half"]["mean"]} for p in ok]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.divider()
    st.subheader("Top projections by position")
    pos = st.selectbox("Position", ["QB", "RB", "WR", "TE"])
    if st.button("Project top 20"):
        pc = engine.player_current
        cand = (pc[(pc.position == pos) & (pc.season == pc.season.max())]
                .nlargest(20, "r_fantasy_points_ppr"))
        with st.spinner("Projecting …"):
            rows = []
            for name in cand.player_name:
                pr = engine.project_player(name)
                if pr:
                    f = pr["fantasy_points_ppr"]
                    rows.append({"player": pr["player"], "team": pr["team"],
                                 "PPR mean": f["mean"], "floor": f["floor"],
                                 "ceiling": f["ceiling"],
                                 "standard": pr["fantasy_points"]["mean"]})
        st.dataframe(pd.DataFrame(rows).sort_values("PPR mean", ascending=False),
                     hide_index=True, height=500)

# ---------------- Report Card ----------------
with tab_report:
    st.subheader("How good is this model, honestly?")
    st.markdown(
        "- Every number below is **walk-forward out-of-sample**: the model "
        "predicting seasons it never trained on (2010–2025, ~4,300 games).\n"
        "- **Vegas closing lines are the benchmark to beat, and they are hard "
        "to beat.** Breakeven against the spread at standard −110 odds is 52.4%.\n"
        "- Use the model for *magnitude and calibration* (probabilities, "
        "floors/ceilings), not as a guaranteed edge.")
    try:
        ev_pure = json.loads((REPORTS / "game_eval_pure.json").read_text())
        ev_mkt = json.loads((REPORTS / "game_eval_mkt.json").read_text())
        m, mm = ev_pure["margin"], ev_mkt["margin"]
        df = pd.DataFrame([
            {"model": "This engine (no Vegas info)", "straight-up": f"{m['su_acc']:.1%}",
             "margin MAE": f"{m['mae']:.2f}", "ATS": f"{m['ats_pct']:.1%}",
             "Brier": f"{m['brier']:.4f}"},
            {"model": "This engine (with market)", "straight-up": f"{mm['su_acc']:.1%}",
             "margin MAE": f"{mm['mae']:.2f}", "ATS": f"{mm['ats_pct']:.1%}",
             "Brier": f"{mm['brier']:.4f}"},
            {"model": "Elo baseline", "straight-up": f"{m['elo_su']:.1%}",
             "margin MAE": "—", "ATS": "—", "Brier": f"{m['brier_elo']:.4f}"},
            {"model": "Vegas closing line", "straight-up": f"{m['vegas_su']:.1%}",
             "margin MAE": f"{m['vegas_mae']:.2f}", "ATS": "50% by definition",
             "Brier": "—"},
        ])
        st.table(df)
        seasons = {int(k): v for k, v in ev_pure["su_by_season"].items()}
        fig = go.Figure()
        fig.add_scatter(x=list(seasons), y=list(seasons.values()),
                        mode="lines+markers", line=dict(color=C1, width=2),
                        marker=dict(size=8), name="engine straight-up",
                        hovertemplate="%{x}: %{y:.1%}<extra></extra>")
        fig.add_hline(y=m["vegas_su"], line_dash="dot", line_color=INK_MUTED)
        fig.add_annotation(x=list(seasons)[1], y=m["vegas_su"] + 0.008,
                           text="Vegas favorites", showarrow=False,
                           font=dict(color=INK_MUTED, size=12))
        chart_layout(fig, "Straight-up accuracy by season (out-of-sample)")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    except FileNotFoundError:
        st.warning("Game evaluation reports not found — run "
                   "`python -m nfl_engine.models.game_models` first.")
    try:
        pl = json.loads((REPORTS / "player_eval.json").read_text())
        st.subheader("Player models vs naive rolling average (out-of-sample 2016–2025)")
        rows = [{"target": k, "model MAE": round(v["mae"], 3),
                 "naive MAE": round(v["naive_mae"], 3),
                 "improvement": f"{v['improve_pct']:+.1f}%",
                 "correlation": round(v["corr"], 3)} for k, v in pl.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=530)
    except FileNotFoundError:
        st.warning("Player evaluation report not found.")
