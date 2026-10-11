"""LLM benchmark dashboard. Run locally with:  streamlit run streamlit_app.py

Ranks models by expected cost per task (Actuarial Cost Model, Sprint 3 revised):
    E[C] = cost to run + chance of a wrong answer x cost of that wrong answer + response time x value of time
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from costmodel import (ATTENTION, CATEGORIES, RATE_PER_HOUR, REQUIRED_COLUMNS, c_fail, explain,
                       rank_category, summarize, v_time)
from scoring import grade_of, score as weighted_score
from tasks import CATALOG, CATALOG_LABELS, TASK_CATALOG, match_task

st.set_page_config(page_title="Right-Size Model Bench", page_icon="⚖️", layout="wide")

HERE = Path(__file__).parent
CAT_IDS = list(CATEGORIES)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa6ad"

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.eyebrow {letter-spacing:.12em; text-transform:uppercase; font-size:12.5px; opacity:.6; margin:0 0 .2rem;}
.hero h1 {font-size: 2.6rem; line-height:1.1; margin:.1rem 0 .6rem;}
.hero p {font-size: 1.08rem; opacity:.8; max-width: 62ch;}
.mcard-name {font-weight:700; font-size:1.05rem; margin:0;}
.mcard-role {display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px;
             background:rgba(42,120,214,.12); color:#2a78d6; margin:.25rem 0 .35rem;}
.mcard-meta {font-size:13px; opacity:.7; margin:0;}
.bigcost {font-size:2.6rem; font-weight:700; line-height:1.1;}
</style>
""", unsafe_allow_html=True)


# ---------------- Data ----------------
@st.cache_resource
def load_data():
    cfg = json.loads((HERE / "models.json").read_text())
    models = {m["name"]: m for m in cfg["models"]}
    df = pd.read_csv(HERE / "data" / "results.csv")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"results.csv is missing columns: {', '.join(missing)}")
        st.stop()
    info = {
        "runs": int(df["run"].nunique()),
        "questions": df.groupby("test_set")["question_id"].nunique().to_dict(),
        "not_in_settings": sorted(set(df["model"]) - set(models)),
        "no_results": sorted(set(models) - set(df["model"])),
        "snapshot": cfg.get("price_snapshot", "unknown"),
    }
    return summarize(df, models), models, info


SUMMARY, MODELS, info = load_data()


def label(name):
    m = MODELS.get(name)
    return f"{name} · {m['role']}" if m else name


def price_text(m):
    if m.get("local"):
        return f"{m['provider']} · runs locally · energy cost ({m.get('gpu_watts', 150)} W GPU)"
    return f"{m['provider']} · ${m['price_in_per_1m']:g} in / ${m['price_out_per_1m']:g} out per 1M tokens"


def money(x):
    if x >= 0.095:
        return f"${x:,.2f}"
    return f"${x:.3f}" if x >= 0.0095 else f"${x:.4f}"


def start_grading(text, category=None, wait=None, reason=""):
    st.session_state.update(query=text.strip(), forced_cat=category, forced_wait=wait, reason=reason)
    for k in [k for k in st.session_state if str(k).startswith(("cat-", "pick-", "R-", "m-", "w-"))]:
        del st.session_state[k]


# ===================================================================
# HOMESCREEN
# ===================================================================
if not st.session_state.get("query"):
    st.markdown("<div class='hero'><p class='eyebrow'>Cheap Thrills · CS 124H · LLM Benchmarking</p>"
                "<h1>Which AI model should you use?</h1>"
                "<p>A cheap model isn't cheap if its answers keep needing fixes. Describe your task and we'll "
                "find the model with the lowest real cost per task: the price to run it, plus the mistakes "
                "you'd have to fix, plus the time you spend waiting.</p></div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.form("own-task", border=True):
            st.markdown("**Type your own task**")
            text = st.text_area("Describe your task", label_visibility="collapsed", height=120,
                                placeholder="e.g. A bot that answers students' questions about our course "
                                            "syllabus, or: check my linear algebra homework answers")
            go_own = st.form_submit_button("Find the cheapest model →", type="primary", use_container_width=True)
        if go_own:
            if text.strip():
                start_grading(text)
                st.rerun()
            else:
                st.warning("Type a task first, or pick one from the list.")

    with right:
        with st.container(border=True):
            st.markdown(f"**Or search {len(TASK_CATALOG)} common tasks**")
            pick = st.selectbox("Search tasks", CATALOG_LABELS, index=None, label_visibility="collapsed",
                                placeholder="Start typing: email, SQL, chatbot, homework…")
            if st.button("Find the cheapest model for this task", use_container_width=True, disabled=pick is None):
                task, cat, wait = CATALOG[pick]
                start_grading(task, cat, wait, f"you picked “{task}”")
                st.rerun()
            with st.expander("Browse all tasks by category"):
                groups = {}
                for g, t, _, _ in TASK_CATALOG:
                    groups.setdefault(g, []).append(t)
                for g, ts in groups.items():
                    st.markdown(f"**{g}**")
                    st.caption(" · ".join(ts))

    st.markdown("#### Models being compared")
    cols = st.columns(len(MODELS) or 1)
    for col, m in zip(cols, MODELS.values()):
        with col.container(border=True):
            st.markdown(f"<p class='mcard-name'>{m['name']}</p><span class='mcard-role'>{m['role']}</span>"
                        f"<p class='mcard-meta'>{price_text(m)}</p>"
                        f"<p class='mcard-meta'><code>{m['model_id']}</code></p>", unsafe_allow_html=True)
    st.caption("Benchmarks: " + ", ".join(f"{c['label'].lower()} ({c['benchmark']}, "
                                          f"{info['questions'].get(c['benchmark'], 0):,} questions)"
                                          for c in CATEGORIES.values())
               + f". Each question run {info['runs']} times. Sample data until the benchmark runner is connected.")
    st.stop()


# ===================================================================
# RESULTS PAGE
# ===================================================================
query = st.session_state["query"]
if st.session_state.get("forced_cat"):
    matched, wait_guess, reason = (st.session_state["forced_cat"], st.session_state["forced_wait"],
                                   st.session_state.get("reason", ""))
else:
    matched, wait_guess, reason = match_task(query)

top_l, top_r = st.columns([5, 1])
top_l.markdown("<p class='eyebrow'>Right-Size Model Bench</p>", unsafe_allow_html=True)
if top_r.button("← New task", use_container_width=True):
    st.session_state["query"] = ""
    st.rerun()
st.markdown(f"### “{query}”")

if matched is None:
    st.info("We couldn't tell which kind of task this is. Which is closest?")
    matched = st.radio("Category", CAT_IDS, key=f"pick-{query}", label_visibility="collapsed",
                       format_func=lambda c: f"{CATEGORIES[c]['label']} ({CATEGORIES[c]['benchmark']})")
    reason = "you chose it"

# ---- Sidebar: overrides and settings
with st.sidebar:
    st.header("Settings")
    category = st.selectbox("Task category", CAT_IDS, index=CAT_IDS.index(matched), key=f"cat-{query}-{matched}",
                            format_func=lambda c: f"{CATEGORIES[c]['label']} ({CATEGORIES[c]['benchmark']})")
    cat = CATEGORIES[category]
    min_acc = st.number_input(f"Minimum accuracy on {cat['benchmark']}", 0.0, 1.0, cat["min_acc"], 0.05,
                              key=f"min-{category}",
                              help="Models below this are left out before ranking (LLM Measurements doc).")
    n_boot = st.select_slider("Bootstrap resamples for ties", [1000, 5000, 10000], value=10000)
    st.subheader("Models (models.json)")
    for m in MODELS.values():
        st.caption(f"**{m['name']}** · {m['role']}  \n{price_text(m)}")
    st.caption(f"Price snapshot: {info['snapshot']}")

if category == matched:
    st.caption(f"Ranked as **{cat['label']}** ({cat['benchmark']}) because {reason}. "
               "Change the category in the sidebar if that's wrong.")
else:
    st.caption(f"Ranked as **{cat['label']}** ({cat['benchmark']}), changed in the sidebar.")
for name in info["not_in_settings"]:
    st.warning(f"'{name}' is in results.csv but not in models.json, so it's hidden.")
for name in info["no_results"]:
    st.info(f"'{name}' is in models.json but has no results yet.")

# ---- The two MVP sliders: cost of a wrong answer, value of time
with st.container(border=True):
    st.markdown("**What does your time cost?** These set the cost of a wrong answer and the cost of waiting.")
    c1, c2, c3 = st.columns(3)
    rate = c1.slider("Your time is worth ($ per hour)", 5, 150, int(RATE_PER_HOUR), 5, key=f"R-{query}")
    lo_m, hi_m = cat["range"]
    minutes = c2.slider("A wrong answer costs me (minutes to notice and fix)", lo_m, hi_m,
                        float(cat["minutes"]), 0.1 if hi_m <= 10 else 0.5, key=f"m-{query}-{category}")
    wait_names = list(ATTENTION)
    wait = c3.selectbox("While I wait for an answer…", wait_names,
                        index=wait_names.index(wait_guess or wait_names[1]), key=f"w-{query}")
    cf, vt = c_fail(minutes, rate), v_time(rate, ATTENTION[wait])
    c1.caption(f"Default $30/h. One rate drives both mistakes and waiting.")
    c2.caption(f"= **{money(cf)} per wrong answer**. Default for {cat['label'].lower()}: "
               f"{cat['minutes']:g} min. {cat['why']}")
    c3.caption(f"Attention factor {ATTENTION[wait]:g} → {money(vt * 4)} for a 4-second wait.")

result = rank_category(SUMMARY, category, rate, minutes, ATTENTION[wait], min_acc, n_boot)
T, top = result["table"], result["top"]

# ---- Recommendation
with st.container(border=True):
    if top is None:
        st.error(explain(result, category, MODELS, info["runs"]))
    else:
        g, txt = st.columns([2, 7])
        tied = "tied" in top["status"]
        g.markdown(f"<div class='bigcost'>{money(top['ec'])}</div>"
                   f"<div style='opacity:.7;font-size:13px'>per task, mistakes included<br>"
                   f"95% range {money(top['ec_best'])} – {money(top['ec_worst'])}</div>", unsafe_allow_html=True)
        txt.subheader(f"Recommended: {top['model']}" + ("  ·  tied, low confidence" if tied else ""))
        txt.caption(f"{MODELS[top['model']]['role']} · {price_text(MODELS[top['model']])}")
        txt.markdown(explain(result, category, MODELS, info["runs"]))

if T.empty:
    st.stop()

# ---- Chart 1: what each model costs per task, split into its three parts
st.subheader("What each model really costs you per task")
st.caption("Running cost + expected cost of mistakes + cost of waiting. Error bars are 95% ranges from the "
           f"accuracy's Wilson interval on {cat['benchmark']}. Lower is better.")
C = T.iloc[::-1].copy()
C["lbl"] = C["model"].map(label) + np.where(C["eligible"], "", "  (below min accuracy)")
fig = go.Figure()
for col, name, color in [("c_run", "Running cost", AQUA), ("wait", "Waiting", ORANGE), ("mistake", "Mistakes", BLUE)]:
    fig.add_bar(y=C["lbl"], x=C[col], name=name, orientation="h",
                marker=dict(color=[color if e else GREY for e in C["eligible"]], line=dict(color="white", width=1)),
                hovertemplate="%{y}<br>" + name + ": $%{x:.4f}<extra></extra>")
fig.add_scatter(y=C["lbl"], x=C["ec"], mode="markers", name="Total with 95% range",
                marker=dict(color="#222", size=12, symbol="line-ns-open", line=dict(width=2)),
                error_x=dict(type="data", array=C["ec_worst"] - C["ec"], arrayminus=C["ec"] - C["ec_best"],
                             color="#444", thickness=1.5, width=6),
                hovertemplate="%{y}<br>Total $%{x:.3f}<extra></extra>")
fig.add_scatter(y=C["lbl"], x=C["ec_worst"], mode="text", text=[f"   <b>{money(v)}</b>" for v in C["ec"]],
                textposition="middle right", showlegend=False, hoverinfo="skip")
fig.update_layout(barmode="stack", height=90 * len(C) + 110, margin=dict(l=10, r=40, t=10, b=40),
                  xaxis=dict(title="Expected cost per task ($)", rangemode="tozero", range=[0, C["ec_worst"].max() * 1.25],
                             tickprefix="$"),
                  legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)

# ---- Chart 2: the cheapest model depends on what a mistake costs
elig = T[T["eligible"]]
if len(elig) >= 1:
    st.subheader("The cheapest model depends on what a mistake costs")
    st.caption("Expected cost per task as the cost of one wrong answer changes. Blue line: the cheapest model "
               "at each point. Dashed line: your current setting.")
    x_max = max(50.0, cf * 2)
    xs = np.logspace(-2, np.log10(x_max), 200)
    fig2 = go.Figure()
    curves = {}
    for _, r in elig.iterrows():
        ys = r["c_run"] + r["t"] * vt + (1 - r["p"]) * xs
        curves[r["model"]] = ys
        fig2.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=GREY, width=2), name=r["model"],
                         hovertemplate=r["model"] + "<br>mistake costs $%{x:.2f} → $%{y:.3f} per task<extra></extra>")
        fig2.add_annotation(x=np.log10(xs[-1]), y=np.log10(ys[-1]), text=f"{r['model']} · {r['p']:.0%}",
                            showarrow=False, xanchor="left", font=dict(size=11))
    env = np.min(np.vstack(list(curves.values())), axis=0)
    fig2.add_scatter(x=xs, y=env, mode="lines", line=dict(color=BLUE, width=4), name="Cheapest",
                     hoverinfo="skip")
    fig2.add_vline(x=cf, line=dict(color="#c23b3a", dash="dash"),
                   annotation_text=f"your setting: {money(cf)}", annotation_position="bottom right")
    dec = [0.01, 0.1, 1, 10, 100]
    fig2.update_layout(xaxis=dict(type="log", title="Cost of one wrong answer ($, log scale)",
                                  range=[-2, np.log10(x_max)], tickvals=dec, ticktext=[money(d) for d in dec]),
                       yaxis=dict(type="log", title="Expected cost per task ($, log)",
                                  tickvals=dec, ticktext=[money(d) for d in dec]),
                       height=420, margin=dict(l=10, r=150, t=20, b=40), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---- Table
st.subheader("Model breakdown")
tbl = pd.DataFrame({
    "Model": T["model"], "Type": T["model"].map(lambda m: MODELS[m]["role"]), "Status": T["status"],
    "Expected cost / task": T["ec"].map(money),
    "95% range": [f"{money(a)} – {money(b)}" for a, b in zip(T["ec_best"], T["ec_worst"])],
    "Accuracy": [f"{p:.0%} ({p_lo:.0%}–{p_hi:.0%})" for p, p_lo, p_hi in zip(T["p"], T["p_lo"], T["p_hi"])],
    "Questions": T["n"], "Median time": T["t"].map(lambda x: f"{x:.1f}s"),
    "Run cost": T["c_run"].map(money), "Mistakes": T["mistake"].map(money), "Waiting": T["wait"].map(money),
    "Runs disagree": T["unstable"].map(lambda x: f"{x:.0%} of questions"),
    "Δ vs cheapest (95%)": [("—" if pd.isna(lo) else f"{lo:+.2f} to {hi:+.2f}")
                             for lo, hi in zip(T["diff_lo"], T["diff_hi"])],
})
st.dataframe(tbl, hide_index=True, use_container_width=True)
st.caption("“Runs disagree” is the stability measure: the share of questions where the 5 runs weren't unanimous. "
           "Δ is this model's expected cost minus the cheapest model's, from a paired bootstrap over questions; "
           "if the range includes 0 the two are tied.")

# ---- All categories at your settings
st.subheader("Every category at your time value")
st.caption("Expected cost per task in each category, using each category's default fix time, your hourly rate "
           "and waiting setting. ★ marks the cheapest model.")
overview = {}
for cid, c in CATEGORIES.items():
    r = rank_category(SUMMARY, cid, rate, None, ATTENTION[wait], None, n_boot=1000)["table"]
    col = {}
    for i, row in r.iterrows():
        col[label(row["model"])] = ("below min" if not row["eligible"] else
                                    ("★ " if i == 0 else "") + money(row["ec"]))
    overview[f"{c['label']} ({c['benchmark']})"] = col
st.dataframe(pd.DataFrame(overview), use_container_width=True)

# ---- Secondary view: weighted 0-100 score and grade (cost model doc, step 4)
LEGACY = {"coding": ((.9, .05, .05), (5, 40), (.001, .05)), "math": ((.9, .05, .05), (5, 40), (.001, .05)),
          "ifeval": ((.4, .4, .2), (1, 6), (.0002, .01)), "knowledge": ((.6, .2, .2), (2, 15), (.0005, .02))}
with st.expander("Secondary view: weighted 0–100 score and letter grade"):
    w, ttc, ctc = LEGACY[category]
    sec = T[T["eligible"]].copy()
    sec["score"] = [float(weighted_score(p, t, c, w, ttc, ctc)) for p, t, c in zip(sec["p"], sec["t"], sec["c_run"])]
    sec["grade"] = sec["score"].map(grade_of)
    st.dataframe(pd.DataFrame({"Model": sec["model"], "Score (0–100)": sec["score"].round(1),
                               "Grade": sec["grade"]}), hide_index=True, use_container_width=True)
    st.caption(f"From the LLM Measurements doc: weights {w[0]}/{w[1]}/{w[2]} (accuracy/time/cost) for this "
               "category, fixed time and cost targets. Expected cost above is the main ranking; this is for "
               "people who prefer a single 0–100 number.")
