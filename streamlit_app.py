"""LLM benchmark dashboard. Run locally with:  streamlit run streamlit_app.py"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scoring import (GRADES, MIN_ACCURACY, REQUIRED_COLUMNS, TASK_TYPES, TEST_SETS,
                     evaluate_task, explain, load_models, per_question)
from tasks import CATALOG_LABELS, CATALOG_TYPE, TASK_CATALOG, match_task

st.set_page_config(page_title="Right-Size Model Bench", page_icon="⚖️", layout="wide")

HERE = Path(__file__).parent
GRADE_COLORS = {"A": "#12805c", "B": "#2a78d6", "C": "#b87400", "D": "#c23b3a", "F": "#c23b3a", "N": "#7a868e"}
TASK_BLURBS = {
    "Default": "General questions, writing, following instructions. Accuracy matters most.",
    "Hard stuff (math, coding)": "Math and code, where being right matters far more than speed or cost.",
    "Chatbot / needs to be fast": "Live conversations, where slow answers hurt as much as wrong ones.",
    "On a budget": "Large volumes or tight budgets, where cost per correct answer matters a lot.",
}

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.eyebrow {letter-spacing:.12em; text-transform:uppercase; font-size:12.5px; opacity:.6; margin:0 0 .2rem;}
.hero h1 {font-size: 2.6rem; line-height:1.1; margin:.1rem 0 .6rem;}
.hero p {font-size: 1.08rem; opacity:.8; max-width: 60ch;}
.mcard-name {font-weight:700; font-size:1.05rem; margin:0;}
.mcard-role {display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px;
             background:rgba(42,120,214,.12); color:#2a78d6; margin:.25rem 0 .35rem;}
.mcard-meta {font-size:13px; opacity:.7; margin:0;}
</style>
""", unsafe_allow_html=True)


# ---------------- Data ----------------
@st.cache_data
def load_data(results_path=HERE / "data" / "results.csv", models_path=HERE / "models.json"):
    models = load_models(models_path)
    df = pd.read_csv(results_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"results.csv is missing columns: {', '.join(missing)}")
        st.stop()
    info = {
        "runs": int(df["run"].nunique()),
        "questions": df.groupby("test_set")["question_id"].nunique().to_dict(),
        "not_in_settings": sorted(set(df["model"]) - set(models)),
        "no_results": sorted(set(models) - set(df["model"])),
    }
    return per_question(df, models), models, info


@st.cache_data
def run_eval(pq, task_name, weights, min_acc_items, n_boot):
    return evaluate_task(pq, TASK_TYPES[task_name], weights=weights,
                         min_acc=dict(min_acc_items), n_boot=n_boot)


pq, MODELS, info = load_data()


def label(name):
    """Display label for a model: 'Claude Opus 5.5 · Strong API model'."""
    m = MODELS.get(name)
    return f"{name} · {m['role']}" if m else name


def price_text(m):
    if m["local"]:
        return f"{m['provider']} · runs locally, no per-token price"
    return f"{m['provider']} · ${m['price_in_per_1m']:g} in / ${m['price_out_per_1m']:g} out per 1M tokens"


def start_grading(text, task_type=None, reason=""):
    st.session_state["query"] = text.strip()
    st.session_state["forced_type"] = task_type
    st.session_state["reason"] = reason
    for k in [k for k in st.session_state if str(k).startswith(("task-", "pick-"))]:
        del st.session_state[k]


# ===================================================================
# HOMESCREEN
# ===================================================================
if not st.session_state.get("query"):
    st.markdown("<div class='hero'><p class='eyebrow'>Cheap Thrills · CS 124H · LLM Benchmarking</p>"
                "<h1>Which AI model should you use?</h1>"
                "<p>Describe what you need a model for. We grade every model we've benchmarked on accuracy, "
                "speed and cost, then recommend the one that does the job well without overpaying.</p></div>",
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.form("own-task", border=True):
            st.markdown("**Type your own task**")
            text = st.text_area("Describe your task", label_visibility="collapsed", height=120,
                                placeholder="e.g. A bot that answers students' questions about our course "
                                            "syllabus, or: check my linear algebra homework answers")
            go_own = st.form_submit_button("Grade the models →", type="primary", use_container_width=True)
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
            if st.button("Grade the models for this task", use_container_width=True, disabled=pick is None):
                start_grading(pick.split(" — ", 1)[1], CATALOG_TYPE[pick],
                              f"you picked “{pick.split(' — ', 1)[1]}”")
                st.rerun()
            with st.expander("Browse all tasks by category"):
                cats = {}
                for cat, task, _ in TASK_CATALOG:
                    cats.setdefault(cat, []).append(task)
                for cat, tasks in cats.items():
                    st.markdown(f"**{cat}**")
                    st.caption(" · ".join(tasks))

    st.markdown("#### Models being compared")
    cols = st.columns(len(MODELS) or 1)
    for col, m in zip(cols, MODELS.values()):
        with col.container(border=True):
            st.markdown(f"<p class='mcard-name'>{m['name']}</p><span class='mcard-role'>{m['role']}</span>"
                        f"<p class='mcard-meta'>{price_text(m)}</p>"
                        f"<p class='mcard-meta'><code>{m['model_id']}</code></p>", unsafe_allow_html=True)
    st.caption(f"Benchmarks: {', '.join(f'{ts} ({n:,} questions)' for ts, n in info['questions'].items())}, "
               f"each question run {info['runs']} times. Sample data until the benchmark runner is connected.")
    st.stop()


# ===================================================================
# GRADING PAGE
# ===================================================================
query = st.session_state["query"]
forced = st.session_state.get("forced_type")
if forced:
    matched, reason = forced, st.session_state.get("reason", "")
else:
    matched, reason = match_task(query)
task_names = list(TASK_TYPES)

top_l, top_r = st.columns([5, 1])
top_l.markdown("<p class='eyebrow'>Right-Size Model Bench</p>", unsafe_allow_html=True)
if top_r.button("← New task", use_container_width=True):
    st.session_state["query"] = ""
    st.rerun()
st.markdown(f"### “{query}”")

if matched is None:
    st.info("We couldn't tell what kind of task this is. Which of these is closest?")
    matched = st.radio("Task type", task_names, key=f"pick-{query}", label_visibility="collapsed",
                       format_func=lambda n: f"{n}: {TASK_BLURBS[n]}")
    reason = "you chose it"

with st.sidebar:
    st.header("Settings")
    task_name = st.selectbox("Task type", task_names, index=task_names.index(matched),
                             key=f"task-{query}-{matched}")
    task = TASK_TYPES[task_name]
    st.caption(TASK_BLURBS[task_name])
    st.caption("Test sets: " + ", ".join(task["test_sets"]))

    st.subheader("Weights")
    w0 = task["weights"]
    wa = st.slider("Accuracy", 0.0, 1.0, float(w0[0]), 0.05, key=f"wa-{task_name}")
    wt = st.slider("Time", 0.0, 1.0, float(w0[1]), 0.05, key=f"wt-{task_name}")
    wc = st.slider("Cost", 0.0, 1.0, float(w0[2]), 0.05, key=f"wc-{task_name}")
    total = wa + wt + wc or 1.0
    weights = (wa / total, wt / total, wc / total)
    st.caption(f"Normalized to sum to 1: {weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f}")

    with st.expander("Minimum accuracy per test set"):
        min_acc = {ts: st.number_input(ts, 0.0, 1.0, MIN_ACCURACY[ts], 0.05, key=f"min-{ts}")
                   for ts in TEST_SETS}
    n_boot = st.select_slider("Bootstrap samples", [200, 500, 1000, 2000], value=1000)

    st.subheader("Models (from models.json)")
    for m in MODELS.values():
        st.caption(f"**{m['name']}** · {m['role']}  \n{price_text(m)}")

if task_name == matched:
    st.caption(f"Graded as **{task_name}** because {reason}. {TASK_BLURBS[task_name]} "
               "Change it in the sidebar if that's wrong.")
else:
    st.caption(f"Graded as **{task_name}** (changed in the sidebar). {TASK_BLURBS[task_name]}")
for name in info["not_in_settings"]:
    st.warning(f"'{name}' is in results.csv but not in models.json, so it's hidden. Add it to models.json to show it.")
for name in info["no_results"]:
    st.info(f"'{name}' is in models.json but has no results yet. Run the benchmark for it.")

res = run_eval(pq, task_name, weights, tuple(min_acc.items()), n_boot)
res["label"] = res["model"].map(label)
elig = res[res["eligible"]]
unit = res["cost_unit"].iloc[0]
fmt_cost = (lambda v: f"${v:.4f}") if unit == "$" else (lambda v: f"{v:,.0f} tok")

# ---------------- Recommendation ----------------
with st.container(border=True):
    if elig.empty:
        st.error(explain(res, task_name, task, min_acc, info["runs"], info["questions"]))
    else:
        top = elig.iloc[0]
        col_g, col_txt = st.columns([1, 7])
        col_g.markdown(
            f"<div style='font-size:64px;font-weight:700;color:{GRADE_COLORS[top['grade'][0]]};"
            f"line-height:1.1'>{top['grade']}</div>", unsafe_allow_html=True)
        col_txt.subheader(f"Recommended: {top['model']}")
        col_txt.caption(f"{MODELS[top['model']]['role']} · {price_text(MODELS[top['model']])}")
        col_txt.markdown(explain(res, task_name, task, min_acc, info["runs"], info["questions"]))
        ties = res[res["status"] == "Tied with top"]
        if not ties.empty:
            st.warning("Statistical tie: the confidence intervals of " + ", ".join(ties["model"]) +
                       f" overlap with {top['model']}'s. Treat them as equal and choose on cost or speed.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{top['accuracy']:.2f}")
        m2.metric("Time per answer", f"{top['time_s']:.1f}s")
        m3.metric("Cost per correct answer", fmt_cost(top["cost_per_correct"]))
if unit == "tokens":
    st.caption("ℹ️ One model runs locally with no per-token price, so cost is compared in tokens per correct "
               "answer for every model, as the measurements doc specifies.")

# ---------------- Score chart ----------------
st.subheader("Scores with 95% confidence intervals")
chart = res.iloc[::-1]  # best at the top
fig = go.Figure(go.Bar(
    x=chart["score"].fillna(0), y=chart["label"], orientation="h",
    marker_color=[GRADE_COLORS[g[0]] for g in chart["grade"]],
    error_x=dict(type="data", array=(chart["ci_high"] - chart["score"]).fillna(0),
                 arrayminus=(chart["score"] - chart["ci_low"]).fillna(0), color="#555"),
    text=[f"{g}  {s:.1f}" if pd.notna(s) else f"{g}  ({stat.lower()})"
          for g, s, stat in zip(chart["grade"], chart["score"], chart["status"])],
    textposition="outside",
    hovertemplate="%{y}<br>Score %{x:.1f}<extra></extra>",
))
fig.update_layout(xaxis=dict(range=[0, 110], title="Score (0–100)"), height=70 * len(chart) + 90,
                  margin=dict(l=10, r=10, t=10, b=40))
st.plotly_chart(fig, use_container_width=True)

# ---------------- Table ----------------
st.subheader("Scorecard")
res["role"] = res["model"].map(lambda m: MODELS[m]["role"])
res["provider"] = res["model"].map(lambda m: MODELS[m]["provider"])
table = res[["grade", "model", "role", "provider", "status", "score", "ci_low", "ci_high", "accuracy",
             *[f"acc_{ts}" for ts in task["test_sets"]], "time_s", "cost_per_correct"]].copy()
table = table.rename(columns={"ci_low": "CI low", "ci_high": "CI high", "time_s": "time (s)",
                              "cost_per_correct": f"cost per correct ({unit})",
                              **{f"acc_{ts}": f"acc {ts}" for ts in task["test_sets"]}})
st.dataframe(table.style.format(precision=3, na_rep="—"), hide_index=True, use_container_width=True)

# ---------------- Report card ----------------
st.subheader("Report card")
st.caption("Grade on every task type, using each one's default weights and the minimums above.")
card = {}
for name in TASK_TYPES:
    r = run_eval(pq, name, TASK_TYPES[name]["weights"], tuple(min_acc.items()), 300)
    card[name] = r.set_index("model")["grade"]
card = pd.DataFrame(card)
card.index = [label(m) for m in card.index]
st.dataframe(card.style.map(lambda g: f"color:{GRADE_COLORS[g[0]]};font-weight:700"),
             use_container_width=True)

with st.expander("Grade scale and targets"):
    st.write("Grades: " + ", ".join(f"{g} ≥ {c}" for c, g in GRADES[:-1]) + ", F below that "
             "or below minimum accuracy. N/A means the model has no results for a test set this task needs.")
    cost_tc = task["cost"] if unit == "$" else task["tokens"]
    cost_txt = (f"${cost_tc[0]} target, ${cost_tc[1]} cutoff" if unit == "$"
                else f"{cost_tc[0]:,} tokens target, {cost_tc[1]:,} tokens cutoff")
    st.write(f"{task_name} targets → time: {task['time'][0]}s target, {task['time'][1]}s cutoff; "
             f"cost per correct answer: {cost_txt}.")
    st.caption("Targets, cutoffs and grade scale are placeholders until the team agrees on them (see scoring.py).")
