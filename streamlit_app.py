"""LLM benchmark dashboard. Run locally with:  streamlit run streamlit_app.py"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scoring import (GRADES, MIN_ACCURACY, REQUIRED_COLUMNS, TASK_TYPES, TEST_SETS,
                     evaluate_task, per_question)

st.set_page_config(page_title="Right-Size Model Bench", layout="wide")

GRADE_COLORS = {"A": "#12805c", "B": "#2a78d6", "C": "#b87400", "D": "#c23b3a", "F": "#c23b3a", "N": "#7a868e"}


@st.cache_data
def load_results(path=Path(__file__).parent / "data" / "results.csv"):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"results.csv is missing columns: {', '.join(missing)}")
        st.stop()
    return per_question(df)


@st.cache_data
def run_eval(pq, task_name, weights, min_acc_items, n_boot):
    return evaluate_task(pq, TASK_TYPES[task_name], weights=weights,
                         min_acc=dict(min_acc_items), n_boot=n_boot)


pq = load_results()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    task_name = st.selectbox("Task type", list(TASK_TYPES))
    task = TASK_TYPES[task_name]
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

res = run_eval(pq, task_name, weights, tuple(min_acc.items()), n_boot)
elig = res[res["eligible"]]
unit = res["cost_unit"].iloc[0]
fmt_cost = (lambda v: f"${v:.4f}") if unit == "$" else (lambda v: f"{v:,.0f} tok")

# ---------------- Header + recommendation ----------------
st.title("Right-Size Model Bench")
st.caption("Each model gets a 0–100 score from accuracy, time and cost per correct answer, "
           "measured against fixed targets for the task type, then a letter grade.")

if elig.empty:
    st.error("No model meets the minimum accuracy on every test set for this task type. "
             "Lower a minimum in the sidebar.")
else:
    top = elig.iloc[0]
    col_g, col_txt = st.columns([1, 5])
    col_g.markdown(
        f"<div style='font-size:56px;font-weight:700;color:{GRADE_COLORS[top['grade'][0]]};"
        f"line-height:1'>{top['grade']}</div>", unsafe_allow_html=True)
    col_txt.subheader(f"Recommended: {top['model']}")
    col_txt.write(f"Score **{top['score']:.1f}** (95% CI: {top['ci_low']:.1f} to {top['ci_high']:.1f}) "
                  f"on **{task_name}**.")
    ties = res[res["status"] == "Tied with top"]
    if not ties.empty:
        st.warning("Statistical tie: the confidence intervals of " + ", ".join(ties["model"]) +
                   f" overlap with {top['model']}'s. Treat them as equal and choose on cost or speed.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{top['accuracy']:.2f}")
    m2.metric("Time per answer", f"{top['time_s']:.1f}s")
    m3.metric("Cost per correct answer", fmt_cost(top["cost_per_correct"]))

# ---------------- Score chart ----------------
st.subheader("Scores with 95% confidence intervals")
chart = res.iloc[::-1]  # best at the top
fig = go.Figure(go.Bar(
    x=chart["score"].fillna(0), y=chart["model"], orientation="h",
    marker_color=[GRADE_COLORS[g[0]] for g in chart["grade"]],
    error_x=dict(type="data", array=(chart["ci_high"] - chart["score"]).fillna(0),
                 arrayminus=(chart["score"] - chart["ci_low"]).fillna(0), color="#555"),
    text=[f"{g}  {s:.1f}" if pd.notna(s) else f"{g}  (cut)" for g, s in zip(chart["grade"], chart["score"])],
    textposition="outside",
    hovertemplate="%{y}<br>Score %{x:.1f}<extra></extra>",
))
fig.update_layout(xaxis=dict(range=[0, 105], title="Score"), height=60 * len(chart) + 80,
                  margin=dict(l=10, r=10, t=10, b=40))
st.plotly_chart(fig, use_container_width=True)

# ---------------- Table ----------------
st.subheader("Scorecard")
table = res[["grade", "model", "status", "score", "ci_low", "ci_high", "accuracy",
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
st.dataframe(card.style.map(lambda g: f"color:{GRADE_COLORS[g[0]]};font-weight:700"),
             use_container_width=True)

with st.expander("Grade scale and targets"):
    st.write("Grades: " + ", ".join(f"{g} ≥ {c}" for c, g in GRADES[:-1]) + ", F below that "
             "or below minimum accuracy.")
    st.write(f"{task_name} targets → time: {task['time'][0]}s target, {task['time'][1]}s cutoff; "
             f"cost per correct: ${task['cost'][0]} target, ${task['cost'][1]} cutoff.")
    st.caption("Targets, cutoffs and grade scale are placeholders until the team agrees on them (see scoring.py).")
