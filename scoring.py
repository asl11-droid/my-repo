"""Scoring logic from the LLM Measurements doc (revised version).

Pure Python + pandas/numpy, no Streamlit, so it can be unit-tested on its own.

Score = 100 x (w_acc * accuracy + w_time * time_score + w_cost * cost_score)
time_score = (cutoff - time) / (cutoff - target), clipped to [0, 1]
cost_score = (cutoff - cost_per_correct) / (cutoff - target), clipped to [0, 1]
cost_per_correct = cost_per_answer / accuracy
"""
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings. Numbers marked TBD still need a team decision (see the doc).
# ---------------------------------------------------------------------------
TEST_SETS = ["GSM8K", "HumanEval+", "IFEval", "MMLU-Pro"]

TASK_TYPES = {
    "Default": {
        "test_sets": ["MMLU-Pro", "IFEval"],
        "weights": (0.6, 0.2, 0.2),
        # TBD: time in seconds, cost in $ per correct answer, tokens per correct answer
        "time": (2.0, 15.0), "cost": (0.0005, 0.02), "tokens": (500, 1500),
    },
    "Hard stuff (math, coding)": {
        "test_sets": ["GSM8K", "HumanEval+"],
        "weights": (0.9, 0.05, 0.05),
        "time": (5.0, 40.0), "cost": (0.001, 0.05), "tokens": (500, 2500),
    },
    "Chatbot / needs to be fast": {
        "test_sets": ["IFEval"],
        "weights": (0.4, 0.4, 0.2),
        "time": (1.0, 6.0), "cost": (0.0002, 0.01), "tokens": (300, 1200),
    },
    "On a budget": {
        "test_sets": ["MMLU-Pro", "IFEval"],
        "weights": (0.4, 0.2, 0.4),
        "time": (2.0, 15.0), "cost": (0.0005, 0.02), "tokens": (500, 1500),
    },
}

# Proposed minimum accuracy per test set (from the doc; adjust after first runs)
MIN_ACCURACY = {"GSM8K": 0.70, "HumanEval+": 0.50, "IFEval": 0.60, "MMLU-Pro": 0.40}

# TBD: grade cutoffs (score >= cutoff earns the grade)
GRADES = [(90, "A+"), (85, "A"), (80, "A-"), (75, "B+"), (70, "B"), (65, "B-"),
          (60, "C+"), (55, "C"), (50, "C-"), (40, "D"), (0, "F")]


def grade_of(score):
    if score is None or np.isnan(score):
        return "F"
    return next(g for cutoff, g in GRADES if score >= cutoff)


def slide(value, target, cutoff):
    """1 at/better than target, 0 at/worse than cutoff, linear in between. Works on arrays."""
    return np.clip((cutoff - value) / (cutoff - target), 0.0, 1.0)


def score(acc, time_s, cost_per_answer, weights, time_tc, cost_tc):
    """Score for one model. cost_per_answer is $ (or tokens if no model has a price)."""
    cost_per_correct = cost_per_answer / acc
    ts = slide(time_s, *time_tc)
    cs = slide(cost_per_correct, *cost_tc)
    w_acc, w_time, w_cost = weights
    return 100 * (w_acc * acc + w_time * ts + w_cost * cs)


# ---------------------------------------------------------------------------
# From raw results to per-model scores
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["model", "test_set", "question_id", "run", "correct",
                    "time_s", "input_tokens", "output_tokens", "cost_usd"]


def per_question(results: pd.DataFrame) -> pd.DataFrame:
    """Average over runs so there is one row per model / test set / question."""
    df = results.copy()
    df["tokens"] = df["input_tokens"] + df["output_tokens"]
    return (df.groupby(["model", "test_set", "question_id"], as_index=False)
              .agg(correct=("correct", "mean"), time_s=("time_s", "mean"),
                   tokens=("tokens", "mean"), cost_usd=("cost_usd", "mean")))


def evaluate_task(pq: pd.DataFrame, task: dict, weights=None, min_acc=None,
                  n_boot=1000, seed=0) -> pd.DataFrame:
    """Score every model on one task type, with a 95% bootstrap confidence interval.

    Returns one row per model with accuracy, time, cost, score, CI, grade and status.
    """
    weights = weights or task["weights"]
    min_acc = min_acc or MIN_ACCURACY
    sets = task["test_sets"]
    models = sorted(pq["model"].unique())
    rng = np.random.default_rng(seed)

    # Dollars unless some model has no price; then tokens for everyone (doc rule).
    sub = pq[pq["test_set"].isin(sets)]
    use_tokens = sub["cost_usd"].isna().any()
    cost_col = "tokens" if use_tokens else "cost_usd"
    cost_tc = task["tokens"] if use_tokens else task["cost"]

    # Per test set: model x question matrices, and the same bootstrap resample for all models.
    acc_pt, time_pt, cost_pt = [], [], []        # point estimates, list over test sets
    acc_bs, time_bs, cost_bs = [], [], []        # bootstrap draws, (models x n_boot)
    per_set_acc = {}
    applicable = pd.Series(True, index=models)
    warnings.filterwarnings("ignore", "Mean of empty slice")  # models missing a test set -> N/A
    for ts in sets:
        d = sub[sub["test_set"] == ts]
        present = set(d["model"].unique())
        for m in models:
            if m not in present:
                applicable[m] = False            # "not applicable", not zero
        piv = {c: d.pivot(index="model", columns="question_id", values=c).reindex(models)
               for c in ["correct", "time_s", cost_col]}
        C, T, K = (piv[c].to_numpy(dtype=float) for c in ["correct", "time_s", cost_col])
        n = C.shape[1]
        idx = rng.integers(0, n, size=(n_boot, n))
        per_set_acc[ts] = np.nanmean(C, axis=1)
        acc_pt.append(np.nanmean(C, axis=1)); time_pt.append(np.nanmean(T, axis=1)); cost_pt.append(np.nanmean(K, axis=1))
        acc_bs.append(np.nanmean(C[:, idx], axis=2))
        time_bs.append(np.nanmean(T[:, idx], axis=2))
        cost_bs.append(np.nanmean(K[:, idx], axis=2))

    # Task accuracy = average of its test sets' accuracies (same for time and cost).
    acc = np.mean(acc_pt, axis=0); tim = np.mean(time_pt, axis=0); cst = np.mean(cost_pt, axis=0)
    score_pt = score(acc, tim, cst, weights, task["time"], cost_tc)
    score_bs = score(np.mean(acc_bs, axis=0), np.mean(time_bs, axis=0),
                     np.mean(cost_bs, axis=0)[:, :], weights, task["time"], cost_tc)
    lo, hi = np.percentile(score_bs, [2.5, 97.5], axis=1)

    out = pd.DataFrame({
        "model": models, "accuracy": acc, "time_s": tim,
        "cost_per_correct": cst / acc, "cost_unit": "tokens" if use_tokens else "$",
        "score": score_pt, "ci_low": lo, "ci_high": hi,
    })
    for ts in sets:
        out[f"acc_{ts}"] = per_set_acc[ts]

    # Minimum accuracy on EVERY test set in the task type, else removed before scoring.
    passes = np.all([per_set_acc[ts] >= min_acc[ts] for ts in sets], axis=0)
    out["applicable"] = applicable.to_numpy()
    out["eligible"] = passes & out["applicable"]
    out.loc[~out["eligible"], ["score", "ci_low", "ci_high"]] = np.nan
    out["grade"] = [grade_of(s) if e else ("N/A" if not a else "F")
                    for s, e, a in zip(out["score"], out["eligible"], out["applicable"])]

    # Status: winner, tied (CIs overlap with the winner), eligible, or cut.
    out = out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    status = []
    win = out.iloc[0] if out["eligible"].any() else None
    for i, r in out.iterrows():
        if not r["applicable"]:
            status.append("Not applicable")
        elif not r["eligible"]:
            status.append("Below minimum accuracy")
        elif i == 0:
            status.append("Top score")
        elif r["ci_high"] >= win["ci_low"]:
            status.append("Tied with top")
        else:
            status.append("Eligible")
    out["status"] = status
    return out


if __name__ == "__main__":
    # Check against the worked examples in the doc (tokens, time target 1s / cutoff 6s,
    # tokens-per-correct target 500 / cutoff 1500).
    ex = {"X": (0.90, 4.0, 1200), "Y": (0.82, 1.5, 600), "Z": (0.75, 1.0, 800)}
    expected = {"Default": (65.3, 82.6, 73.7), "Hard": (83.8, 82.1, 74.7), "Chatbot": (55.3, 84.2, 78.7)}
    for name, w in [("Default", (.6, .2, .2)), ("Hard", (.9, .05, .05)), ("Chatbot", (.4, .4, .2))]:
        got = tuple(round(float(score(*ex[m], w, (1.0, 6.0), (500, 1500))), 1) for m in "XYZ")
        assert got == expected[name], (name, got, expected[name])
        print(f"{name:8s} X/Y/Z = {got}  OK")
