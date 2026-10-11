"""Expected cost per task, from "Actuarial Cost Model — Sprint 3 (revised)".

    E[C] = C_run + p_fail * C_fail + t * v_time

    C_run   API:   (n_in * p_in + n_out * p_out) / 1e6        (prices per 1M tokens)
            local: P_gpu * t / 3.6e6 * p_kWh                    (energy)
    p_fail  1 - accuracy; accuracy = mean over distinct questions of s_q,
            s_q = fraction of a question's runs that were correct
    C_fail  minutes to detect and fix * R / 60
    v_time  R / 3600 * attention factor alpha
    t       median latency over repeated runs

Ranges: Wilson 95% interval on accuracy (n = distinct questions) -> E[C] range.
Ties:   paired bootstrap over questions on the difference in E[C]; if the 95% range
        contains 0 the models are tied, and the lower point estimate is still
        recommended, labeled "tied, low confidence".
"""
from math import sqrt

import numpy as np
import pandas as pd

RATE_PER_HOUR = 30.0   # R, shared by failure cost and time cost

# category id -> label, benchmark, default fix minutes, slider range (min), minimum accuracy
CATEGORIES = {
    "coding":    {"label": "Coding", "benchmark": "HumanEval+", "minutes": 10, "range": (2.0, 50.0),
                  "min_acc": 0.50, "why": "Broken code must be debugged; bugs can hide until later."},
    "knowledge": {"label": "General knowledge", "benchmark": "MMLU-Pro", "minutes": 3, "range": (1.0, 20.0),
                  "min_acc": 0.40, "why": "Wrong facts are hard to spot; checking needs a lookup."},
    "math":      {"label": "Math", "benchmark": "GSM8K", "minutes": 2, "range": (0.5, 10.0),
                  "min_acc": 0.70, "why": "The final number is easy to recheck, but the error may propagate."},
    "ifeval":    {"label": "Instruction following", "benchmark": "IFEval", "minutes": 1, "range": (0.2, 4.0),
                  "min_acc": 0.60, "why": "Format violations are visible; usually a quick edit or rerun."},
}

ATTENTION = {  # preset -> alpha
    "Interactive (chat, pair coding)": 1.0,
    "Semi-interactive (I switch tabs)": 0.3,
    "Batch / background": 0.1,
}


# ---------------------------------------------------------------------------
# The formulas (same as the doc's code sketch)
# ---------------------------------------------------------------------------
def c_fail(minutes, rate_per_hour=RATE_PER_HOUR):
    return minutes * rate_per_hour / 60


def v_time(rate_per_hour=RATE_PER_HOUR, attention=1.0):
    return rate_per_hour / 3600 * attention


def run_cost_api(n_in, n_out, price_in_per_m, price_out_per_m):
    return (n_in * price_in_per_m + n_out * price_out_per_m) / 1e6


def run_cost_local(gpu_watts, seconds, price_per_kwh=0.15):
    return gpu_watts * seconds / 3.6e6 * price_per_kwh


def wilson(p, n, z=1.96):
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def break_even(a, b, v):
    """C_fail at which models a and b have equal E[C]. a, b: dicts with c_run, p, t. None if never."""
    fa, fb = a["c_run"] + a["t"] * v, b["c_run"] + b["t"] * v
    if a["p"] == b["p"]:
        return None
    x = (fa - fb) / (a["p"] - b["p"])
    return x if x > 0 else None


def paired_tie_check(s_a, s_b, fixed_a, fixed_b, cf, n_boot=10_000, seed=0):
    """Paired bootstrap on E[C_a] - E[C_b]. s_a, s_b: per-question scores in the same order.
    Returns (diff_point, diff_lo, diff_hi, tied)."""
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    n = len(s_a)
    d = s_b - s_a                                   # per-question p_fail_a - p_fail_b
    fixed = fixed_a - fixed_b
    point = fixed + cf * d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
    boots = fixed + cf * d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, bool(lo <= 0 <= hi)


# ---------------------------------------------------------------------------
# From raw results to a ranking for one category
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["model", "test_set", "question_id", "run", "correct",
                    "time_s", "input_tokens", "output_tokens"]


def summarize(results: pd.DataFrame, models: dict) -> pd.DataFrame:
    """One row per model x test set: accuracy, n, median time, mean tokens, run cost, stability."""
    df = results[results["model"].isin(models)]
    rows = []
    for (m, ts), g in df.groupby(["model", "test_set"]):
        s_q = g.groupby("question_id")["correct"].mean()
        unanimous = ((s_q == 0) | (s_q == 1)).mean()
        t = float(g["time_s"].median())
        n_in, n_out = float(g["input_tokens"].mean()), float(g["output_tokens"].mean())
        cfg = models[m]
        if cfg.get("local"):
            c_run = run_cost_local(cfg.get("gpu_watts", 150), t, cfg.get("price_per_kwh", 0.15))
        else:
            c_run = run_cost_api(n_in, n_out, cfg["price_in_per_1m"], cfg["price_out_per_1m"])
        rows.append({"model": m, "test_set": ts, "p": float(s_q.mean()), "n": int(len(s_q)),
                     "t": t, "n_in": n_in, "n_out": n_out, "c_run": c_run,
                     "unstable": 1 - float(unanimous), "scores": s_q})
    return pd.DataFrame(rows)


def rank_category(summary: pd.DataFrame, category: str, rate=RATE_PER_HOUR, minutes=None,
                  attention=1.0, min_acc=None, n_boot=10_000) -> dict:
    """Rank models for one category by E[C]. Returns a dict with the table and the inputs used."""
    cat = CATEGORIES[category]
    minutes = cat["minutes"] if minutes is None else minutes
    min_acc = cat["min_acc"] if min_acc is None else min_acc
    cf, v = c_fail(minutes, rate), v_time(rate, attention)
    s = summary[summary["test_set"] == cat["benchmark"]].copy()
    if s.empty:
        return {"table": s, "cf": cf, "v": v, "minutes": minutes, "min_acc": min_acc, "top": None}

    s["fixed"] = s["c_run"] + s["t"] * v
    s["mistake"] = (1 - s["p"]) * cf
    s["wait"] = s["t"] * v
    s["ec"] = s["fixed"] + s["mistake"]
    rng_ = [wilson(p, n) for p, n in zip(s["p"], s["n"])]
    s["p_lo"], s["p_hi"] = [r[0] for r in rng_], [r[1] for r in rng_]
    s["ec_best"] = s["fixed"] + (1 - s["p_hi"]) * cf
    s["ec_worst"] = s["fixed"] + (1 - s["p_lo"]) * cf
    s["eligible"] = s["p"] >= min_acc
    s = s.sort_values(["eligible", "ec"], ascending=[False, True]).reset_index(drop=True)

    top = s.iloc[0] if s["eligible"].any() else None
    s["diff_lo"] = s["diff_hi"] = np.nan
    status = []
    for i, r in s.iterrows():
        if not r["eligible"]:
            status.append("Below minimum accuracy")
            continue
        if i == 0:
            status.append("Cheapest")
            continue
        a, b = r["scores"].align(top["scores"], join="inner")
        _, lo, hi, tied = paired_tie_check(a.values, b.values, r["fixed"], top["fixed"], cf, n_boot)
        s.at[i, "diff_lo"], s.at[i, "diff_hi"] = lo, hi
        status.append("Tied with cheapest" if tied else "Costs more")
    s["status"] = status
    if top is not None and (s["status"] == "Tied with cheapest").any():
        s.loc[0, "status"] = "Cheapest (tied, low confidence)"
    return {"table": s, "cf": cf, "v": v, "minutes": minutes, "min_acc": min_acc,
            "top": s.iloc[0] if top is not None else None}


def explain(result: dict, category: str, models: dict, runs: int) -> str:
    """'Model X is the best choice for this task because of results from these tests: ...'"""
    cat, s, top = CATEGORIES[category], result["table"], result["top"]
    if s.empty:
        return (f"There are no {cat['benchmark']} results yet, so there's no recommendation for "
                f"{cat['label'].lower()}. Run that benchmark, then refresh.")
    if top is None:
        return (f"No model reaches the minimum accuracy of {result['min_acc']:.2f} on {cat['benchmark']}, "
                "so there's no recommendation. Lower the minimum in the sidebar or add a stronger model.")
    txt = (f"**{top['model']}** is the best choice for this task because of results from these tests: "
           f"on {cat['benchmark']} ({top['n']:,} questions, {runs} runs each) it answered "
           f"**{top['p']:.0%}** correctly, with a median response time of {top['t']:.1f}s. "
           f"It costs you about **${top['ec']:.2f} per task, mistakes included**: "
           f"${top['c_run']:.4f} to run, ${top['mistake']:.2f} in expected mistakes and "
           f"${top['wait']:.3f} of waiting.")
    others = s[s["eligible"]].iloc[1:]
    for _, r in others.iterrows():
        be = break_even({"c_run": r["c_run"], "p": r["p"], "t": r["t"]},
                        {"c_run": top["c_run"], "p": top["p"], "t": top["t"]}, result["v"])
        if r["status"] == "Tied with cheapest":
            txt += (f" **Tied, low confidence** with {r['model']}: the 95% range on the cost difference "
                    f"(${r['diff_lo']:+.2f} to ${r['diff_hi']:+.2f}) includes zero. More questions would settle it.")
        elif be is not None and r["p"] < top["p"]:
            txt += f" {r['model']} beats it only if a mistake costs you less than ${be:.2f}."
        elif be is not None and r["p"] > top["p"]:
            txt += f" {r['model']} beats it only if a mistake costs you more than ${be:.2f}."
    for _, r in s[~s["eligible"]].iterrows():
        txt += (f" {r['model']} was left out: {r['p']:.0%} accuracy is below the "
                f"{result['min_acc']:.0%} minimum.")
    return txt


if __name__ == "__main__":
    # Check against every number in the cost-model doc's worked example.
    v = v_time()
    X = {"c_run": run_cost_api(200, 1000, 3, 15), "p": 0.90, "t": 4.0}
    Y = {"c_run": run_cost_api(200, 400, 0.15, 0.60), "p": 0.80, "t": 1.5}
    Z = {"c_run": run_cost_local(150, 1.0), "p": 0.75, "t": 1.0}
    ec = {k: m["c_run"] + (1 - m["p"]) * 5 + m["t"] * v for k, m in zip("XYZ", (X, Y, Z))}
    assert [round(ec[k], 2) for k in "XYZ"] == [0.55, 1.01, 1.26], ec
    assert round(X["c_run"], 4) == 0.0156 and round(Y["c_run"], 5) == 0.00027
    assert round(break_even(Y, X, v), 2) == 0.36 and round(break_even(Z, Y, v), 2) == 0.09
    assert [round(x, 2) for x in wilson(0.9, 20)] == [0.70, 0.97]
    assert [round(x, 2) for x in wilson(0.8, 200)] == [0.74, 0.85]
    fx, fy = X["c_run"] + X["t"] * v, Y["c_run"] + Y["t"] * v
    sx = [1] * 15 + [1, 1, 1] + [0] + [0]          # X right on 18
    sy = [1] * 15 + [0, 0, 0] + [1] + [0]          # Y right on 16; 3 only-X, 1 only-Y
    pt, lo, hi, tied = paired_tie_check(sx, sy, fx, fy, 5.0)
    assert round(pt, 2) == -0.46 and tied, (pt, lo, hi)
    pt2, lo2, hi2, tied2 = paired_tie_check(sx * 10, sy * 10, fx, fy, 5.0)
    assert not tied2, (lo2, hi2)
    print(f"E[C] X/Y/Z = {[round(ec[k], 2) for k in 'XYZ']}  break-even Y/X $0.36, Z/Y $0.09  OK")
    print(f"tie n=20: {pt:+.2f} ({lo:+.2f} to {hi:+.2f}) tied;  n=200: {pt2:+.2f} ({lo2:+.2f} to {hi2:+.2f}) X cheaper  OK")
