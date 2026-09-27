"""Generate data/results.csv with fake benchmark results in the agreed format.

One row per model / test set / question / run. Replace with the real runner's output.
Run:  python make_fake_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
RUNS = 5

# name: (skill 0-1, $ per 1M input, $ per 1M output, output tokens/sec, verbosity)
MODELS = {
    "Model A (large)":  (0.93, 15.0, 75.0,  55, 1.35),
    "Model B (medium)": (0.90,  3.0, 15.0,  85, 1.10),
    "Model C (large)":  (0.88,  5.0, 20.0,  70, 1.20),
    "Model D (small)":  (0.82,  1.0,  5.0, 150, 0.90),
    "Model E (mini)":   (0.75,  0.4,  1.6, 160, 0.85),
}
# test set: (questions, difficulty exponent, avg input tokens, avg output tokens)
TEST_SETS = {
    "GSM8K":      (1319, 0.35, 220, 320),
    "HumanEval+": (164,  1.10, 450, 380),
    "IFEval":     (541,  1.00, 180, 350),
    "MMLU-Pro":   (500,  2.00, 300, 200),   # fixed random subset
}

rows = []
for ts, (n, diff, tin, tout) in TEST_SETS.items():
    q_hard = rng.normal(0, 1.0, n)                      # some questions are harder for everyone
    for model, (skill, pin, pout, tps, verb) in MODELS.items():
        base = skill ** diff
        logit = np.log(base / (1 - base)) - q_hard
        p_q = 1 / (1 + np.exp(-logit))
        for run in range(1, RUNS + 1):
            correct = (rng.random(n) < p_q).astype(int)
            in_tok = np.maximum(20, rng.normal(tin, tin * 0.2, n)).round().astype(int)
            out_tok = np.maximum(5, rng.normal(tout * verb, tout * verb * 0.3, n)).round().astype(int)
            time_s = 0.4 + in_tok / 12000 + out_tok / tps * rng.uniform(0.85, 1.15, n)
            cost = (in_tok * pin + out_tok * pout) / 1e6
            rows.append(pd.DataFrame({
                "model": model, "test_set": ts, "question_id": [f"{ts}-{i:04d}" for i in range(n)],
                "run": run, "correct": correct, "time_s": time_s.round(3),
                "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost.round(7),
            }))

df = pd.concat(rows, ignore_index=True)
df.to_csv("data/results.csv", index=False)
print(f"Wrote data/results.csv: {len(df):,} rows")
