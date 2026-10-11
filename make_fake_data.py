"""Generate data/results.csv with fake benchmark results in the agreed format.

Models come from models.json, so adding a model there and re-running this script
adds it to the fake data too. Replace results.csv with the real runner's output later.
Run:  python make_fake_data.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(42)
RUNS = 5

# Fake-only behaviour per model: target accuracy on each test set, output tokens/sec, verbosity.
# Tuned so different models win different categories at the default slider settings
# (sample data only; real results come from the benchmark runner).
FAKE_BEHAVIOUR = {
    "Claude Opus 5.5":  ({"GSM8K": .96, "HumanEval+": .90, "IFEval": .88, "MMLU-Pro": .80}, 55, 1.30),
    "Claude Haiku 4.5": ({"GSM8K": .95, "HumanEval+": .78, "IFEval": .87, "MMLU-Pro": .78}, 150, 0.90),
    "Llama 3.1 8B":     ({"GSM8K": .80, "HumanEval+": .55, "IFEval": .84, "MMLU-Pro": .52}, 60, 0.85),
}
# test set: (questions, avg input tokens, avg output tokens)
TEST_SETS = {
    "GSM8K":      (1319, 220, 320),
    "HumanEval+": (164,  450, 380),
    "IFEval":     (541,  180, 350),
    "MMLU-Pro":   (500,  300, 200),   # fixed random subset
}

models = json.loads((HERE / "models.json").read_text())["models"]


def behaviour(name):
    if name in FAKE_BEHAVIOUR:
        return FAKE_BEHAVIOUR[name]
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    acc = 0.65 + (h % 25) / 100
    return {ts: acc for ts in TEST_SETS}, 60 + h % 100, 0.8 + (h % 5) / 10


def per_question_probs(target, q_hard):
    """Per-question chance of a correct answer whose average equals `target`."""
    lo, hi = -10.0, 10.0
    for _ in range(60):                      # bisection on the logit offset
        mid = (lo + hi) / 2
        mean = (1 / (1 + np.exp(-(mid - q_hard)))).mean()
        lo, hi = (mid, hi) if mean < target else (lo, mid)
    return 1 / (1 + np.exp(-(lo - q_hard)))


rows = []
for ts, (n, tin, tout) in TEST_SETS.items():
    q_hard = rng.normal(0, 1.0, n)                      # some questions are harder for everyone
    for m in models:
        accs, tps, verb = behaviour(m["name"])
        p_q = per_question_probs(accs[ts], q_hard)
        prefill = 1500 if m["local"] else 12000         # local models read the prompt slower
        for run in range(1, RUNS + 1):
            correct = (rng.random(n) < p_q).astype(int)
            in_tok = np.maximum(20, rng.normal(tin, tin * 0.2, n)).round().astype(int)
            out_tok = np.maximum(5, rng.normal(tout * verb, tout * verb * 0.3, n)).round().astype(int)
            time_s = 0.3 + in_tok / prefill + out_tok / tps * rng.uniform(0.85, 1.15, n)
            rows.append(pd.DataFrame({
                "model": m["name"], "test_set": ts, "question_id": [f"{ts}-{i:04d}" for i in range(n)],
                "run": run, "correct": correct, "time_s": time_s.round(3),
                "input_tokens": in_tok, "output_tokens": out_tok,
            }))

df = pd.concat(rows, ignore_index=True)
(HERE / "data").mkdir(exist_ok=True)
df.to_csv(HERE / "data" / "results.csv", index=False)
print(f"Wrote data/results.csv: {len(df):,} rows for {len(models)} models")
