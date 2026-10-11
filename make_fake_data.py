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

# Fake-only behaviour per model: (skill 0-1, output tokens/sec, verbosity).
# A model in models.json that isn't listed here gets made-up values from its name.
FAKE_BEHAVIOUR = {
    "Claude Opus 5.5":  (0.93, 55, 1.30),
    "Claude Haiku 4.5": (0.82, 150, 0.90),
    "Llama 3.1 8B":     (0.62, 40, 0.85),
}
# test set: (questions, difficulty exponent, avg input tokens, avg output tokens)
TEST_SETS = {
    "GSM8K":      (1319, 0.35, 220, 320),
    "HumanEval+": (164,  1.10, 450, 380),
    "IFEval":     (541,  1.00, 180, 350),
    "MMLU-Pro":   (500,  2.00, 300, 200),   # fixed random subset
}

models = json.loads((HERE / "models.json").read_text())["models"]


def behaviour(name):
    if name in FAKE_BEHAVIOUR:
        return FAKE_BEHAVIOUR[name]
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return 0.70 + (h % 20) / 100, 60 + h % 100, 0.8 + (h % 5) / 10


rows = []
for ts, (n, diff, tin, tout) in TEST_SETS.items():
    q_hard = rng.normal(0, 1.0, n)                      # some questions are harder for everyone
    for m in models:
        skill, tps, verb = behaviour(m["name"])
        base = skill ** diff
        p_q = 1 / (1 + np.exp(-(np.log(base / (1 - base)) - q_hard)))
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
