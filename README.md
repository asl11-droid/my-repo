# Right-Size Model Bench (Cheap Thrills · CS 124H)

Streamlit dashboard. The homescreen asks what you need a model for (type it, or search 61 common tasks).
The results page recommends the model with the **lowest expected cost per task**, following
*Actuarial Cost Model — Sprint 3 (revised)*:

    E[C] = cost to run + chance of a wrong answer × cost of that wrong answer + response time × value of time

with sliders for your hourly rate, minutes to fix a wrong answer, and how much waiting costs you.

## Files
- `streamlit_app.py`: the dashboard (homescreen + results page)
- `costmodel.py`: expected cost, Wilson ranges, break-even, paired-bootstrap tie check, the four categories.
  Run `python costmodel.py` to check it against every number in the cost-model doc.
- `tasks.py`: task list and matching typed tasks to a category (coding, knowledge, math, instruction following)
- `scoring.py`: the older weighted 0–100 score (LLM Measurements doc), shown as a secondary view
- `models.json`: settings file. API models: price per 1M tokens. Local models: `gpu_watts`, `price_per_kwh`.
- `make_fake_data.py`: writes sample `data/results.csv` for the models in models.json
- `data/results.csv`: one row per model / test set / question / run:
  `model, test_set, question_id, run, correct, time_s, input_tokens, output_tokens`

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Add a model (Sprint 8)
Add an entry to `models.json`, run its benchmarks (or `python make_fake_data.py`), push.
