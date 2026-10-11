# Right-Size Model Bench (Cheap Thrills · CS 124H)

A Streamlit dashboard. The homescreen asks what the user needs a model for. The grading page then
says "X model is the best for this task because of data from these tests", grades every model
A+ to F, and shows scores with 95% confidence intervals (LLM Measurements doc).

## Files
- `streamlit_app.py`: the dashboard (start page + grading page)
- `scoring.py`: scoring, task types, targets, minimum accuracies, grade scale, task matching.
  Run `python scoring.py` to check it against the doc's worked examples.
- `tasks.py`: the searchable task list on the homescreen (61 tasks) and the matching for typed-in tasks. Add rows to `TASK_CATALOG` to grow the list.
- `models.json`: the settings file. One entry per model with prices per 1M tokens (null for local).
  To add a model, add an entry here; no code changes.
- `make_fake_data.py`: writes fake `data/results.csv` for the models in models.json
- `data/results.csv`: one row per model / test set / question / run:
  `model, test_set, question_id, run, correct, time_s, input_tokens, output_tokens`
  (dollar cost is computed from models.json prices)

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Add a 4th model (Sprint 8)
1. Add an entry to `models.json`.
2. Run the benchmark for it (or `python make_fake_data.py` for fake data).
3. Push to GitHub; the live site updates.

## Still to decide (placeholders in scoring.py)
Time / cost targets and cutoffs per task type, grade cutoffs, minimum accuracies after first runs.
