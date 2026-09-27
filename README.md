# Right-Size Model Bench (Streamlit dashboard)

Scores each LLM 0–100 from accuracy, time and cost per correct answer (LLM Measurements doc), with a 95% bootstrap confidence interval and a letter grade.

## Files
- `streamlit_app.py`: the dashboard (front end)
- `scoring.py`: scoring logic, targets, minimums, grade scale. Run `python scoring.py` to check it against the doc's examples.
- `make_fake_data.py`: writes fake `data/results.csv` until the real runner exists
- `data/results.csv`: one row per model / test set / question / run:
  `model, test_set, question_id, run, correct, time_s, input_tokens, output_tokens, cost_usd`
  (`cost_usd` empty for local models)

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Deploy
Push this folder to GitHub, then on share.streamlit.io: Deploy a public app from GitHub →
repo, branch, main file path `streamlit_app.py`.

## Still to decide (placeholders in scoring.py)
Time / cost targets and cutoffs per task type, grade cutoffs, minimum accuracies after first runs.
