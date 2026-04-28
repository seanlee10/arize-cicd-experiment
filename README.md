# cicd-experiments

A starter pipeline that runs an Arize experiment on every push to `main` (and on manual dispatch) using the Arize Python SDK v8. The build fails if the experiment's mean evaluation score drops below the configured threshold.

## What runs

A classification task with an exact-match evaluator. On each run:

1. GitHub Actions checks out the repo and installs pinned dependencies into Python 3.13.
2. `experiments/classification_test.py` calls `client.experiments.run`, which downloads the dataset, runs the task on each example, evaluates each output, and uploads results to Arize.
3. The script computes the mean score and exits `0` if it meets `EXPERIMENT_THRESHOLD`, `1` if below, `2` on unexpected dataframe shape.

## Setup

1. **Local environment** — Python 3.13 + pinned deps:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure the GitHub repository:**
   - Add Secret `ARIZE_API_KEY`
   - Add Secret `OPENAI_API_KEY`
   - Add Variable `ARIZE_DATASET_ID` (the dataset name or ID from Arize)

3. **Edit `experiments/classification_test.py` to match your dataset:**
   - `CATEGORIES` — your label set
   - `dataset_row.get("attributes.input.value")` — the input column on your dataset
   - `dataset_row.get("attributes.output.value", "")` — the expected-label column

4. (Optional) Tune `EXPERIMENT_THRESHOLD` in `.github/workflows/arize-experiment.yml`.

5. Push to `main` to trigger the first run, or run manually from the Actions tab via *workflow_dispatch*.

## Local smoke test

```bash
source .venv/bin/activate
export ARIZE_API_KEY=...
export OPENAI_API_KEY=...
export ARIZE_DATASET_ID=...
python experiments/classification_test.py
```

For iterating without uploading results to Arize, use `dry_run=True` and `dry_run_count=N` on `client.experiments.run` (see Arize SDK v8 docs).

## Notes

- `client.experiments` is in **beta** in SDK v8. A one-time warning is emitted on first use; this is expected and does not affect the exit code.
- The script scans for any column whose name contains `score` to find the evaluator's score column. If you add evaluators that don't emit a score, adjust `score_cols` in `main()` accordingly.
