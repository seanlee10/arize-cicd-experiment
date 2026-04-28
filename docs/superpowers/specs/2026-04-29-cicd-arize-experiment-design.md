# CI/CD pipeline for Arize experiments (SDK v8)

**Date:** 2026-04-29
**Status:** Approved design, ready for implementation plan

## Goal

Stand up a working starter template that runs an Arize experiment on every push to the `main` branch of a GitHub repository, using the Arize Python SDK v8 (`from arize import ArizeClient`). The pipeline gates the build: if the experiment's mean evaluation score falls below a configurable threshold, the workflow exits non-zero and the GitHub Actions check fails.

## Scope

- A single demo experiment (classification with exact-match evaluator) wired end-to-end against an existing Arize dataset that the user supplies via repo configuration.
- The repository is starting empty; this design covers initial repo creation through a working first push.
- Secrets and configuration are managed via GitHub Secrets and Variables — no `.env` checked in.

Out of scope: a real LLM application under test, multiple experiments, dataset bootstrapping (the user has an existing dataset), Slack/PR result-posting hooks.

## Repository layout

```
cicd-experiments/
├── .github/
│   └── workflows/
│       └── arize-experiment.yml
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-29-cicd-arize-experiment-design.md  (this file)
├── experiments/
│   └── classification_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

Single experiment script keeps the example readable top-to-bottom. Additional experiments can be added later as `experiments/<name>_test.py` with their own workflow files (or matrix entries) following the same shape.

## Components

### 1. Experiment script — `experiments/classification_test.py`

Responsibility: load credentials from env vars, define a classification task and an exact-match evaluator, call `client.experiments.run`, and exit with a status code based on whether the mean score meets `EXPERIMENT_THRESHOLD`.

Key shape:

```python
import os, sys
from arize import ArizeClient
from arize.experiments import EvaluationResult
from openai import OpenAI

ARIZE_API_KEY  = os.environ["ARIZE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATASET        = os.environ["ARIZE_DATASET_ID"]   # name or id; SDK v8 accepts either
THRESHOLD      = float(os.environ.get("EXPERIMENT_THRESHOLD", "0.7"))
NAME_PREFIX    = os.environ.get("EXPERIMENT_NAME_PREFIX", "local")
GIT_SHA_SHORT  = os.environ.get("GITHUB_SHA", "local")[:7]

CATEGORIES = ["billing", "technical", "account", "other"]   # adjust to your dataset's labels
TASK_MODEL = "gpt-4o-mini"

arize  = ArizeClient(api_key=ARIZE_API_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

def task(dataset_row) -> str:
    user_input = dataset_row.get("attributes.input.value")   # adjust column to your dataset
    resp = openai.chat.completions.create(
        model=TASK_MODEL,
        temperature=0,
        messages=[
            {"role": "system",
             "content": f"Classify into one of: {', '.join(CATEGORIES)}. Reply with only the label."},
            {"role": "user", "content": user_input},
        ],
    )
    return resp.choices[0].message.content.strip().lower()

def exact_match(output, dataset_row) -> EvaluationResult:
    expected = str(dataset_row.get("attributes.output.value", "")).strip().lower()  # adjust column
    actual   = str(output).strip().lower()
    score = 1 if actual == expected else 0
    return EvaluationResult(
        score=score,
        label="correct" if score else "incorrect",
        explanation=f"expected={expected!r} actual={actual!r}",
    )

def main():
    experiment, df = arize.experiments.run(
        name=f"{NAME_PREFIX}-{GIT_SHA_SHORT}",
        dataset=DATASET,
        task=task,
        evaluators=[exact_match],
        concurrency=10,
    )

    score_cols = [c for c in df.columns if "score" in c.lower()]
    if not score_cols:
        print(f"ERROR: no score column in result df. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)
    mean = df[score_cols[0]].mean()

    print(
        f"experiment={experiment.name if experiment else 'n/a'} "
        f"score_col={score_cols[0]} mean={mean:.3f} threshold={THRESHOLD}"
    )
    sys.exit(0 if mean >= THRESHOLD else 1)

if __name__ == "__main__":
    main()
```

**Hand-edit points before first push:**
- The two `dataset_row.get(...)` keys — must match the dataset's actual column names. Defaults assume OpenInference attribute paths (`attributes.input.value`, `attributes.output.value`).
- The `CATEGORIES` list — replace with the dataset's real label set.

**Defensive score-column scan.** The exact column name on the returned `experiment_df` is not pinned in the v8 docs, and `client.experiments` is currently in beta. Scanning for any column whose name contains `score` is more robust than hardcoding a name and breaking on the first run. If the script ever finds no score column, it exits with code 2 and prints the dataframe columns so the user can adjust.

### 2. GitHub Actions workflow — `.github/workflows/arize-experiment.yml`

```yaml
name: Arize Experiment - main

on:
  push:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: arize-experiment-main
  cancel-in-progress: false

jobs:
  run-experiment:
    runs-on: ubuntu-latest
    env:
      ARIZE_API_KEY:          ${{ secrets.ARIZE_API_KEY }}
      OPENAI_API_KEY:         ${{ secrets.OPENAI_API_KEY }}
      ARIZE_DATASET_ID:       ${{ vars.ARIZE_DATASET_ID }}
      EXPERIMENT_THRESHOLD:   "0.7"
      EXPERIMENT_NAME_PREFIX: "main-push"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python experiments/classification_test.py
```

**Trigger:** push to `main`, plus `workflow_dispatch` for manual reruns from the Actions tab.

**Concurrency group** prevents two simultaneous runs against the same dataset (which would produce confusing experiment ordering). `cancel-in-progress: false` lets the in-flight run finish — we never abandon a half-uploaded experiment.

**Secrets vs. variables.** API keys are GitHub Secrets. `ARIZE_DATASET_ID` is a GitHub *Variable* — it isn't sensitive and putting it in `vars` keeps the workflow legible.

**Threshold** is an env var on the workflow, not a secret, so it can be tuned with one YAML edit and shows up clearly in run logs.

### 3. Dependencies — `requirements.txt`

Strict version pinning for reproducible CI runs:

```
arize==<exact 8.x version>
openai==<exact version>
```

The exact versions are determined during initial local setup (see README) by installing `arize` and `openai` into a fresh virtual environment and copying the resolved versions out of `pip freeze`. We pin the actual versions used during development rather than guessing in this design — this keeps the spec honest about not fabricating package versions that may not exist or may differ from what's currently shipped.

### 4. README — `README.md`

Three sections:

1. **What this is.** One paragraph: a starter pipeline that runs an Arize experiment on every push to `main` and fails the build if the mean evaluation score drops below the configured threshold. SDK v8.

2. **Setup checklist.** Concrete, numbered:
   1. Create a Python 3.13 virtual environment and install pinned dependencies: `python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
   2. Configure repository on GitHub:
      - Secret `ARIZE_API_KEY`
      - Secret `OPENAI_API_KEY`
      - Variable `ARIZE_DATASET_ID`
   3. Edit `experiments/classification_test.py`:
      - `CATEGORIES` — your label set
      - `dataset_row.get("attributes.input.value")` — input column
      - `dataset_row.get("attributes.output.value", "")` — expected-label column
   4. (Optional) tune `EXPERIMENT_THRESHOLD` in the workflow YAML.
   5. Push to `main` to trigger the first run.

3. **Notes.**
   - `client.experiments` is currently in beta in SDK v8 — a one-time warning will appear in CI logs.
   - Score-column detection is intentionally permissive (scans for `*score*`); if you add evaluators that don't emit a score, adjust accordingly.

### 5. `.gitignore`

Standard Python: `__pycache__/`, `*.pyc`, `.venv/`, `.env`, `.pytest_cache/`.

## Configuration surface

All runtime configuration is environment-variable-driven. There is no config file.

| Variable                  | Source                | Required | Purpose                                         |
| ------------------------- | --------------------- | -------- | ----------------------------------------------- |
| `ARIZE_API_KEY`           | GitHub Secret         | yes      | Auth for `ArizeClient`                          |
| `OPENAI_API_KEY`          | GitHub Secret         | yes      | Auth for the LLM call inside the task           |
| `ARIZE_DATASET_ID`        | GitHub Variable       | yes      | Target dataset (name or ID accepted by v8)      |
| `EXPERIMENT_THRESHOLD`    | Workflow env (string) | no       | Pass/fail threshold; defaults to `0.7`          |
| `EXPERIMENT_NAME_PREFIX`  | Workflow env (string) | no       | Prefix for experiment name; defaults to `local` |
| `GITHUB_SHA`              | Set by GitHub Actions | no       | First 7 chars used as experiment name suffix    |

## Data flow

1. GitHub Actions runner is provisioned on push to `main` (or manual dispatch).
2. Repo is checked out, Python 3.13 is set up, dependencies are installed from the pinned `requirements.txt`.
3. `experiments/classification_test.py` runs:
   - Reads credentials and config from environment variables.
   - Calls `arize.experiments.run(...)` which downloads the dataset, executes `task()` per example with concurrency 10, runs `exact_match()` against each output, and uploads results to Arize.
   - Computes the mean score from the returned dataframe.
   - Exits 0 if mean ≥ threshold, 1 if below, 2 if the dataframe shape is unexpected.
4. GitHub marks the workflow run pass/fail based on the exit code. Failures are visible as a red check on the commit on GitHub.

## Error handling

- **Missing required env vars** — script raises `KeyError` immediately on import-level reads (`os.environ["..."]`). This is the desired behavior: fail fast with a clear traceback rather than running a degraded experiment.
- **Unexpected dataframe shape** — exit code 2 with the column list printed to stderr. Distinct from "experiment ran and scored low" (exit 1) so CI logs can distinguish setup bugs from genuine quality regressions.
- **OpenAI / Arize API errors** — bubble up as exceptions and fail the workflow naturally. No retry logic in v1; if flakiness becomes an issue, the SDK's own retry behavior plus a workflow-level rerun is the first line of defense.
- **Beta API warning** — emitted by the SDK on first use of `client.experiments`; appears in logs but does not affect exit code.

## Testing

Local smoke test before pushing:

```bash
source .venv/bin/activate
export ARIZE_API_KEY=...
export OPENAI_API_KEY=...
export ARIZE_DATASET_ID=...
python experiments/classification_test.py
```

For iterating on the task/evaluator without writing to Arize, the SDK supports `dry_run=True` and `dry_run_count=N` on `client.experiments.run` — useful during local development but not enabled in CI.

CI itself is the integration test: the first push to `main` validates the workflow, secrets, dataset access, and the column-name assumptions in one shot. If the column names are wrong, the run fails clearly and the user adjusts the two `.get(...)` calls.

## Open considerations / non-goals

- **Branch protection.** Wiring the workflow as a required check on `main` is a GitHub UI configuration step, not part of this repo. The README mentions the option but doesn't automate it.
- **PR-time experiments.** This pipeline runs on `push: branches: [main]` only — it validates `main` after merges, not PRs. Adding a `pull_request` trigger is a small future change but was explicitly out of scope (the user requested push-to-main).
- **Multiple experiments.** Pattern scales by adding more `experiments/*.py` files plus either a matrix in the workflow or additional workflow files. Out of scope for v1.
- **Posting results back to PRs/commits.** The Arize UI is the primary surface for inspecting results. Workflow logs include the experiment name and mean score. A `gh pr comment` step or status-check enrichment is a possible follow-up.

## Implementation order

1. Initialize git repo, create `.gitignore`.
2. Create local Python 3.13 virtualenv, install `arize` and `openai`, capture exact resolved versions into `requirements.txt` via `pip freeze`. (One-time during repo creation; downstream users install from this pinned file.)
3. Write `experiments/classification_test.py`.
4. Write `.github/workflows/arize-experiment.yml`.
5. Write `README.md`.
6. Initial commit.
7. (User) push to GitHub remote, configure secrets/variable, push to `main` to trigger first run.
