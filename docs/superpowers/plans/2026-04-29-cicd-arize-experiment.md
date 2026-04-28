# CI/CD Arize Experiment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a working starter repository that runs an Arize experiment (Python SDK v8) on every push to `main` via GitHub Actions, gating the build on a configurable mean-score threshold.

**Architecture:** A single Python script (`experiments/classification_test.py`) defines a classification task and an exact-match evaluator, calls `arize.experiments.run`, and exits non-zero if the mean score falls below `EXPERIMENT_THRESHOLD`. A GitHub Actions workflow triggered on `push: branches: [main]` (and `workflow_dispatch`) installs pinned dependencies into a Python 3.13 runner and executes the script. Configuration is environment-variable-driven; secrets and dataset ID come from GitHub Secrets/Variables.

**Tech Stack:** Python 3.13, `arize` SDK v8, `openai`, GitHub Actions, OpenInference attribute conventions for dataset rows.

**Reference spec:** `docs/superpowers/specs/2026-04-29-cicd-arize-experiment-design.md`

---

## File structure

| Path | Responsibility |
| ---- | -------------- |
| `.gitignore` | Exclude Python bytecode, venv, dotenv files. |
| `.github/workflows/arize-experiment.yml` | GitHub Actions workflow: trigger, env wiring, install + run. |
| `experiments/classification_test.py` | Task, evaluator, `arize.experiments.run` call, threshold gate. |
| `requirements.txt` | Strictly-pinned versions of `arize` and `openai`. |
| `README.md` | Setup checklist for a new clone: venv, secrets, variable, hand-edit points. |
| `docs/superpowers/specs/...` | Already committed. |
| `docs/superpowers/plans/...` | This file. |

`experiments/` holds one script today; the directory is the obvious place to add more experiment scripts later. No `tests/` directory: this is a starter template where CI itself is the integration test (per spec § Testing). The pure functions in the script are short and reviewable; adding pytest scaffolding for them would be ceremony without meaningful coverage gain.

---

## Task 1: Add `.gitignore`

**Files:**
- Create: `/Users/sean/projects/cicd-experiments/.gitignore`

The repo was initialized during brainstorming (`git init` already ran). The user's global gitignore excludes `docs/`, so the spec was force-added. Standard local exclusions still need a project-level `.gitignore`.

- [ ] **Step 1: Write `.gitignore`**

Create `/Users/sean/projects/cicd-experiments/.gitignore` with:

```
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.env
.env.*
!.env.example
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
```

- [ ] **Step 2: Verify ignore rules apply**

Run: `cd /Users/sean/projects/cicd-experiments && touch test.pyc && git status --short test.pyc; rm test.pyc`

Expected: empty output (the file is ignored, so `git status --short` prints nothing for it).

- [ ] **Step 3: Commit**

```bash
cd /Users/sean/projects/cicd-experiments
git add .gitignore
git commit -m "$(cat <<'EOF'
add project gitignore

Excludes Python build artifacts, virtualenvs, and dotenv files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one new commit; `git log --oneline -1` shows the message above.

---

## Task 2: Create Python 3.13 virtualenv and pin dependencies

**Files:**
- Create: `/Users/sean/projects/cicd-experiments/.venv/` (not committed)
- Create: `/Users/sean/projects/cicd-experiments/requirements.txt`

The user's `python3.13` was confirmed at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13` (`Python 3.13.7`) during planning. We create a venv, install `arize` and `openai`, then capture the resolved versions into `requirements.txt`.

- [ ] **Step 1: Create the virtualenv**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
python3.13 -m venv .venv
.venv/bin/python --version
```

Expected: `Python 3.13.7` (or whatever 3.13.x is installed).

- [ ] **Step 2: Upgrade pip inside the venv**

Run:
```bash
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip --version
```

Expected: pip version printed; no errors.

- [ ] **Step 3: Install arize (>=8) and openai**

Run:
```bash
.venv/bin/pip install --quiet 'arize>=8,<9' 'openai>=1'
```

Expected: install completes silently. If `arize>=8` cannot be resolved (i.e., v8 isn't published yet under that name), the command will error — STOP and surface the error to the user; do not silently downgrade. The fallback question is then: is the v8 SDK on a different package name (e.g., `arize-ax`) or a pre-release that needs `--pre`?

- [ ] **Step 4: Verify install and capture versions**

Run:
```bash
.venv/bin/pip freeze | grep -iE '^(arize|openai)=='
```

Expected: two lines, e.g.:
```
arize==8.0.x
openai==1.x.y
```

- [ ] **Step 5: Write `requirements.txt`**

Write the two lines from Step 4 to `/Users/sean/projects/cicd-experiments/requirements.txt`. Use the *exact* versions returned by `pip freeze` — do not paraphrase or pick versions that weren't installed.

The file should contain only those two pinned lines plus a trailing newline. No comments, no transitive dependencies (we trust pip to resolve).

- [ ] **Step 6: Verify import works in the venv**

Run:
```bash
.venv/bin/python -c "from arize import ArizeClient; from arize.experiments import EvaluationResult; from openai import OpenAI; print('imports ok')"
```

Expected: `imports ok` printed. If any import fails (`ImportError: cannot import name 'EvaluationResult'`, etc.), STOP — the v8 API surface differs from what the spec assumed and the user needs to be consulted before adjusting the script.

- [ ] **Step 7: Commit**

```bash
cd /Users/sean/projects/cicd-experiments
git add requirements.txt
git commit -m "$(cat <<'EOF'
pin arize and openai dependencies

Versions captured from pip freeze in a clean Python 3.13 venv so CI runs
match local development exactly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Write the experiment script

**Files:**
- Create: `/Users/sean/projects/cicd-experiments/experiments/__init__.py` (empty)
- Create: `/Users/sean/projects/cicd-experiments/experiments/classification_test.py`

- [ ] **Step 1: Create the package directory and empty `__init__.py`**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
mkdir -p experiments
touch experiments/__init__.py
```

Expected: directory exists, `__init__.py` is an empty file.

- [ ] **Step 2: Write `experiments/classification_test.py`**

Create the file with this exact content:

```python
"""Classification experiment runner.

Triggered on every push to main. Loads a dataset from Arize, classifies each
example with an LLM, scores via exact match, and exits non-zero if the mean
score is below EXPERIMENT_THRESHOLD.
"""

import os
import sys

from arize import ArizeClient
from arize.experiments import EvaluationResult
from openai import OpenAI

ARIZE_API_KEY = os.environ["ARIZE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATASET = os.environ["ARIZE_DATASET_ID"]
THRESHOLD = float(os.environ.get("EXPERIMENT_THRESHOLD", "0.7"))
NAME_PREFIX = os.environ.get("EXPERIMENT_NAME_PREFIX", "local")
GIT_SHA_SHORT = os.environ.get("GITHUB_SHA", "local")[:7]

# Adjust to match your dataset's label set.
CATEGORIES = ["billing", "technical", "account", "other"]
TASK_MODEL = "gpt-4o-mini"

arize = ArizeClient(api_key=ARIZE_API_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)


def task(dataset_row) -> str:
    user_input = dataset_row.get("attributes.input.value")
    resp = openai.chat.completions.create(
        model=TASK_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify the user message into one of: {', '.join(CATEGORIES)}. "
                    "Reply with only the label."
                ),
            },
            {"role": "user", "content": user_input},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


def exact_match(output, dataset_row) -> EvaluationResult:
    expected = str(dataset_row.get("attributes.output.value", "")).strip().lower()
    actual = str(output).strip().lower()
    score = 1 if actual == expected else 0
    return EvaluationResult(
        score=score,
        label="correct" if score else "incorrect",
        explanation=f"expected={expected!r} actual={actual!r}",
    )


def main() -> None:
    experiment, df = arize.experiments.run(
        name=f"{NAME_PREFIX}-{GIT_SHA_SHORT}",
        dataset=DATASET,
        task=task,
        evaluators=[exact_match],
        concurrency=10,
    )

    score_cols = [c for c in df.columns if "score" in c.lower()]
    if not score_cols:
        print(
            f"ERROR: no score column in result df. Columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(2)
    mean = df[score_cols[0]].mean()

    name = experiment.name if experiment else "n/a"
    print(
        f"experiment={name} score_col={score_cols[0]} "
        f"mean={mean:.3f} threshold={THRESHOLD}"
    )
    sys.exit(0 if mean >= THRESHOLD else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax-check the script**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
.venv/bin/python -m py_compile experiments/classification_test.py
echo "exit=$?"
```

Expected: `exit=0` (file compiles).

- [ ] **Step 4: Verify imports load when env vars are present**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
ARIZE_API_KEY=stub OPENAI_API_KEY=stub ARIZE_DATASET_ID=stub \
  .venv/bin/python -c "import experiments.classification_test as m; print('module loaded:', m.__name__)"
```

Expected: `module loaded: experiments.classification_test`. The module reads env vars at import time, so all three must be set; the script does not call any network APIs at import.

- [ ] **Step 5: Verify the script fails clearly when env vars are missing**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
unset ARIZE_API_KEY
.venv/bin/python experiments/classification_test.py 2>&1 | head -5
echo "exit=$?"
```

Expected: a `KeyError: 'ARIZE_API_KEY'` traceback and a non-zero exit (typically `1`). This is the desired fail-fast behavior.

- [ ] **Step 6: Commit**

```bash
cd /Users/sean/projects/cicd-experiments
git add experiments/__init__.py experiments/classification_test.py
git commit -m "$(cat <<'EOF'
add classification experiment script

Defines a classification task using gpt-4o-mini, an exact-match evaluator,
and a threshold gate that exits non-zero when the mean score falls below
EXPERIMENT_THRESHOLD. Targets Arize Python SDK v8 (client.experiments.run).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Write the GitHub Actions workflow

**Files:**
- Create: `/Users/sean/projects/cicd-experiments/.github/workflows/arize-experiment.yml`

- [ ] **Step 1: Create the workflows directory**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
mkdir -p .github/workflows
```

- [ ] **Step 2: Write `.github/workflows/arize-experiment.yml`**

Create the file with this exact content:

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
      ARIZE_API_KEY: ${{ secrets.ARIZE_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      ARIZE_DATASET_ID: ${{ vars.ARIZE_DATASET_ID }}
      EXPERIMENT_THRESHOLD: "0.7"
      EXPERIMENT_NAME_PREFIX: "main-push"
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Arize experiment
        run: python experiments/classification_test.py
```

- [ ] **Step 3: Verify the YAML parses**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
.venv/bin/python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/arize-experiment.yml')); print('yaml ok')"
```

Expected: `yaml ok`. If `yaml` is not installed, run `.venv/bin/pip install --quiet pyyaml` first (it is not added to `requirements.txt` — it is only used here for a one-shot validation, not at runtime).

- [ ] **Step 4: Sanity-check the trigger matches the spec**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
grep -E "^on:|branches:|workflow_dispatch" .github/workflows/arize-experiment.yml
```

Expected output includes `on:`, `    branches: [main]`, and `  workflow_dispatch:` — confirming push-to-main and manual-dispatch triggers are both present.

- [ ] **Step 5: Commit**

```bash
cd /Users/sean/projects/cicd-experiments
git add .github/workflows/arize-experiment.yml
git commit -m "$(cat <<'EOF'
add github actions workflow for arize experiment

Triggered on push to main and via workflow_dispatch. Installs pinned
dependencies, runs the experiment script, and surfaces pass/fail via the
script's exit code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Write the README

**Files:**
- Create: `/Users/sean/projects/cicd-experiments/README.md`

- [ ] **Step 1: Write `README.md`**

Create the file with this exact content:

```markdown
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
```

- [ ] **Step 2: Verify the file exists and looks right**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
wc -l README.md && head -5 README.md
```

Expected: line count > 30; first line is `# cicd-experiments`.

- [ ] **Step 3: Commit**

```bash
cd /Users/sean/projects/cicd-experiments
git add README.md
git commit -m "$(cat <<'EOF'
add README with setup instructions

Documents the pipeline behavior, GitHub Secrets/Variables required,
hand-edit points before the first run, and a local smoke-test recipe.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: End-to-end repo sanity check

**Files:** read-only verification — no new files.

- [ ] **Step 1: Confirm the working tree is clean**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
git status --short
```

Expected: empty output (everything is committed; nothing untracked except `.venv/` which is ignored).

- [ ] **Step 2: List committed files**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
git ls-files
```

Expected output (order may vary):
```
.github/workflows/arize-experiment.yml
.gitignore
README.md
docs/superpowers/plans/2026-04-29-cicd-arize-experiment.md
docs/superpowers/specs/2026-04-29-cicd-arize-experiment-design.md
experiments/__init__.py
experiments/classification_test.py
requirements.txt
```

- [ ] **Step 3: Show the full commit history**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
git log --oneline
```

Expected: 6 commits — design doc, plan doc, gitignore, requirements, experiment script, workflow, README. (The plan-doc commit will exist if it was committed during planning; if not, it's added now — see Step 4.)

- [ ] **Step 4: Commit the implementation plan if not already tracked**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
git status --short docs/superpowers/plans/
```

If the plan file is untracked, commit it (it lives under the globally-ignored `docs/`, so use `-f`):

```bash
git add -f docs/superpowers/plans/2026-04-29-cicd-arize-experiment.md
git commit -m "$(cat <<'EOF'
add implementation plan for CI/CD arize experiment pipeline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If the file is already committed, skip this step.

- [ ] **Step 5: Final byte-level verification of the experiment script**

Run:
```bash
cd /Users/sean/projects/cicd-experiments
ARIZE_API_KEY=stub OPENAI_API_KEY=stub ARIZE_DATASET_ID=stub \
  .venv/bin/python -c "import experiments.classification_test as m; \
print('threshold:', m.THRESHOLD); \
print('categories:', m.CATEGORIES); \
print('task callable:', callable(m.task)); \
print('exact_match callable:', callable(m.exact_match))"
```

Expected:
```
threshold: 0.7
categories: ['billing', 'technical', 'account', 'other']
task callable: True
exact_match callable: True
```

---

## Task 7: Hand-off notes (no code changes)

The repo is now ready for the user to publish and configure on GitHub. These steps require the user's GitHub account and cannot be automated from this session.

- [ ] **Step 1: Surface the hand-off checklist to the user**

After Task 6 succeeds, print a summary in the assistant turn covering:

1. Create a GitHub repo (or push to an existing one). Suggested commands when ready:
   ```bash
   git remote add origin git@github.com:<user>/cicd-experiments.git
   git branch -M main
   git push -u origin main
   ```
2. In GitHub → Settings → Secrets and variables → Actions:
   - Secrets: `ARIZE_API_KEY`, `OPENAI_API_KEY`
   - Variables: `ARIZE_DATASET_ID`
3. Edit `experiments/classification_test.py` so `CATEGORIES` and the two `dataset_row.get(...)` keys match the user's actual dataset.
4. Push the edited commit to `main` (or trigger via *workflow_dispatch*) to validate.
5. If the first run fails on `score_cols` being empty, adjust the column name; if it fails on `KeyError`, the env wiring is off; if it fails on missing API surface, the v8 SDK version may differ — check the actual installed `arize` version vs. what the spec assumed.

---

## Self-review notes

**Spec coverage:**
- § Repository layout → Tasks 1, 3, 4, 5 (covers all listed files except `docs/`, which is preserved from brainstorming).
- § Experiment script → Task 3 (full body matches spec).
- § GitHub Actions workflow → Task 4 (env, concurrency, triggers all match spec § 2).
- § Dependencies → Task 2 (strict pin via `pip freeze`).
- § README → Task 5 (three sections per spec § 4).
- § `.gitignore` → Task 1.
- § Configuration surface → all six env vars are referenced (three in workflow `env:`, three with defaults in the script).
- § Data flow → exercised via Task 6 sanity check + Task 7 hand-off.
- § Error handling → Task 3 Step 5 verifies the fail-fast `KeyError` path; the score-column scan + exit code 2 path is in the script body.
- § Testing → Task 6 covers the local smoke pieces (compile, import, attribute presence). The CI run is the integration test, per spec.
- § Implementation order → Tasks 1–6 map 1:1 to the spec's seven implementation steps (with step 7 as Task 7 hand-off).

**Placeholder scan:** None. All code blocks contain runnable content; all expected outputs are concrete.

**Type / signature consistency:**
- `task(dataset_row) -> str` — same shape in script and verification step.
- `exact_match(output, dataset_row) -> EvaluationResult` — same shape in script and verification step.
- `arize.experiments.run(name, dataset, task, evaluators, concurrency)` — keyword args match SDK v8 docs the user provided.
- Env var names (`ARIZE_API_KEY`, `OPENAI_API_KEY`, `ARIZE_DATASET_ID`, `EXPERIMENT_THRESHOLD`, `EXPERIMENT_NAME_PREFIX`) are identical across spec, plan, script, and workflow.
