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
