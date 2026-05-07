"""Grounding-with-search evaluation runner.

Triggered on every push to main. Loads a dataset from Arize, runs the task
under test, then uses an LLM judge to score outputs. Exits non-zero if the
mean score is below EXPERIMENT_THRESHOLD.
"""

import os
import sys

from arize import ArizeClient
from arize.experiments import EvaluationResult
from openai import OpenAI
from tavily import TavilyClient

ARIZE_API_KEY = os.environ["ARIZE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
ARIZE_DATASET_ID = os.environ["ARIZE_DATASET_ID"]
THRESHOLD = float(os.environ.get("EXPERIMENT_THRESHOLD", "0.7"))
GIT_SHA_SHORT = os.environ.get("GITHUB_SHA", "local")[:7]

TASK_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o"
SEARCH_MAX_RESULTS = 5

EVAL_PROMPT_TEMPLATE = """\
### Grounding Evaluation ###
You are checking whether a model response is supported by web evidence.

Question: {input}

Response: {model_response}

Web evidence (top search results):
{evidence}

Is the response factually supported by the web evidence above?
- Answer "grounded" if the response's main claims are corroborated by the evidence.
- Answer "ungrounded" if the response contradicts the evidence, or makes claims the evidence does not support.

Answer only: grounded OR ungrounded
"""

client = ArizeClient(api_key=ARIZE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def _retrieve_evidence(query: str) -> str:
    results = tavily_client.search(
        query=query,
        max_results=SEARCH_MAX_RESULTS,
        search_depth="basic",
    ).get("results", [])
    if not results:
        return "(no search results)"
    chunks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        content = (r.get("content") or "").strip()
        chunks.append(f"[{i}] {title} — {url}\n{content}")
    return "\n\n".join(chunks)


def task(dataset_row) -> str:
    # TODO: replace `query` with the actual input column in your dataset
    query = dataset_row.get("query")
    resp = openai_client.chat.completions.create(
        model=TASK_MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content.strip()


def grounded_with_search(output, dataset_row) -> EvaluationResult:
    query = dataset_row.get("English")
    evidence = _retrieve_evidence(query)
    message_content = EVAL_PROMPT_TEMPLATE.format(
        input=query,
        model_response=output,
        evidence=evidence,
    )
    resp = openai_client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": message_content}],
    )
    judgment = resp.choices[0].message.content.lower().strip()
    grounded = judgment == "grounded"
    return EvaluationResult(
        score=int(grounded),
        label="good" if grounded else "bad",
        explanation=f"judge_verdict={judgment!r}",
    )


def _write_github_output(**fields: str) -> None:
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if not gh_output:
        return
    with open(gh_output, "a") as f:
        for key, value in fields.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    experiment, df = client.experiments.run(
        name=f"grounding-with-search-{GIT_SHA_SHORT}",
        dataset=ARIZE_DATASET_ID,
        task=task,
        evaluators=[grounded_with_search],
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

    if mean < THRESHOLD:
        _write_github_output(
            regressed="true",
            mean=f"{mean:.3f}",
            threshold=f"{THRESHOLD}",
            experiment_name=name,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
