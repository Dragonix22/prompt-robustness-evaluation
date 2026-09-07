from pathlib import Path

import pandas as pd

from .clients import call_model
from .models import Task, Perturbation
from .normalization import canonicalize_output


RESULT_COLUMNS = [
    "perturbation_id",
    "base_task_id",
    "task_type",
    "family",
    "raw_baseline_output",
    "raw_perturbed_output",
    "baseline_answer",
    "perturbed_answer",
    "baseline_correct",
    "perturbed_correct",
    "consistent",
    "robustness_failure",
    "error",
]


def evaluate_task(
    task: Task,
) -> tuple[str, str]:

    prompt = (
        f"{task.instruction}\n\n"
        f"{task.input_text}"
    )

    output = call_model(prompt)

    return output, canonicalize_output(
        output,
        task.task_type.value,
        task.options,
    )


def evaluate_perturbation(
    task: Task,
    perturbation: Perturbation,
    baseline_raw: str,
    baseline_answer: str,
) -> dict:

    perturbed_raw = call_model(
        perturbation.perturbed_prompt
    )

    perturbed_answer = canonicalize_output(
        perturbed_raw,
        task.task_type.value,
        task.options,
    )

    baseline_correct = (
        baseline_answer
        == canonicalize_output(
            task.answer,
            task.task_type.value,
            task.options,
        )
    )

    perturbed_correct = (
        perturbed_answer
        == canonicalize_output(
            task.answer,
            task.task_type.value,
            task.options,
        )
    )

    consistent = (
        baseline_answer
        == perturbed_answer
    )

    robustness_failure = (
        baseline_correct
        and not perturbed_correct
    )

    return {
        "perturbation_id": perturbation.id,
        "base_task_id": task.id,
        "task_type": task.task_type.value,
        "family": perturbation.family.value,

        "raw_baseline_output": baseline_raw,
        "raw_perturbed_output": perturbed_raw,

        "baseline_answer": baseline_answer,
        "perturbed_answer": perturbed_answer,

        "baseline_correct": baseline_correct,
        "perturbed_correct": perturbed_correct,

        "consistent": consistent,
        "robustness_failure": robustness_failure,

        "error": None,
    }


def run_evaluation(
    tasks: dict[str, Task],
    perturbations: list[Perturbation],
    output_path: str = "results/evaluation_results.csv",
    baseline_path: str = "results/baseline_results.csv",
):

    output_file = Path(output_path)
    baseline_file = Path(baseline_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    if output_file.exists():

        results_df = pd.read_csv(
            output_file
        )

        completed = set(
            results_df[
                "perturbation_id"
            ].dropna()
        )

        print(
            f"Loaded checkpoint: "
            f"{len(completed)} completed."
        )

    else:

        results_df = pd.DataFrame(
            columns=RESULT_COLUMNS
        )

        completed = set()

    # ---------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------

    if baseline_file.exists():

        baseline_df = pd.read_csv(
            baseline_file
        )

        baseline_lookup = {
            row["base_task_id"]: row
            for _, row in baseline_df.iterrows()
        }

    else:

        baseline_rows = []

        for task in tasks.values():

            raw, canonical = evaluate_task(
                task
            )

            gold = canonicalize_output(
                task.answer,
                task.task_type.value,
                task.options,
            )

            baseline_rows.append({
                "base_task_id": task.id,
                "task_type": task.task_type.value,
                "raw_output": raw,
                "canonical_answer": canonical,
                "correct": canonical == gold,
            })

        baseline_df = pd.DataFrame(
            baseline_rows
        )

        baseline_df.to_csv(
            baseline_file,
            index=False,
        )

        baseline_lookup = {
            row["base_task_id"]: row
            for _, row in baseline_df.iterrows()
        }

    # ---------------------------------------------------------
    # Perturbation evaluation
    # ---------------------------------------------------------

    for i, perturbation in enumerate(
        perturbations,
        start=1,
    ):

        if not perturbation.valid:
            continue

        if perturbation.id in completed:
            continue

        task = tasks[
            perturbation.base_task_id
        ]

        baseline = baseline_lookup[
            task.id
        ]

        print(
            f"[{i}] "
            f"{task.id} "
            f"{perturbation.family.value}"
        )

        result = evaluate_perturbation(
            task=task,
            perturbation=perturbation,
            baseline_raw=baseline[
                "raw_output"
            ],
            baseline_answer=baseline[
                "canonical_answer"
            ],
        )

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame([result]),
            ],
            ignore_index=True,
        )

        # CRITICAL:
        # save after every successful request.
        results_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"  baseline: "
            f"{result['baseline_answer']}"
        )

        print(
            f"  perturbed: "
            f"{result['perturbed_answer']}"
        )

        print(
            f"  consistent: "
            f"{result['consistent']}"
        )

    return results_df
