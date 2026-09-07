import pandas as pd


def calculate_metrics(
    results: pd.DataFrame,
) -> dict:

    if results.empty:
        return {}

    baseline_accuracy = (
        results["baseline_correct"]
        .mean()
    )

    perturbed_accuracy = (
        results["perturbed_correct"]
        .mean()
    )

    consistency = (
        results["consistent"]
        .mean()
    )

    robustness_failures = (
        results["robustness_failure"]
        .sum()
    )

    answer_flips = (
        (
            results["baseline_answer"]
            != results["perturbed_answer"]
        )
        .sum()
    )

    baseline_correct_results = results[
        results["baseline_correct"]
    ]

    if len(baseline_correct_results):

        accuracy_retention = (
            baseline_correct_results[
                "perturbed_correct"
            ].mean()
        )

    else:
        accuracy_retention = float("nan")

    return {
        "evaluated": len(results),

        "baseline_accuracy":
            baseline_accuracy,

        "perturbed_accuracy":
            perturbed_accuracy,

        "accuracy_delta":
            perturbed_accuracy
            - baseline_accuracy,

        "consistency_rate":
            consistency,

        "accuracy_retention":
            accuracy_retention,

        "answer_flips":
            int(answer_flips),

        "robustness_failures":
            int(robustness_failures),
    }


def family_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby("family")
        .agg(
            evaluated=(
                "perturbation_id",
                "count",
            ),
            consistency_rate=(
                "consistent",
                "mean",
            ),
            baseline_accuracy=(
                "baseline_correct",
                "mean",
            ),
            perturbed_accuracy=(
                "perturbed_correct",
                "mean",
            ),
            robustness_failures=(
                "robustness_failure",
                "sum",
            ),
        )
        .reset_index()
    )


def task_type_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby("task_type")
        .agg(
            evaluated=(
                "perturbation_id",
                "count",
            ),
            consistency_rate=(
                "consistent",
                "mean",
            ),
            baseline_accuracy=(
                "baseline_correct",
                "mean",
            ),
            perturbed_accuracy=(
                "perturbed_correct",
                "mean",
            ),
            robustness_failures=(
                "robustness_failure",
                "sum",
            ),
        )
        .reset_index()
    )


def get_answer_flips(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return results[
        results["baseline_answer"]
        != results["perturbed_answer"]
    ].copy()


def get_robustness_failures(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return results[
        results["robustness_failure"]
        == True
    ].copy()
