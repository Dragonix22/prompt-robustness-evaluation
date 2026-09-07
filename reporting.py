from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import (
    calculate_metrics,
    family_summary,
    task_type_summary,
    get_answer_flips,
    get_robustness_failures,
)


def generate_report(
    results: pd.DataFrame,
    output_dir: str = "results",
):

    output = Path(output_dir)
    figures = output / "figures"

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
        ]
    ).to_csv(
        output / "overall_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------

    family = family_summary(
        results
    )

    task_types = task_type_summary(
        results
    )

    family.to_csv(
        output / "family_summary.csv",
        index=False,
    )

    task_types.to_csv(
        output / "task_type_summary.csv",
        index=False,
    )

    flips = get_answer_flips(
        results
    )

    failures = get_robustness_failures(
        results
    )

    flips.to_csv(
        output / "answer_flips.csv",
        index=False,
    )

    failures.to_csv(
        output / "robustness_failures.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Figure 1: consistency by family
    # ---------------------------------------------------------

    plt.figure()

    plt.bar(
        family["family"],
        family["consistency_rate"] * 100,
    )

    plt.ylabel(
        "Consistency (%)"
    )

    plt.xlabel(
        "Perturbation family"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        figures
        / "consistency_by_family.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 2: baseline vs perturbed
    # ---------------------------------------------------------

    plt.figure()

    x = range(len(family))

    plt.plot(
        x,
        family["baseline_accuracy"] * 100,
        marker="o",
        label="Baseline",
    )

    plt.plot(
        x,
        family["perturbed_accuracy"] * 100,
        marker="o",
        label="Perturbed",
    )

    plt.xticks(
        list(x),
        family["family"],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figures
        / "baseline_vs_perturbed_accuracy.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 3: task type
    # ---------------------------------------------------------

    plt.figure()

    plt.bar(
        task_types["task_type"],
        task_types["consistency_rate"] * 100,
    )

    plt.ylabel(
        "Consistency (%)"
    )

    plt.xlabel(
        "Task type"
    )

    plt.tight_layout()

    plt.savefig(
        figures
        / "consistency_by_task_type.png",
        dpi=200,
    )

    plt.close()
