from robusteval.tasks import load_tasks
from robusteval.perturbations import generate_all_perturbations
from robusteval.validation import validate_perturbation
from robusteval.evaluation import run_evaluation
from robusteval.reporting import generate_report


def main():

    tasks = load_tasks(
        "data/base_tasks.jsonl"
    )

    perturbations = (
        generate_all_perturbations(tasks)
    )

    # Validate everything BEFORE spending API calls.
    for perturbation in perturbations:

        task = tasks[
            perturbation.base_task_id
        ]

        valid, error = validate_perturbation(
            task,
            perturbation.perturbed_prompt,
        )

        perturbation.valid = valid
        perturbation.validation_error = error

    save_perturbations(
        perturbations,
        "data/perturbations.jsonl",
    )

    # Only valid perturbations reach the API.
    results = run_evaluation(
        tasks=tasks,
        perturbations=perturbations,
    )

    generate_report(
        results,
        output_dir="results",
    )


if __name__ == "__main__":
    main()
