import re

from .models import Task


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def validate_perturbation(
    task: Task,
    perturbed_prompt: str | None,
) -> tuple[bool, str | None]:

    if not perturbed_prompt:
        return False, "Empty perturbation."

    if normalize(perturbed_prompt) == normalize(
        task.instruction + "\n\n" + task.input_text
    ):
        return False, "Perturbation did not change the prompt."

    # MCQ invariant:
    # every option must survive.
    if task.task_type.value == "mcq":

        if not task.options:
            return False, "MCQ has no options."

        normalized_prompt = normalize(
            perturbed_prompt
        )

        for option in task.options:

            if normalize(option) not in normalized_prompt:
                return False, (
                    f"MCQ option disappeared: {option}"
                )

    return True, None
