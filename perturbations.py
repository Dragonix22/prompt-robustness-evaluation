import hashlib
import random
import re

from .models import Task


FAMILIES = [
    "paraphrase",
    "typo_noise",
    "formatting",
    "fewshot_order",
    "option_order",
    "instruction_position",
    "distractor",
]


def stable_seed(*parts) -> int:
    text = "|".join(map(str, parts))

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def apply_paraphrase(task: Task) -> str:
    # Deterministic templates for the current benchmark.
    prompt = task.instruction + "\n\n" + task.input_text

    replacements = {
        "What is": "Which of the following asks:",
        "Answer the following question.": "Respond to the question below.",
        "Perform the requested extraction.": "Complete the requested extraction.",
        "Perform the requested classification.": "Complete the requested classification.",
    }

    for old, new in replacements.items():
        prompt = prompt.replace(old, new)

    return prompt


def apply_typo_noise(task: Task) -> str:
    prompt = task.instruction + "\n\n" + task.input_text

    replacements = {
        "question": "quesiton",
        "following": "folowing",
        "answer": "anser",
        "classification": "classifcation",
        "extraction": "extracion",
    }

    for old, new in replacements.items():
        prompt = prompt.replace(old, new, 1)

    return prompt


def apply_formatting(task: Task) -> str:
    prompt = (
        task.instruction
        + "\n\n"
        + task.input_text
    )

    return (
        "============================== TASK ==============================\n"
        f"{prompt.upper()}\n"
        "============================== END TASK ==========================="
    )


def apply_option_order(task: Task) -> str | None:
    if task.task_type.value != "mcq":
        return None

    assert task.options is not None

    options = list(reversed(task.options))

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    option_text = "\n".join(
        f"{labels[i]}. {option}"
        for i, option in enumerate(options)
    )

    return (
        f"{task.instruction}\n\n"
        f"{task.input_text}\n\n"
        f"Options:\n{option_text}"
    )


def apply_instruction_position(task: Task) -> str:
    return (
        f"{task.input_text}\n\n"
        f"INSTRUCTION: {task.instruction}"
    )


def apply_distractor(task: Task) -> str:
    prefix = {
        "mcq": (
            "For context, this question is part of a "
            "general knowledge exercise."
        ),
        "classification": (
            "For context, this classification task is "
            "part of a general information exercise."
        ),
        "extraction": (
            "For context, this extraction task is "
            "part of a general information exercise."
        ),
        "short_answer": (
            "The following background statement is included "
            "only as a distractor."
        ),
    }[task.task_type.value]

    return (
        f"{prefix}\n\n"
        f"{task.instruction}\n\n"
        f"{task.input_text}"
    )


def apply_fewshot_order(task: Task) -> str | None:
    if not task.few_shot_examples:
        return None

    examples = list(task.few_shot_examples)
    rng = random.Random(
        stable_seed(task.id, "fewshot_order")
    )

    rng.shuffle(examples)

    sections = []

    for example in examples:
        sections.append(
            f"Example:\n"
            f"{example.input_text}\n"
            f"Answer: {example.output_text}"
        )

    sections.append(
        f"Task:\n{task.input_text}"
    )

    return (
        task.instruction
        + "\n\n"
        + "\n\n".join(sections)
    )


def generate_perturbation(
    task: Task,
    family: str,
) -> str | None:

    if family == "paraphrase":
        return apply_paraphrase(task)

    if family == "typo_noise":
        return apply_typo_noise(task)

    if family == "formatting":
        return apply_formatting(task)

    if family == "option_order":
        return apply_option_order(task)

    if family == "instruction_position":
        return apply_instruction_position(task)

    if family == "distractor":
        return apply_distractor(task)

    if family == "fewshot_order":
        return apply_fewshot_order(task)

    raise ValueError(
        f"Unknown perturbation family: {family}"
    )
