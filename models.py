from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    MCQ = "mcq"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SHORT_ANSWER = "short_answer"


class PerturbationFamily(str, Enum):
    PARAPHRASE = "paraphrase"
    TYPO_NOISE = "typo_noise"
    FORMATTING = "formatting"
    FEWSHOT_ORDER = "fewshot_order"
    OPTION_ORDER = "option_order"
    INSTRUCTION_POSITION = "instruction_position"
    DISTRACTOR = "distractor"


class FewShotExample(BaseModel):
    input_text: str
    output_text: str


class Task(BaseModel):
    id: str
    task_type: TaskType

    instruction: str
    input_text: str

    options: list[str] | None = None
    answer: str

    few_shot_examples: list[FewShotExample] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Perturbation(BaseModel):
    id: str
    base_task_id: str

    family: PerturbationFamily

    original_prompt: str
    perturbed_prompt: str

    valid: bool = False
    validation_error: str | None = None

    seed: int | None = None


class EvaluationResult(BaseModel):
    perturbation_id: str
    base_task_id: str
    task_type: str
    family: str

    raw_baseline_output: str | None = None
    raw_perturbed_output: str | None = None

    baseline_answer: str | None = None
    perturbed_answer: str | None = None

    baseline_correct: bool = False
    perturbed_correct: bool = False

    consistent: bool = False

    # A real robustness failure means:
    # baseline was correct but perturbation caused an incorrect answer.
    robustness_failure: bool = False

    error: str | None = None
