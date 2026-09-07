# Prompt Robustness & Consistency Evaluation Suite

## Overview

This project is an automated evaluation pipeline for measuring how sensitive
LLMs are to small changes in prompt wording and structure.

The system takes a set of base tasks, generates controlled prompt
perturbations, evaluates the original and perturbed prompts using an LLM,
and calculates robustness and consistency metrics.

The goal of this project is to provide a reproducible framework that can
be extended with additional tasks, perturbation types, and models.

---

## Project Goals

- Measure LLM consistency under prompt perturbations
- Identify which types of prompt changes cause failures
- Distinguish genuine answer changes from changes in output formatting
- Automatically validate generated perturbations
- Produce structured results and visualizations
- Make experiments reproducible and easy to extend

---

## Pipeline

    Base Tasks
        ↓
    Generate Perturbations
        ↓
    Validate Perturbations
        ↓
    Baseline Evaluation
        ↓
    Perturbed Evaluation
        ↓
    Answer Normalization
        ↓
    Calculate Metrics
        ↓
    CSV Results + Visualizations

---

## Perturbation Types

The system currently supports seven perturbation families:

| Perturbation | Description |
|---|---|
| Paraphrase | Rewords the prompt while preserving its meaning |
| Typo Noise | Introduces small spelling/typing errors |
| Formatting | Changes formatting without changing the task |
| Few-Shot Order | Changes the ordering of examples |
| Option Order | Reorders multiple-choice options |
| Instruction Position | Moves instructions to different positions |
| Distractor | Adds irrelevant information to the prompt |

---

## Task Types

The benchmark currently contains four task types:

- Multiple Choice Questions (MCQ)
- Classification
- Extraction
- Short Answer

The initial validation benchmark contains **9 base tasks**.

The planned expanded benchmark contains approximately **30 base tasks**.

---

## Metrics

The pipeline calculates several metrics.

### Baseline Accuracy

Accuracy on the original, unmodified tasks.

### Perturbed Accuracy

Accuracy after applying prompt perturbations.

### Exact Consistency

Percentage of perturbations where the model's normalized output
matches the baseline output.

### Accuracy Retention

How much baseline accuracy is retained after perturbation.

### Answer Flips

Cases where the baseline and perturbed outputs differ.

### Robustness Failures

Cases where a perturbation causes the model to change from a correct
answer to an incorrect answer.

---

## Initial Results

The initial pipeline validation was performed using Gemini 3.1 Flash-Lite.

### Dataset

- 9 base tasks
- 63 perturbations generated
- 45 perturbations evaluated in the initial run
- 13 perturbations marked not applicable
- 5 perturbations identified as invalid during validation

### Results

| Metric | Result |
|---|---:|
| Baseline Accuracy | 100% |
| Perturbed Accuracy | 100% |
| Accuracy Retention | 100% |
| Exact Consistency | 91.11% |
| Actual Robustness Failures | 0 |
| Exact Answer Flips | 4 |

These results are from a small **pipeline validation run** and should not
be interpreted as a general evaluation of the model.

---

## Important Finding: Exact vs. Semantic Consistency

One important issue discovered during evaluation was that an output can
change textually without changing the underlying answer.

For example:

> **Baseline:** `negative`

> **Perturbed:** `The sentiment of the statement is negative.`

A simple string comparison would classify this as an answer flip, even
though the model gave the same answer.

This motivated the addition of answer normalization and canonicalization.

For multiple-choice questions, answers are also mapped back to their
underlying option text so that representations such as `B` and `2` do not
incorrectly appear to be different answers.

---

## Validation

Generated perturbations are validated before evaluation.

The validation system checks for issues such as:

- Missing or malformed prompts
- Perturbations that remove the original task
- Invalid multiple-choice formatting
- Perturbations that are not applicable to a task type
- Other malformed generated records

Invalid perturbations are excluded from model evaluation.

---

## Project Structure

    prompt-robustness-evaluation/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── src/
    │   ├── perturbations.py
    │   ├── validation.py
    │   ├── evaluation.py
    │   ├── normalization.py
    │   └── metrics.py
    │
    ├── data/
    │   ├── base_tasks.json
    │   └── perturbations.json
    │
    ├── results/
    │   ├── evaluation_results.csv
    │   ├── overall_metrics.csv
    │   ├── family_summary.csv
    │   ├── task_type_summary.csv
    │   ├── answer_flips.csv
    │   └── robustness_failures.csv
    │
    ├── plots/
    │   └── ...
    │
    └── notebooks/
        └── evaluation.ipynb

Adjust this section to match the actual files in the repository.

---

## Installation

    git clone <repository-url>
    cd prompt-robustness-evaluation
    pip install -r requirements.txt

---

## API Configuration

The evaluation pipeline requires an API key for the selected model provider.

Set the API key as an environment variable:

    export GEMINI_API_KEY="your-api-key"

**Never commit API keys or other secrets to the repository.**

If using Google Colab, store the API key using Colab Secrets rather than
placing it directly in the notebook.

---

## Running the Evaluation

If the project is run as a Python script:

    python src/evaluation.py

If the project is primarily run through Google Colab, provide the notebook
name and explain the required steps here instead.

---

## Output Files

| File | Description |
|---|---|
| `evaluation_results.csv` | Individual baseline and perturbation results |
| `overall_metrics.csv` | Overall robustness metrics |
| `family_summary.csv` | Results grouped by perturbation family |
| `task_type_summary.csv` | Results grouped by task type |
| `answer_flips.csv` | Cases where baseline and perturbed outputs differ |
| `robustness_failures.csv` | Cases where perturbations cause actual failures |

---

## Known Issues

- The initial benchmark is relatively small.
- Some perturbation families are not applicable to every task type.
- Semantic equivalence detection can be more difficult than exact string
  matching.
- More models and larger task sets are needed for broader comparisons.

---

## Next Steps

1. Expand the benchmark from 9 to approximately 30 base tasks.
2. Run the complete perturbation suite on the expanded benchmark.
3. Evaluate additional models.
4. Run multiple trials to improve reliability.
5. Compare robustness across perturbation families.
6. Compare robustness across task types and models.
7. Improve semantic answer matching where necessary.

---

## Handoff

This repository is intended to provide the next developer with everything
needed to understand, run, debug, and extend the evaluation pipeline.

The main areas to extend are:

- `data/` — add benchmark tasks
- Perturbation generation — add or modify perturbation families
- Validation — add new validation rules
- Normalization — improve semantic answer matching
- Evaluation — add models or evaluation settings
- Metrics — add additional robustness measurements

---
