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

```text
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
