# M4 Robustness Evaluation Guide

## Overview

The M4 robustness module (`backend/app/pipeline/robustness.py`) provides controlled, reproducible evaluation of pipeline baseline models against benign input perturbations, including spelling noise, Unicode variations, spacing irregularities, code-switching, and semantic paraphrasing.

## Supported Transformations

| Transformation | Mechanism | Use Case |
|---|---|---|
| `unicode_variation` | NFKD character decomposition, benign zero-width space injection | Testing resilience against character encoding mutations and hidden spaces |
| `spelling_noise` | Bounded adjacent character swaps and letter duplication | Simulating casual typos and keyboard input variations |
| `spacing_noise` | Irregular whitespace, double-spacing, and trailing whitespace | Simulating copy-paste formatting anomalies and token boundary shifts |
| `code_switching` | Intersperse conversational markers in Swahili, Hausa, or Bengali | Evaluating model stability on multi-dialect and code-switched text |
| `paraphrase_synonyms` | Conservative lexical synonym replacement preserving core semantics | Testing semantic generalization against lexical substitutions |

## Evaluation Metrics

For each transformation applied across the evaluation dataset:
- **Clean Score**: Baseline model evaluation metric on unaltered input data.
- **Perturbed Score**: Model evaluation metric on transformed text inputs.
- **Score Delta ($\Delta$)**: $\Delta = \text{Clean Score} - \text{Perturbed Score}$. Positive delta indicates performance degradation under perturbation.
- **Retention Ratio**: $\text{Retention} = \frac{\text{Perturbed Score}}{\text{Clean Score}}$. Values near $1.0$ indicate high robustness.
- **Subgroup Breakdown**: Disaggregates performance by detected language (via M2 language identification) to detect disparate impact across languages.

## CLI Usage

Run robustness evaluation against any trained pipeline model directory:

```bash
# Evaluate all supported transformations
python backend/scripts/pipeline_robustness_eval.py \
  --model-dir backend/outputs/m1_binary

# Evaluate specific transformations and output a JSON report
python backend/scripts/pipeline_robustness_eval.py \
  --model-dir backend/outputs/m1_binary \
  --transformations spelling_noise unicode_variation \
  --output-report backend/outputs/robustness_report.json
```
