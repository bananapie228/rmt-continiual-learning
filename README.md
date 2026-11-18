# Continual LoRA Fine-Tuning: Replay vs Regularisation

**Research Methods and Tools — Assignment 4 (SE-2418)**
Mirzanov Temirlan · Turemuratov Galymzhan · Abdikul Bakdaulet

**Research question.** How do replay-based and regularization-based continual fine-tuning approaches compare with standard sequential LoRA fine-tuning in mitigating catastrophic forgetting in open-source language models under limited computational resources?

## Setup

- **Model:** `Qwen/Qwen2.5-0.5B` with LoRA (rank 8, alpha 16, dropout 0.05, q/k/v/o projections)
- **Task sequence:** AG News (4 classes) → SST-2 (2 classes) → Emotion (6 classes); 1,000 train / 200 validation / 800 test per task, seed 1234
- **Conditions:** sequential LoRA, experience replay (10% class-stratified buffer), online EWC (diagonal Fisher on LoRA weights, λ selected on validation data), joint multi-task LoRA
- **Runs:** 19 — 5 EWC λ-calibration runs (seed 1000), 4 conditions × seeds 0/1/2, replay 5% and 20% sensitivity (seed 0)
- **Hardware:** Google Colab, Tesla T4; PyTorch 2.14.1+cu130, Transformers 5.22.1, PEFT 0.21.3

## Results (mean ± SD over 3 seeds)

| Condition | Final ACC (%) | Forgetting FGT (pp) ↓ | Learning acc. LA (%) | Training time (min) |
|---|---|---|---|---|
| Seq-LoRA | 78.89 ± 0.46 | 9.04 ± 0.36 | 84.92 ± 0.61 | 1.57 |
| Experience Replay (10%) | 82.39 ± 0.65 | 4.63 ± 1.35 | 85.33 ± 0.57 | 1.85 |
| Online EWC (λ = 0.01) | 79.33 ± 0.74 | 8.04 ± 0.34 | 84.69 ± 0.60 | 3.55 |
| Joint (upper reference) | 85.21 ± 0.15 | — | — | 1.75 |

Experience replay halves forgetting and gives the best accuracy–cost trade-off; online EWC at the validation-selected λ reduces forgetting only slightly and more than doubles training time because of Fisher estimation. Full statistics (paired bootstrap, McNemar with Holm correction), resource measurements and error analysis are in `results/assignment4_measured/`.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/RMT_A4_real_experiment.ipynb` | Complete experiment: data preparation, training, evaluation, statistics |
| `data/assignment4/` | The 6,000-record dataset used by all runs, its manifest with SHA-256 hashes, and a data card |
| `results/assignment4_measured/runs/` | One JSON per run: accuracy/F1 matrices, per-item predictions for every stage, resource logs |
| `results/assignment4_measured/tables/` | Summary tables, statistical tests, EWC calibration, replay sensitivity, error analysis |
| `results/assignment4_measured/report_figures/` | Figures used in the report |
| `results/assignment4_measured/env.json`, `config.json`, `manifest.json` | Environment, configuration and run manifest |

## Reproduce

1. Open `notebooks/RMT_A4_real_experiment.ipynb` in Google Colab.
2. Select `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
3. The notebook downloads the datasets and the model from the Hugging Face Hub, runs all 19 runs (about 80 minutes on a T4) and writes the same file layout to Google Drive.

## Data sources

AG News (`fancyzhx/ag_news`), SST-2 (`nyu-mll/glue`, `sst2`) and Emotion (`dair-ai/emotion`) from the Hugging Face Hub, used for research and educational purposes under their original terms.
