# Assignment 4 — measured experiment

Measured results of the continual LoRA fine-tuning experiment (Seq-LoRA vs Experience Replay vs Online EWC, plus Joint training).

- Model: `Qwen/Qwen2.5-0.5B` + LoRA (r=8, alpha=16, dropout 0.05, q/k/v/o)
- Tasks: AG News → SST-2 → Emotion, 1,000 / 200 / 800 records per task (seed 1234)
- Runs: 19 (5 EWC λ calibration, 4 methods × seeds 0/1/2, ER 5% and 20% sensitivity); selected λ = 0.01
- Hardware: Google Colab, Tesla T4; PyTorch 2.14.1+cu130, Transformers 5.22.1, PEFT 0.21.3
- Executed: 2026-11-17T12:54:30Z – 2026-11-17T14:12:45Z

## Contents

| Path | What it is |
|---|---|
| `notebooks/RMT_A4_real_experiment.ipynb` | Notebook that produced these results (Colab, T4 GPU → Run all) |
| `data/assignment4/assignment4_real_6000.csv` | The 6,000-record dataset used by every run (SHA-256 in `dataset_manifest_measured.json`) |
| `data/assignment4/data_card.json` | Class counts and token-length statistics per split |
| `results/assignment4_measured/runs/` | One JSON per run: accuracy/F1 matrices, per-item predictions for every stage, resource logs |
| `results/assignment4_measured/tables/` | Summary tables, bootstrap and McNemar statistics, EWC calibration, replay sensitivity, error analysis |
| `results/assignment4_measured/figures/` | Quick-look figure from the notebook |
| `results/assignment4_measured/report_figures/` | Figures 1–3 of the report |
| `results/assignment4_measured/env.json`, `config.json`, `manifest.json` | Environment, configuration and run manifest |

## Reproduce

Open the notebook in Google Colab, select a T4 GPU and run all cells (about 80 minutes). The notebook downloads the datasets and the model from the Hugging Face Hub and writes the same file layout.
