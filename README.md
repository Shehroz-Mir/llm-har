# Cross-Domain Human Activity Recognition with Frozen LLM Backbones

This repository contains the code and results for my MS thesis work on cross-domain Human Activity Recognition (HAR) using frozen large language model (LLM) backbones as feature extractors.

The central question is: **can a frozen LLM trained on text generalize sensor embeddings across entirely different HAR datasets, without retraining?**

---

## Approach

The pipeline represents time-series sensor data as a sequence of patch tokens and feeds them into a frozen, truncated LLM backbone. Only LayerNorm parameters are fine-tuned; the attention and feed-forward weights are fully frozen. This forces generalization through the LLM's pre-trained representations rather than task-specific overfitting.

**Signal path:**
```
Raw IMU (120×6) → SensorInstanceNorm → position-augmented Conv1D patching
→ LearnablePositionalEncoding → [ReprogrammingLayer] → Frozen LLM (truncated)
→ last-token CLS head → 4-class activity prediction
```

Key design choices:
- **Proportional truncation**: keep ~33% of total layers per model (balances expressiveness vs. overfitting)
- **Native hidden size**: Conv1D projects directly to the LLM's `hidden_size` — no 768→d_model bottleneck
- **Position-augmented patching**: time indices [0,1] concatenated with 6 sensor channels before Conv1d, giving the model explicit temporal order
- **SO(3) rotation augmentation**: per-sample random rotation on acc+gyro channels for device-orientation invariance

---

## Experiments

### Phase 1: Multi-LLM Baseline

Seven LLM backbones with LayerNorm-only fine-tuning, evaluated on all 12 cross-domain pairs across 4 HAR datasets.

Results in `results/phase1_multi_llm_results.csv`.

**LLM4HAR paper baseline (GPT-2, KDD 2025): 79.86% acc / 71.42% macro F1**
> Jin et al., "LLM4HAR: Generalizable On-device Human Activity Recognition via Large Language Models", KDD 2025.

| LLM | XD Acc% | XD F1% |
|-----|---------|--------|
| Qwen-1.5B | ~81 | ~72.50 |
| Llama-1B  | ~80 | ~71.73 |
| Qwen-0.5B | ~79 | ~68.54 |
| Gemma-2B  | ~72 | ~61.64 |

### Phase 3: Ablation Study

Four progressive levels isolating each component's contribution to cross-domain transfer:

| Level | Changes |
|-------|---------|
| **L1-Fixes** | UCI `total_acc` fix, position-augmented patching, `wpe` zeroing |
| **L2a-Reprogram** | + Gated cross-attention reprogramming layer (64 prototypes, 8 heads) |
| **L2b-TwoStage** | + Self-supervised alignment pre-training + reconstruction auxiliary loss |
| **L3-Full** | All of the above combined |

Results in `results/phase3_ablation_results.csv`.

**Best result: Qwen-1.5B × L2b-TwoStage — 81.55% acc / 74.48% F1 (+3.06% F1 over paper baseline)**

---

## Datasets

Four publicly available IMU datasets, all resampled to 20 Hz and windowed at 6 s (120 samples), 6 channels (acc + gyro):

| Dataset | Activities | Subjects | Notes |
|---------|-----------|---------|-------|
| UCI HAR | Walk, Stand/Sit, Upstairs, Downstairs | 30 | Pre-windowed 128→120 |
| Shoaib  | Walk, Stand, Upstairs, Downstairs | 10 | Right-pocket placement |
| MotionSense | Walk, Stand, Upstairs, Downstairs | 24 | iPhone, userAcc+rotationRate |
| HHAR | Walk, Stand, Upstairs, Downstairs | 9 | Multiple device placements |

Data loaders handle all dataset-specific quirks (nested CSVs, header offsets, per-subject resampling). Download datasets separately and point scripts at the root directory.

---

## Project Structure

```
llm-har/
├── src/
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── model.py             # Phase 3 model (position-augmented + reprogramming)
│   └── model_baseline.py    # Phase 1 model (simple Conv1d patching)
├── notebooks/
│   └── results_analysis.ipynb  # Figures and summary tables
├── results/
│   ├── phase1_multi_llm_results.csv
│   └── phase3_ablation_results.csv
├── run_baseline.py          # Phase 1: 7-LLM benchmark
├── run_ablation.py          # Phase 3: 4-level ablation
├── requirements.txt
└── .gitignore
```

---

## Usage

```bash
pip install -r requirements.txt

# Phase 1 — 7-LLM cross-domain benchmark
python run_baseline.py --data_path /path/to/Datasets --results_dir results/

# Phase 3 — 4-level ablation study
python run_ablation.py --data_path /path/to/Datasets \
                       --results_dir results/ \
                       --ckpt_dir checkpoints/
```

Both scripts resume automatically if interrupted — previously completed runs are skipped.

---

## Environment

Developed on Google Colab A100 (40 GB). Phase 3 full run takes ~18–22 hours on A100.

```
Python 3.10
PyTorch 2.x
transformers 4.40+
```

Some LLMs (Llama-3.2, Gemma-2) require accepting their HuggingFace license agreements.
