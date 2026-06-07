# Cross-Domain Human Activity Recognition with Frozen LLM Backbones

This repository contains the code and results for my MS thesis work on cross-domain Human Activity Recognition (HAR) using frozen large language model (LLM) backbones as feature extractors.

The central question is: **can a frozen LLM trained on text generalize sensor embeddings across entirely different HAR datasets, without retraining?**

---

## Methodology

The pipeline is organized into three modules that together bridge the gap between raw IMU sensor data and a frozen LLM's representation space.

### Sensor Data Adaptation Module

This module is responsible for converting raw IMU windows into a sequence of patch tokens that the LLM can process.

**Per-channel instance normalization** (`SensorInstanceNorm`) is applied first. Each of the 6 sensor channels (3-axis accelerometer + 3-axis gyroscope) is independently normalized per sample. This removes subject-specific amplitude differences and device placement biases before any further processing.

**Position-augmented patching** (`SensorEmbedding`) then segments each channel into 8 non-overlapping patches of 15 timesteps each, producing 48 tokens (6 channels × 8 patches). Critically, normalized time indices [0, 1/(T−1), …, 1] are concatenated with the sensor values within each patch before the Conv1D projection. This gives the model explicit temporal order information without relying on absolute positional encodings that may not transfer across datasets. The Conv1D kernel and stride both equal 30 (= 2 × patch length) to ensure non-overlapping, non-leaking projections into the LLM's native hidden size.

A **learnable positional encoding** (`LearnablePositionalEncoding`) is added on top of the patch embeddings to give the LLM coarse token-order awareness at the sequence level.

### Sensor Knowledge Learning Module

This module addresses the modality gap between sensor time-series and the text representations the LLM was pretrained on.

A **gated cross-attention reprogramming layer** (`ReprogrammingLayer`) is inserted between the patch embeddings and the LLM backbone. It maintains a small set of 64 learned activity prototypes in the LLM's embedding space. Each sensor patch token attends to these prototypes via multi-head cross-attention (8 heads), pulling in activity-relevant information from the LLM's prior knowledge. A learnable scalar gate (initialized to 0) controls how strongly this prototype information is blended in via a residual connection:

```
output = LayerNorm(patch + gate * cross_attn(patch, prototypes, prototypes))
```

The gate-at-zero initialization ensures the model starts as a pure passthrough and gradually learns to leverage prototypes, which stabilizes early training. This is inspired by the OpenTSLM gating mechanism.

To further align sensor representations with the LLM's input distribution, a **two-stage training** procedure is used:

- **Stage 1 (Alignment)**: The model is pre-trained in a self-supervised manner using a next-patch prediction objective. The `AlignmentHead` predicts the embedding of each next patch from the current LLM hidden state. The loss combines time-domain MSE with a frequency-domain FFT loss, so the model captures the periodic and rhythmic patterns characteristic of human motion.
- **Stage 2 (Classification)**: Standard cross-entropy training with early stopping on validation macro F1. An auxiliary `ReconstructionHead` loss (weighted at α=0.1) encourages the LLM to preserve sensor-level information throughout the forward pass.

### Efficiency Enhancement Module

Deploying a full LLM for sensor classification is computationally prohibitive. This module makes the pipeline practical without sacrificing transferability.

**Proportional layer truncation**: only the first ~33% of transformer layers are kept (e.g., 4 of 12 for GPT-2, 9 of 28 for Qwen-1.5B). Deeper layers in LLMs tend to encode task-specific text patterns that are less useful for sensor data; shallow layers encode more general sequential representations that transfer better.

**LayerNorm-only fine-tuning**: all attention and feed-forward weights are frozen. Only the LayerNorm scale and bias parameters — which constitute less than 1% of total parameters — are trained. This prevents overfitting to a single source dataset and forces cross-domain generalization through the frozen LLM's representations.

**Native hidden size**: the Conv1D in the Sensor Data Adaptation Module projects directly into the LLM's native `hidden_size` (auto-detected from `config.hidden_size`). This eliminates the 768→d_model→768 projection bottleneck that appears when forcing all LLMs to a fixed embedding dimension.

**Full signal path:**
```
Raw IMU (N×120×6)
  → SensorInstanceNorm             [per-channel, per-sample]
  → SensorEmbedding                [position-augmented Conv1D: 6×8 patches → (N, 48, d_model)]
  → LearnablePositionalEncoding    [token-level position bias]
  → ReprogrammingLayer             [gated cross-attn with 64 activity prototypes]
  → Frozen LLM Backbone            [truncated, LayerNorm trainable only]
  → ActivityProjection             [last-token → 4-class logits]
     └─ ReconstructionHead         [auxiliary MSE+FFT loss during Stage 2]
```

---

## Experiments

### Phase 1: Multi-LLM Baseline

Seven LLM backbones tested with the same simple architecture: Conv1D patching directly to the LLM's hidden size, LayerNorm-only fine-tuning, no reprogramming. Evaluated on all 12 cross-domain pairs across 4 HAR datasets.

Results in `results/phase1_multi_llm_results.csv`.

**LLM4HAR paper baseline (GPT-2, KDD 2025): 79.86% acc / 71.42% macro F1**
> Jin et al., "LLM4HAR: Generalizable On-device Human Activity Recognition via Large Language Models", KDD 2025.

| LLM | Params | XD Acc% | XD F1% |
|-----|--------|---------|--------|
| Qwen-1.5B | 1.5B | ~81 | ~72.50 |
| Llama-1B  | 1B   | ~80 | ~71.73 |
| Qwen-0.5B | 0.5B | ~79 | ~68.54 |
| Pythia-160M | 160M | — | — |
| OPT-125M  | 125M | — | — |
| Phi-3.5-Mini | 3.8B | — | ~68.58 (cross-domain) |
| Gemma-2B  | 2B   | ~72 | ~61.64 |

### Phase 2: Native d_model Architecture

Based on Phase 1, a key bottleneck was identified: forcing all LLMs through a fixed 768-dimensional projection introduced a representational bottleneck for larger models (Qwen-1.5B has d_model=1536, Llama-1B has d_model=2048). Phase 2 eliminated the input and output projection layers entirely, projecting sensor patches directly into each LLM's native `hidden_size` via the Conv1D.

Additional data-side fixes were applied in this phase:
- **UCI signal correction**: replaced `body_acc` (gravity-subtracted) with `total_acc` (raw signal including gravity component) to match the signal characteristics of Shoaib, MotionSense, and HHAR — all three provide raw accelerometer readings. Using `body_acc` in cross-domain training causes the model to learn a gravity-free signal distribution that fails to generalize.
- **GPT-2 `wpe` zeroing**: GPT-2's learned word position embeddings (`wpe`) were zeroed and frozen to prevent double positional encoding when `LearnablePositionalEncoding` is already applied.

This architecture forms the foundation for Phase 3.

### Phase 3: Ablation Study

Four progressive levels isolating each component's contribution, run across 4 LLM backbones and all 12 cross-domain pairs.

| Level | Components Added |
|-------|-----------------|
| **L1-Fixes** | Phase 2 architecture (native d_model, UCI fix, position-augmented patching, `wpe` zeroing) |
| **L2a-Reprogram** | + Sensor Knowledge Learning Module (ReprogrammingLayer, 64 prototypes, 8 heads) |
| **L2b-TwoStage** | + Two-stage training (Stage 1 alignment + Stage 2 reconstruction auxiliary loss) |
| **L3-Full** | All components: reprogramming + two-stage training combined |

Results in `results/phase3_ablation_results.csv`.

**Best result: Qwen-1.5B × L2b-TwoStage — 81.55% acc / 74.48% macro F1 (+3.06% F1 over paper baseline)**

Key findings:
- L1-Fixes alone achieves competitive results (~71% avg XD F1) with only ~132K trainable parameters
- Two-stage alignment (L2b) is the single most impactful addition across all LLMs
- Reprogramming (L2a) benefits larger models (Qwen-1.5B) but can hurt smaller ones with limited data
- L3-Full does not consistently outperform L2b — combining both components introduces optimization tension for smaller datasets

---

## Datasets

Four publicly available IMU datasets, all resampled to 20 Hz and windowed at 6 s (120 samples), 6 channels (acc + gyro):

| Dataset | Activities | Subjects | Notes |
|---------|-----------|---------|-------|
| UCI HAR | Walk, Stand/Sit, Upstairs, Downstairs | 30 | Pre-windowed 128→120; uses `total_acc` |
| Shoaib  | Walk, Stand, Upstairs, Downstairs | 10 | Right-pocket placement |
| MotionSense | Walk, Stand, Upstairs, Downstairs | 24 | iPhone, userAcc+rotationRate |
| HHAR | Walk, Stand, Upstairs, Downstairs | 9 | Multiple device placements |

Activities are mapped to 4 classes: Walking (0), Standing/Sitting merged (1), Upstairs (2), Downstairs (3). Data loaders handle all dataset-specific quirks (nested CSVs, header offsets, per-subject resampling). Download datasets separately and point scripts at the root directory.

---

## Project Structure

```
llm-har/
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing (all 4 datasets)
│   ├── model.py                # Phase 2/3 model (full architecture)
│   └── model_baseline.py       # Phase 1 model (simple Conv1D patching)
├── notebooks/
│   └── results_analysis.ipynb  # Figures and summary tables
├── results/
│   ├── phase1_multi_llm_results.csv
│   └── phase3_ablation_results.csv
├── run_baseline.py             # Phase 1: 7-LLM cross-domain benchmark
├── run_ablation.py             # Phase 3: 4-level ablation study
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

Both scripts resume automatically if interrupted — previously completed runs are skipped based on the output CSV.

---

## Environment

Developed on Google Colab A100 (40 GB). Phase 3 full run takes approximately 18–22 hours on A100.

```
Python 3.10
PyTorch 2.x
transformers 4.40+
```

Some LLMs (Llama-3.2, Gemma-2) require accepting their HuggingFace license agreements before download.
