# Cross-Domain Human Activity Recognition with Frozen LLM Backbones

This repository contains the code and results for my thesis work on cross-domain Human Activity Recognition (HAR) using frozen large language model (LLM) backbones as feature extractors.

The central question is: **can a frozen LLM trained on text generalize sensor embeddings across entirely different HAR datasets, without retraining?**

---

## Methodology

The pipeline is organized into three modules that together bridge the gap between raw IMU sensor data and a frozen LLM's representation space.

### Sensor Data Adaptation Module

This module converts raw IMU windows into a sequence of patch tokens the LLM can process.

**Per-channel instance normalization** (`SensorInstanceNorm`) is applied first. Each of the 6 sensor channels (3-axis accelerometer + 3-axis gyroscope) is independently normalized per sample, removing subject-specific amplitude differences and device placement biases before any further processing.

**Position-augmented patching** (`SensorEmbedding`) segments each channel into 8 non-overlapping patches of 15 timesteps, producing 48 tokens (6 channels × 8 patches). Normalized time indices [0, 1/(T−1), …, 1] are concatenated with the sensor values within each patch before the Conv1D projection. This gives the model explicit temporal order information without relying on absolute positional encodings that may not transfer across datasets. The Conv1D projects directly into each LLM's native `hidden_size` (auto-detected from `config.hidden_size`), eliminating the 768→d_model projection bottleneck.

A **learnable positional encoding** (`LearnablePositionalEncoding`) is added on top of the patch embeddings to give the LLM coarse token-order awareness at the sequence level.

### Sensor Knowledge Learning Module

This module addresses the modality gap between sensor time-series and the text representations the LLM was pretrained on.

A **gated cross-attention reprogramming layer** (`ReprogrammingLayer`) maintains a small set of 64 learned activity prototypes in the LLM's embedding space. Each sensor patch token attends to these prototypes via multi-head cross-attention (8 heads), pulling in activity-relevant information from the LLM's prior knowledge. A learnable scalar gate (initialized to 0) controls how strongly the prototype information blends into the token representation via a residual connection:

```
output = LayerNorm(patch + gate * cross_attn(patch, prototypes, prototypes))
```

The gate-at-zero initialization ensures the model starts as a pure passthrough and gradually learns to leverage prototypes, stabilizing early training.

To further align sensor representations with the LLM's input distribution, **two-stage training** is used:

- **Stage 1 — Alignment**: Self-supervised next-patch prediction. The `AlignmentHead` predicts the embedding of each next patch from the current LLM hidden state. The loss combines time-domain MSE with a frequency-domain FFT loss, so the model learns the periodic and rhythmic patterns characteristic of human motion.
- **Stage 2 — Classification**: Cross-entropy training with early stopping on validation macro F1. An auxiliary `ReconstructionHead` loss (α=0.1) encourages the LLM to preserve sensor-level information throughout the forward pass.

### Efficiency Enhancement Module

This module makes the pipeline practical without sacrificing transferability.

**Proportional layer truncation**: only the first ~33% of transformer layers are kept (e.g., 4 of 12 for GPT-2, 9 of 28 for Qwen-1.5B). Shallow LLM layers encode more general sequential representations that transfer better across modalities; deeper layers encode task-specific text patterns that are less useful for sensor data.

**LayerNorm-only fine-tuning**: all attention and feed-forward weights are frozen. Only the LayerNorm scale and bias parameters — less than 1% of total parameters — are trained. This prevents overfitting to a single source dataset and forces cross-domain generalization through the frozen LLM's representations.

**Full signal path:**
```
Raw IMU (N×120×6)
  → SensorInstanceNorm             [per-channel, per-sample normalization]
  → SensorEmbedding                [position-augmented Conv1D: → (N, 48, d_model)]
  → LearnablePositionalEncoding    [token-level position bias]
  → ReprogrammingLayer             [gated cross-attn with 64 activity prototypes]
  → Frozen LLM Backbone            [truncated ~33%, LayerNorm trainable only]
  → ActivityProjection             [last-token → 4-class logits]
     └─ ReconstructionHead         [auxiliary MSE+FFT loss during Stage 2]
```

---

## Experiments

### Phase 1: Multi-LLM Baseline

Seven LLM backbones tested with the same architecture: Conv1D patching directly to the LLM's native hidden size, LayerNorm-only fine-tuning, no reprogramming. Evaluated on all 12 cross-domain pairs across 4 HAR datasets.

Results in `results/phase1_multi_llm_results.csv`.

**LLM4HAR paper baseline (GPT-2, KDD 2025): 79.86% acc / 71.42% macro F1**
> Jin et al., "LLM4HAR: Generalizable On-device Human Activity Recognition via Large Language Models", KDD 2025.

| LLM | XD Acc% | XD F1% |
|-----|---------|--------|
| Qwen-1.5B | ~81 | ~72.50 |
| Llama-1B  | ~80 | ~71.73 |
| Qwen-0.5B | ~79 | ~68.54 |
| Gemma-2B  | ~72 | ~61.64 |

### Phase 2: Ablation Study

Four progressive levels isolating each component's contribution to cross-domain transfer, run across 4 LLM backbones and all 12 cross-domain pairs.

| Level | Components Added |
|-------|-----------------|
| **L1-Fixes** | UCI signal fix (`total_acc`), position-augmented patching, GPT-2 `wpe` zeroing |
| **L2a-Reprogram** | + Sensor Knowledge Learning Module (ReprogrammingLayer, 64 prototypes, 8 heads) |
| **L2b-TwoStage** | + Two-stage training (Stage 1 alignment + Stage 2 reconstruction auxiliary loss) |
| **L3-Full** | All components: reprogramming + two-stage training combined |

Results in `results/phase2_ablation_results.csv`.

**Best result: Qwen-1.5B × L2b-TwoStage — 81.55% acc / 74.48% macro F1 (+3.06% F1 over paper baseline)**

Key findings:
- L1-Fixes alone achieves ~71% avg XD F1 with only ~132K trainable parameters
- Two-stage alignment (L2b) is the single most impactful addition across all LLMs
- Reprogramming (L2a) benefits larger models but can hurt smaller ones with limited training data
- L3-Full does not consistently outperform L2b — combining both components introduces optimization tension on smaller datasets

---

## Datasets

Four publicly available IMU datasets, all resampled to 20 Hz and windowed at 6 s (120 samples), 6 channels (acc + gyro):

| Dataset | Activities | Subjects | Notes |
|---------|-----------|---------|-------|
| UCI HAR | Walk, Stand/Sit, Upstairs, Downstairs | 30 | Pre-windowed 128→120; uses `total_acc` |
| Shoaib  | Walk, Stand, Upstairs, Downstairs | 10 | Right-pocket placement |
| MotionSense | Walk, Stand, Upstairs, Downstairs | 24 | iPhone, userAcc+rotationRate |
| HHAR | Walk, Stand, Upstairs, Downstairs | 9 | Multiple device placements |

Activities are mapped to 4 classes: Walking (0), Standing/Sitting merged (1), Upstairs (2), Downstairs (3). Data loaders handle all dataset-specific quirks. Download datasets separately and point scripts at the root directory.

---

## Project Structure

```
llm-har/
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing (all 4 datasets)
│   ├── model.py                # Phase 2 model (full architecture with all modules)
│   └── model_baseline.py       # Phase 1 model (simple Conv1D patching)
├── notebooks/
│   └── results_analysis.ipynb  # Figures and summary tables
├── results/
│   ├── phase1_multi_llm_results.csv
│   └── phase2_ablation_results.csv
├── run_baseline.py             # Phase 1: 7-LLM cross-domain benchmark
├── run_ablation.py             # Phase 2: 4-level ablation study
├── requirements.txt
└── .gitignore
```

---

## Usage

```bash
pip install -r requirements.txt

# Phase 1 — 7-LLM cross-domain benchmark
python run_baseline.py --data_path /path/to/Datasets --results_dir results/

# Phase 2 — 4-level ablation study
python run_ablation.py --data_path /path/to/Datasets \
                       --results_dir results/ \
                       --ckpt_dir checkpoints/
```

Both scripts resume automatically if interrupted — previously completed runs are skipped based on the output CSV.

---

## Environment

Developed on Google Colab A100 (40 GB). Phase 2 full run takes approximately 18–22 hours on A100.

```
Python 3.10
PyTorch 2.x
transformers 4.40+
```

Some LLMs (Llama-3.2, Gemma-2) require accepting their HuggingFace license agreements before download.

---

## References

- **LLM4HAR**: Jin, Y., Shao, Z., Liu, J., Wang, H., Niu, B., Chen, X., & Xiong, H. (2025). *LLM4HAR: Generalizable On-device Human Activity Recognition via Large Language Models*. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025). [[Paper]](https://dl.acm.org/doi/10.1145/3690624.3709241)

- **Time-LLM**: Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P. Y., Liang, Y., Li, Y. F., Pan, S., & Wen, Q. (2024). *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*. In International Conference on Learning Representations (ICLR 2024). [[Paper]](https://arxiv.org/abs/2310.01728)

- **GPT-2**: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Blog. [[Paper]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **Qwen2.5**: Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv preprint arXiv:2412.15115. [[Paper]](https://arxiv.org/abs/2412.15115)

- **Llama 3**: Grattafiori, A., et al. (2024). *The Llama 3 Herd of Models*. arXiv preprint arXiv:2407.21783. [[Paper]](https://arxiv.org/abs/2407.21783)
