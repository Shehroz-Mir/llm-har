"""
Phase 1: Multi-LLM Cross-Domain HAR Baseline Study

Trains 7 LLM backbones on each of 4 HAR datasets and evaluates cross-domain
transfer to all other datasets. Uses a frozen LLM backbone with LayerNorm-only
fine-tuning and a Conv1D sensor embedding.

LLM4HAR paper baseline (GPT-2): 79.86% accuracy / 71.42% F1
  Jin et al., "LLM4HAR: Generalizable On-device Human Activity Recognition
  via Large Language Models", KDD 2025.

Usage:
    python run_baseline.py --data_path /path/to/Datasets --results_dir results/
"""
import sys
import os
import gc
import time
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import get_dataset
from src.model_baseline import LLM4HAR

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LLMS_TO_TEST = {
    "OPT-125M":     {"hf_id": "facebook/opt-125m",               "layers_to_keep": 4},
    "Pythia-160M":  {"hf_id": "EleutherAI/pythia-160m",          "layers_to_keep": 4},
    "Qwen-0.5B":    {"hf_id": "Qwen/Qwen2.5-0.5B",              "layers_to_keep": 8},
    "Qwen-1.5B":    {"hf_id": "Qwen/Qwen2.5-1.5B",              "layers_to_keep": 9},
    "Llama-1B":     {"hf_id": "meta-llama/Llama-3.2-1B",         "layers_to_keep": 5},
    "Gemma-2B":     {"hf_id": "google/gemma-2-2b",               "layers_to_keep": 9},
    "Phi-3.5-Mini": {"hf_id": "microsoft/Phi-3.5-mini-instruct", "layers_to_keep": 11},
}

DATASETS = ['uci', 'shoaib', 'motionsense', 'hhar']
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
NUM_CLASSES = 4

# LLM4HAR paper baseline for reference (GPT-2, KDD 2025)
PAPER_BASELINE = {"accuracy": 79.86, "f1": 71.42}

# =============================================================================
# HELPERS
# =============================================================================

def make_loaders(X, y, batch_size=BATCH_SIZE, val_ratio=0.125, test_ratio=0.2):
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    def _dl(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(ys).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True, drop_last=shuffle)

    return (_dl(X[:n_train], y[:n_train], True),
            _dl(X[n_train:n_train+n_val], y[n_train:n_train+n_val], False),
            _dl(X[n_train+n_val:], y[n_train+n_val:], False))


def augment_batch(x):
    """Per-sample random SO(3) rotation on acc and gyro channels."""
    N = x.shape[0]
    R = torch.randn(N, 3, 3, device=x.device)
    Q, _ = torch.linalg.qr(R)
    det = torch.det(Q)
    Q[det < 0, :, 0] *= -1
    x_aug = x.clone()
    x_aug[:, :, :3] = torch.bmm(x[:, :, :3], Q.transpose(1, 2))
    x_aug[:, :, 3:] = torch.bmm(x[:, :, 3:], Q.transpose(1, 2))
    return x_aug


def train_model(model, train_dl, val_dl):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = MAX_EPOCHS * len(train_dl)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=min(100 / max(total_steps, 1), 0.3), anneal_strategy='cos'
    )
    best_f1, wait = 0.0, 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                labels.extend(yb.numpy())
        val_f1 = f1_score(labels, preds, average='macro', zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch}. Best Val F1: {best_f1:.4f}")
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    preds, labels = [], []
    for xb, yb in dl:
        preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return acc, f1


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print(f"Device: {DEVICE}")
    print(f"Paper baseline (LLM4HAR, GPT-2): "
          f"Acc={PAPER_BASELINE['accuracy']}% / F1={PAPER_BASELINE['f1']}%\n")

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading datasets...")
    dataset_cache = {}
    for ds in DATASETS:
        X, y = get_dataset(ds, args.data_path)
        dataset_cache[ds] = (X, y)

    all_results = []

    for llm_name, llm_cfg in LLMS_TO_TEST.items():
        print(f"\n{'='*60}")
        print(f"LLM: {llm_name} ({llm_cfg['hf_id']}) — {llm_cfg['layers_to_keep']} layers")
        print(f"{'='*60}")

        for source in DATASETS:
            print(f"\n  Source: {source}")
            X_src, y_src = dataset_cache[source]
            train_dl, val_dl, test_dl = make_loaders(X_src, y_src)

            try:
                model = LLM4HAR(
                    llm_name=llm_cfg['hf_id'],
                    llm_layers_to_keep=llm_cfg['layers_to_keep'],
                    num_classes=NUM_CLASSES,
                )
                model = train_model(model, train_dl, val_dl)

                for target in DATASETS:
                    if target == source:
                        acc, f1 = evaluate(model, test_dl)
                    else:
                        X_tgt, y_tgt = dataset_cache[target]
                        tgt_dl = DataLoader(
                            TensorDataset(torch.from_numpy(X_tgt).float(),
                                          torch.from_numpy(y_tgt).long()),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True
                        )
                        acc, f1 = evaluate(model, tgt_dl)

                    tag = "ID" if target == source else "XD"
                    print(f"    {source} -> {target} [{tag}]: Acc={acc:.4f} F1={f1:.4f}")
                    all_results.append({
                        'llm': llm_name, 'source': source, 'target': target,
                        'accuracy': acc, 'f1_macro': f1,
                    })

            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()

        # Save after each LLM
        results_path = os.path.join(args.results_dir, 'phase1_multi_llm_results.csv')
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        print(f"\n  Saved -> {results_path}")

    # Summary
    df = pd.DataFrame(all_results)
    xd = df[df['source'] != df['target']]
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY (vs LLM4HAR paper baseline: F1={:.2f}%)".format(
        PAPER_BASELINE['f1']))
    print(f"{'='*60}")
    print(f"{'LLM':<14} {'XD Acc%':>8} {'XD F1%':>8} {'Δ F1':>8}")
    print("-" * 42)
    for llm in LLMS_TO_TEST:
        sub = xd[xd['llm'] == llm]
        if len(sub) > 0:
            xa = sub['accuracy'].mean() * 100
            xf = sub['f1_macro'].mean() * 100
            delta = xf - PAPER_BASELINE['f1']
            print(f"{llm:<14} {xa:8.2f} {xf:8.2f} {delta:+8.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-LLM Cross-Domain HAR Baseline')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Datasets directory')
    parser.add_argument('--results_dir', type=str, default='results/',
                        help='Directory to save results CSV')
    main(parser.parse_args())
