"""
Phase 3: Cross-Domain HAR Ablation Study

4-level ablation across 4 LLM backbones and 4 HAR datasets (UCI, Shoaib,
MotionSense, HHAR). Each level adds architectural or training components
on top of the previous, isolating their contribution to cross-domain transfer.

Ablation levels:
  L1-Fixes       — improved baseline (UCI signal fix, position-augmented patching)
  L2a-Reprogram  — adds gated cross-attention reprogramming layer
  L2b-TwoStage   — adds self-supervised alignment + reconstruction auxiliary loss
  L3-Full        — all components combined

LLM4HAR paper baseline (GPT-2): 79.86% accuracy / 71.42% macro F1
  Jin et al., "LLM4HAR: Generalizable On-device Human Activity Recognition
  via Large Language Models", KDD 2025.

Usage:
    python run_ablation.py --data_path /path/to/Datasets --results_dir results/
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import LLM4HAR
from src.data_loader import get_dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RECON_ALPHA = 0.1  # Weight for reconstruction auxiliary loss

ABLATION_LEVELS = {
    "L1-Fixes": {
        "use_reprogramming": False, "two_stage": False, "use_recon_aux": False,
    },
    "L2a-Reprogram": {
        "use_reprogramming": True, "n_prototypes": 64, "two_stage": False,
        "use_recon_aux": False,
    },
    "L2b-TwoStage": {
        "use_reprogramming": False, "two_stage": True,
        "align_epochs": 30, "align_lr": 1e-4, "use_recon_aux": True,
    },
    "L3-Full": {
        "use_reprogramming": True, "n_prototypes": 64, "two_stage": True,
        "align_epochs": 30, "align_lr": 1e-4, "use_recon_aux": True,
    },
}

ABLATION_LLMS = {
    "GPT-2":     {"hf_id": "gpt2",                    "layers_to_keep": 4},
    "Qwen-0.5B": {"hf_id": "Qwen/Qwen2.5-0.5B",      "layers_to_keep": 8},
    "Llama-1B":  {"hf_id": "meta-llama/Llama-3.2-1B", "layers_to_keep": 5},
    "Qwen-1.5B": {"hf_id": "Qwen/Qwen2.5-1.5B",      "layers_to_keep": 9},
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


def train_alignment(model, train_dl, align_epochs=30, align_lr=1e-4, ckpt_path=None):
    """Stage 1: Self-supervised next-patch prediction alignment (MSE + FFT loss)."""
    model = model.to(DEVICE)

    dummy = torch.randn(2, 120, 6, device=DEVICE)
    _ = model.get_alignment_loss(dummy)

    align_params = [p for p in model.parameters() if p.requires_grad]
    existing_ids = {id(p) for p in align_params}
    for p in model._alignment_head.parameters():
        if id(p) not in existing_ids:
            align_params.append(p)

    optimizer = optim.AdamW(align_params, lr=align_lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=align_epochs * len(train_dl))

    start_epoch = 1
    best_loss = float('inf')

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            print(f"  [Align] Resumed from epoch {ckpt['epoch']}, loss={best_loss:.6f}")
        except Exception as e:
            print(f"  [Align] Checkpoint load failed ({e}), starting fresh")

    print(f"  [Align] Stage 1: next-patch prediction ({align_epochs} epochs)")
    for epoch in range(start_epoch, align_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, _ in train_dl:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            loss = model.get_alignment_loss(xb)
            loss.backward()
            nn.utils.clip_grad_norm_(align_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_dl), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Align Epoch {epoch:3d}/{align_epochs} | Loss: {epoch_loss:.6f}")
        if ckpt_path:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, ckpt_path)

    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    model._alignment_head = None
    print(f"  [Align] Stage 1 complete. Best loss: {best_loss:.6f}")
    return model


def train_model(model, train_dl, val_dl, ckpt_path=None, use_recon_aux=False):
    """Stage 2: Classification with early stopping on val macro F1."""
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
    start_epoch = 1

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_f1 = ckpt['best_f1']
            wait = ckpt['wait']
            best_state = ckpt.get('best_state')
            print(f"  >> Resumed from epoch {ckpt['epoch']}, best F1={best_f1:.4f}")
        except Exception as e:
            print(f"  >> Checkpoint load failed ({e}), starting fresh")
            best_f1, wait, best_state = 0.0, 0, None

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            if use_recon_aux:
                logits, recon_loss = model.forward_with_recon(xb)
                loss = criterion(logits, yb) + RECON_ALPHA * recon_loss
            else:
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

        if epoch % 10 == 0 or epoch == 1:
            val_acc = accuracy_score(labels, preds)
            print(f"  Epoch {epoch:3d} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch}. Best Val F1: {best_f1:.4f}")
                break

        if ckpt_path:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1, 'wait': wait, 'best_state': best_state,
            }, ckpt_path)

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return model


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    preds, labels = [], []
    for xb, yb in dl:
        preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds,
                                                    average='macro', zero_division=0)


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print(f"Device: {DEVICE}")
    print(f"Paper baseline (LLM4HAR, GPT-2): "
          f"Acc={PAPER_BASELINE['accuracy']}% / F1={PAPER_BASELINE['f1']}%\n")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Resume logic
    results_path = os.path.join(args.results_dir, 'phase3_ablation_results.csv')
    completed_runs = set()
    all_results = []

    if os.path.exists(results_path):
        prev_df = pd.read_csv(results_path)
        valid_df = prev_df[prev_df['accuracy'] > 0].copy()
        failed_df = prev_df[prev_df['accuracy'] == 0]
        if len(failed_df) > 0:
            print(f"Dropping {len(failed_df)} rows from failed runs.")
        all_results = valid_df.to_dict('records')
        for (level, llm, src), grp in valid_df.groupby(['level', 'llm', 'source']):
            if len(grp) >= len(DATASETS):
                completed_runs.add((level, llm, src))
        print(f"Resuming: {len(completed_runs)} completed runs found.")
    else:
        print("No existing results. Starting fresh.")

    total_runs = len(ABLATION_LEVELS) * len(ABLATION_LLMS) * len(DATASETS)
    print(f"Total runs: {total_runs} | Completed: {len(completed_runs)} | "
          f"Remaining: {total_runs - len(completed_runs)}\n")

    print("Loading datasets...")
    dataset_cache = {}
    for ds in DATASETS:
        X, y = get_dataset(ds, args.data_path)
        dataset_cache[ds] = (X, y)

    benchmark_start = time.time()

    for level_name, level_cfg in ABLATION_LEVELS.items():
        print(f"\n{'#'*60}")
        print(f"LEVEL: {level_name} | Reprogram={level_cfg['use_reprogramming']} | "
              f"TwoStage={level_cfg['two_stage']} | ReconAux={level_cfg['use_recon_aux']}")
        print(f"{'#'*60}")

        for llm_name, llm_cfg in ABLATION_LLMS.items():
            print(f"\n{'='*50}")
            print(f"LLM: {llm_name} ({llm_cfg['hf_id']}) | {llm_cfg['layers_to_keep']} layers")
            print(f"{'='*50}")

            for source in DATASETS:
                if (level_name, llm_name, source) in completed_runs:
                    print(f"\n  Source: {source} — skipped (already done)")
                    continue

                print(f"\n  Source: {source}")
                X_src, y_src = dataset_cache[source]
                train_dl, val_dl, test_dl = make_loaders(X_src, y_src)
                run_start = time.time()

                try:
                    model = LLM4HAR(
                        llm_name=llm_cfg['hf_id'],
                        llm_layers_to_keep=llm_cfg['layers_to_keep'],
                        num_classes=NUM_CLASSES,
                        use_reprogramming=level_cfg['use_reprogramming'],
                        n_prototypes=level_cfg.get('n_prototypes', 64),
                        use_recon_aux=level_cfg.get('use_recon_aux', False),
                    )
                    stats = model.get_param_stats()
                    print(f"  d_model={stats['d_model']} | "
                          f"Trainable={stats['total_trainable']:,} | "
                          f"Frozen={stats['total_frozen']:,}")

                    if level_cfg['two_stage']:
                        align_ckpt = os.path.join(
                            args.ckpt_dir,
                            f'align_{level_name}_{llm_name}_{source}.pt')
                        model = train_alignment(
                            model, train_dl,
                            align_epochs=level_cfg.get('align_epochs', 30),
                            align_lr=level_cfg.get('align_lr', 1e-4),
                            ckpt_path=align_ckpt,
                        )

                    cls_ckpt = os.path.join(
                        args.ckpt_dir,
                        f'cls_{level_name}_{llm_name}_{source}.pt')
                    model = train_model(
                        model, train_dl, val_dl,
                        ckpt_path=cls_ckpt,
                        use_recon_aux=level_cfg.get('use_recon_aux', False),
                    )

                    for target in DATASETS:
                        if target == source:
                            acc, f1 = evaluate(model, test_dl)
                        else:
                            X_tgt, y_tgt = dataset_cache[target]
                            tgt_dl = DataLoader(
                                TensorDataset(torch.from_numpy(X_tgt).float(),
                                              torch.from_numpy(y_tgt).long()),
                                batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)
                            acc, f1 = evaluate(model, tgt_dl)

                        tag = "ID" if target == source else "XD"
                        print(f"    {source} -> {target} [{tag}]: "
                              f"Acc={acc:.4f} F1={f1:.4f}")
                        all_results.append({
                            'level': level_name, 'llm': llm_name,
                            'source': source, 'target': target,
                            'accuracy': acc, 'f1_macro': f1,
                            'trainable_params': stats['total_trainable'],
                            'use_reprogramming': level_cfg['use_reprogramming'],
                            'two_stage': level_cfg['two_stage'],
                            'use_recon_aux': level_cfg['use_recon_aux'],
                        })

                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    for target in DATASETS:
                        all_results.append({
                            'level': level_name, 'llm': llm_name,
                            'source': source, 'target': target,
                            'accuracy': 0.0, 'f1_macro': 0.0,
                            'trainable_params': 0,
                            'use_reprogramming': level_cfg['use_reprogramming'],
                            'two_stage': level_cfg['two_stage'],
                            'use_recon_aux': level_cfg['use_recon_aux'],
                        })

                print(f"  Run time: {(time.time()-run_start)/60:.1f} min")
                gc.collect()
                torch.cuda.empty_cache()

            pd.DataFrame(all_results).to_csv(results_path, index=False)
            print(f"\n  Saved -> {results_path}")

    print(f"\nTotal time: {(time.time()-benchmark_start)/3600:.2f} hours")

    # Summary
    df = pd.DataFrame(all_results)
    xd = df[(df['source'] != df['target']) & (df['accuracy'] > 0)]
    print(f"\n{'='*65}")
    print(f"CROSS-DOMAIN SUMMARY (paper baseline: F1={PAPER_BASELINE['f1']}%)")
    print(f"{'='*65}")
    print(f"{'Level':<18} {'LLM':<12} {'XD Acc%':>8} {'XD F1%':>8} {'Δ F1':>7}")
    print("-" * 57)
    for level in ABLATION_LEVELS:
        for llm in ABLATION_LLMS:
            sub = xd[(xd['level'] == level) & (xd['llm'] == llm)]
            if len(sub) > 0:
                xa = sub['accuracy'].mean() * 100
                xf = sub['f1_macro'].mean() * 100
                delta = xf - PAPER_BASELINE['f1']
                print(f"{level:<18} {llm:<12} {xa:8.2f} {xf:8.2f} {delta:+7.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Domain HAR Ablation Study')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Datasets directory')
    parser.add_argument('--results_dir', type=str, default='results/',
                        help='Directory to save results CSV')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/',
                        help='Directory for training checkpoints')
    main(parser.parse_args())
