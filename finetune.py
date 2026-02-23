#!/usr/bin/env python3
"""Finetune blast-ct DeepMedic on your labeled head CT exams.

Workflow
--------
1. Loads the best pretrained model from blast-ct/blast_ct/data/saved_models/
2. Runs the blast-ct PyTorch training loop on your data
3. Saves finetuned checkpoints to --job-dir

Usage
-----
# Step 1: prepare data CSVs (only needed once)
python prepare_finetune_data.py

# Step 2: finetune
python finetune.py

# Or with explicit paths:
python finetune.py \\
    --train-csv finetune_data/train.csv \\
    --val-csv   finetune_data/val.csv \\
    --job-dir   finetune_runs/run_001 \\
    --num-epochs 50 \\
    --pretrained-model blast-ct/blast_ct/data/saved_models/model_1.torch_model
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# Patch SegmentationMetrics to show num_correct / num_total instead of
# NaN-prone precision/recall/f1.  The original metric_fns dict captures
# function references at class-definition time, so replacing module-level
# names doesn't help — we need to patch the class directly.
import blast_ct.trainer.metrics as _metrics
import numpy as _np


def _patched_report(self):
    """Print num_correct / num_total per class from the confusion matrix."""
    cm = self.value  # (num_classes, num_classes) numpy array
    total_all = _np.sum(cm)
    correct_all = _np.sum(_np.diag(cm))
    message = f"{'OVERALL'.ljust(20)}:  correct/total = {int(correct_all)} / {int(total_all)}" \
              f"  ({100.0 * correct_all / max(total_all, 1):.2f}%)\n"
    for i, class_name in enumerate(self.class_names):
        total_i = _np.sum(cm[i, :])   # all voxels whose true label is i
        correct_i = cm[i, i]           # correctly predicted as i
        message += f"{class_name.upper().ljust(20)}:  correct/total = {int(correct_i)} / {int(total_i)}" \
                   f"  ({100.0 * correct_i / max(total_i, 1):.2f}%)\n"
    return message


_metrics.SegmentationMetrics.report = _patched_report

from blast_ct.read_config import (
    get_model, get_optimizer, get_loss,
    get_train_loader, get_valid_loader, get_test_loader, get_training_hooks,
)
from blast_ct.trainer.model_trainer import ModelTrainer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DEFAULT_TRAIN_CSV    = REPO_ROOT / "finetune_data" / "train.csv"
DEFAULT_VAL_CSV      = REPO_ROOT / "finetune_data" / "val.csv"
DEFAULT_CONFIG       = REPO_ROOT / "finetune_config.json"
DEFAULT_JOB_DIR      = REPO_ROOT / "finetune_runs" / "run_001"
DEFAULT_PRETRAINED   = (REPO_ROOT / "blast-ct" / "blast_ct" / "data"
                        / "saved_models" / "model_1.torch_model")


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

def load_pretrained_weights(model: torch.nn.Module, model_path: Path) -> None:
    """Transfer pretrained weights into a (possibly wider) model.

    All layers that match exactly are copied.  For the final output conv
    (which changes shape when num_classes differs), pretrained weights for
    the original classes are copied into the corresponding rows and the new
    class rows are left as random init — so the model starts with a good
    feature extractor but learns the new class head from scratch.
    """
    print(f"Loading pretrained weights from: {model_path}")
    state_dict = torch.load(str(model_path), map_location="cpu")

    model_state = model.state_dict()
    loaded, skipped, partial = [], [], []

    for k, pretrained_v in state_dict.items():
        if k not in model_state:
            skipped.append(k)
            continue

        model_v = model_state[k]

        if pretrained_v.shape == model_v.shape:
            # Exact match — copy directly
            model_state[k] = pretrained_v
            loaded.append(k)

        elif pretrained_v.shape[1:] == model_v.shape[1:]:
            # Output dimension expanded (num_classes axis = dim 0):
            # copy pretrained rows, leave new rows as random init
            n_old = pretrained_v.shape[0]
            model_state[k][:n_old] = pretrained_v
            partial.append(f"{k}  [{pretrained_v.shape} → {model_v.shape}, first {n_old} rows copied]")

        else:
            skipped.append(k)

    model.load_state_dict(model_state)

    print(f"  Exact match : {len(loaded)} tensors")
    if partial:
        print(f"  Partial copy (output layer expanded):")
        for p in partial:
            print(f"    {p}")
    if skipped:
        print(f"  Skipped (shape incompatible): {skipped}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(
    train_csv: Path,
    val_csv: Path,
    config_file: Path,
    job_dir: Path,
    num_epochs: int,
    pretrained_model: Path | None,
    random_seed: int,
) -> None:
    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    # CPU only on Mac (no CUDA)
    device = torch.device("cpu")
    print("Running on CPU (MPS not used by blast-ct training loop)")

    job_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(config_file) as f:
        config = json.load(f)

    # Model
    model = get_model(config)
    if pretrained_model and pretrained_model.exists():
        load_pretrained_weights(model, pretrained_model)
    else:
        print("⚠️  No pretrained weights loaded — training from scratch.")

    # Data loaders
    train_loader = get_train_loader(config, model, str(train_csv), use_cuda=False)
    valid_loader = get_valid_loader(config, model, str(val_csv),   use_cuda=False)
    test_loader  = get_test_loader( config, model, str(val_csv),   use_cuda=False)

    # Optimiser + scheduler
    lr_scheduler = get_optimizer(config, model)
    criterion    = get_loss(config)
    hooks        = get_training_hooks(str(job_dir), config, device, valid_loader, test_loader)

    # Train
    print(f"\nStarting finetuning for {num_epochs} epochs → {job_dir}")
    trainer = ModelTrainer(str(job_dir), device, model, criterion, lr_scheduler, hooks, task="segmentation")
    trainer(train_loader, num_epochs)

    print(f"\nDone. Checkpoints saved in: {job_dir / 'saved_models'}/")
    print("To use your finetuned model for inference, point --model to the saved .torch_model file.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv",        type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val-csv",          type=Path, default=DEFAULT_VAL_CSV)
    p.add_argument("--config-file",      type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--job-dir",          type=Path, default=DEFAULT_JOB_DIR)
    p.add_argument("--num-epochs",       type=int,  default=150,
                   help="Total training epochs. ~100-200 is reasonable for <10 cases.")
    p.add_argument("--pretrained-model", type=Path, default=DEFAULT_PRETRAINED,
                   help="Path to a .torch_model pretrained checkpoint to start from.")
    p.add_argument("--random-seed",      type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Sanity checks
    if not args.train_csv.exists():
        print(f"❌ train CSV not found: {args.train_csv}")
        print("   Run:  python prepare_finetune_data.py   first.")
        return
    if not args.val_csv.exists():
        print(f"❌ val CSV not found: {args.val_csv}")
        print("   Run:  python prepare_finetune_data.py   first.")
        return

    finetune(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        config_file=args.config_file,
        job_dir=args.job_dir,
        num_epochs=args.num_epochs,
        pretrained_model=args.pretrained_model,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
