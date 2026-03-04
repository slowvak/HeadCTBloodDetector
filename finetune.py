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
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FocalDiceLoss(nn.Module):
    """Focal loss + soft Dice loss, combined as a weighted per-class average.

    For each true class c present in the batch:
      - Focal loss: mean of -(1-p_t)^gamma * log(p_t) over voxels whose true
        label is c, where p_t is the softmax probability of the correct class.
        The (1-p_t)^gamma term down-weights easy (already well-predicted)
        voxels so training focuses on hard examples.
      - Dice loss: 1 - 2*|P∩Y| / (|P|+|Y|) using soft (probability) maps,
        computed over the whole volume for class c.

    The two per-class losses are summed and divided by the same class-weight
    denominator, so the total loss is:

        L = (focal_total + dice_total) / denom

    SAH contributes twice as much as every other class (sah_weight=2.0).
    With 6 classes the effective denominator is 7 (5×1 + 1×2).
    """

    def __init__(self, class_names: list, sah_weight: float = 2.0,
                 gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        n = len(class_names)
        w = torch.ones(n)
        w[class_names.index('SAH')] = sah_weight
        self.register_buffer('class_weights', w)
        self.n_classes = n
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)   # (B, C, ...)
        focal_total = logits.new_zeros(1)
        dice_total  = logits.new_zeros(1)
        denom       = logits.new_zeros(1)

        for c in range(self.n_classes):
            mask = target == c
            if not mask.any():
                continue
            w_c = self.class_weights[c]
            denom = denom + w_c

            # Focal: mean over voxels whose true class is c
            p_t     = probs[:, c][mask]
            focal_c = ((1.0 - p_t) ** self.gamma * (-torch.log(p_t.clamp(min=1e-8)))).mean()
            focal_total = focal_total + w_c * focal_c

            # Soft Dice: over whole volume for class c
            p_c          = probs[:, c]
            y_c          = mask.float()
            intersection = (p_c * y_c).sum()
            dice_c       = 1.0 - (2.0 * intersection + self.smooth) / (
                               p_c.sum() + y_c.sum() + self.smooth)
            dice_total = dice_total + w_c * dice_c

        denom = denom.clamp(min=1e-8)
        return (focal_total + dice_total) / denom

from blast_ct.read_config import (
    get_model, get_optimizer, get_loss,
    get_train_loader, get_valid_loader, get_test_loader, get_training_hooks,
)
from blast_ct.trainer.model_trainer import ModelTrainer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DEFAULT_DATA_DIR     = Path.home() / "Desktop" / "CTHead"
DEFAULT_TRAIN_CSV    = REPO_ROOT / "finetune_data" / "train.csv"
DEFAULT_VAL_CSV      = REPO_ROOT / "finetune_data" / "val.csv"
DEFAULT_CONFIG       = REPO_ROOT / "finetune_config.json"
DEFAULT_JOB_DIR      = REPO_ROOT / "finetune_runs" / "run_001"
DEFAULT_PRETRAINED   = (REPO_ROOT / "blast-ct" / "blast_ct" / "data"
                        / "saved_models" / "model_1.torch_model")

MEDIAN_RADIUS = 2   # 2D 5×5 per axial slice (radius 2 = ±2 voxels = 5 total)


# ---------------------------------------------------------------------------
# Pre-processing: median filter
# ---------------------------------------------------------------------------

def _filter_nifti(src: Path, dst: Path) -> None:
    """Write a 5×5-per-axial-slice median-filtered copy of src to dst."""
    img = sitk.ReadImage(str(src))
    mf  = sitk.MedianImageFilter()
    mf.SetRadius([MEDIAN_RADIUS, MEDIAN_RADIUS, 0])   # x, y filtered; z unchanged
    sitk.WriteImage(mf.Execute(img), str(dst))


def build_filtered_csv(csv_path: Path, filtered_dir: Path,
                       columns: list[str]) -> Path:
    """Return path to a CSV identical to csv_path except that every file in
    *columns* has been replaced with a 5×5 median-filtered copy cached in
    filtered_dir.  Existing cached files are reused without re-filtering.
    """
    filtered_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    for col in columns:
        if col not in df.columns:
            continue
        new_paths = []
        for path_str in df[col]:
            src = Path(path_str)
            dst = filtered_dir / (src.name.replace(".nii.gz", "_filtered.nii.gz")
                                  .replace(".nii",    "_filtered.nii"))
            if not dst.exists():
                print(f"  Filtering {col}: {src.name} → {dst.name}")
                _filter_nifti(src, dst)
            new_paths.append(str(dst))
        df[col] = new_paths

    out_csv = filtered_dir / csv_path.name
    df.to_csv(out_csv, index=False)
    return out_csv


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

    # Pre-process: apply 5×5 median filter to images and labels
    filtered_dir = job_dir / "filtered_data"
    print(f"\nPre-filtering training data (5×5 median, cached in {filtered_dir})…")
    filtered_train_csv = build_filtered_csv(train_csv, filtered_dir, ["image", "target"])
    filtered_val_csv   = build_filtered_csv(val_csv,   filtered_dir, ["image", "target"])

    # Data loaders
    train_loader = get_train_loader(config, model, str(filtered_train_csv), use_cuda=False)
    valid_loader = get_test_loader(config, model, str(filtered_val_csv),    use_cuda=False)
    test_loader  = get_test_loader( config, model, str(filtered_val_csv),   use_cuda=False)

    # Optimiser + scheduler
    lr_scheduler = get_optimizer(config, model)

    # Loss: focal + soft Dice, per-class balanced, SAH weighted 2x
    criterion = FocalDiceLoss(config['data']['class_names'])
    print("Loss: focal (γ=2) + soft Dice  (SAH weight = 2.0)")
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
    p.add_argument("--data-dir",         type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory containing CT-X / CT-X_stripped / CT-X_prediction files. "
                        "Used to auto-generate train/val CSVs when they do not already exist. "
                        f"Default: {DEFAULT_DATA_DIR}")
    p.add_argument("--train-csv",        type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val-csv",          type=Path, default=DEFAULT_VAL_CSV)
    p.add_argument("--config-file",      type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--job-dir",          type=Path, default=DEFAULT_JOB_DIR)
    p.add_argument("--num-epochs",       type=int,  default=150,
                   help="Total training epochs. ~100-200 is reasonable for <10 cases.")
    p.add_argument("--pretrained-model", type=Path, default=DEFAULT_PRETRAINED,
                   help="Path to a .torch_model pretrained checkpoint to start from.")
    p.add_argument("--random-seed",      type=int,  default=42)
    p.add_argument("--val-fraction",     type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-generate train/val CSVs from --data-dir if they don't exist yet
    if not (args.train_csv.exists() and args.val_csv.exists()):
        from prepare_finetune_data import discover_cases, split_cases
        if not args.data_dir.exists():
            raise SystemExit(f"Data directory not found: {args.data_dir}\n"
                             "Pass --data-dir or pre-generate CSVs with prepare_finetune_data.py.")
        print(f"Discovering cases in {args.data_dir} …")
        csv_dir  = args.train_csv.parent
        csv_dir.mkdir(parents=True, exist_ok=True)
        cases    = discover_cases(args.data_dir, csv_dir)
        if not cases:
            raise SystemExit("No complete cases found (need CT-X, CT-X_stripped, CT-X_prediction).")
        train, val = split_cases(cases, args.val_fraction, args.random_seed)
        pd.DataFrame(train).to_csv(args.train_csv, index=False)
        pd.DataFrame(val).to_csv(args.val_csv,     index=False)
        print(f"  {len(train)} train / {len(val)} val cases → {csv_dir}/")

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
