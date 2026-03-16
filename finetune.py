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
    """Suppress broken blast-ct metrics (our OwnMetricsHook handles reporting)."""
    return ''


def _patched_increment(self, model_state):
    """No-op: bypass blast-ct confusion matrix (crashes when labels contain unexpected classes)."""
    pass


def _patched_save_and_reset(self):
    """Skip calc_* calls that divide by the all-zero confusion matrix and produce RuntimeWarnings."""
    import torch as _torch
    if isinstance(self.running_value, _torch.Tensor):
        self.value = self.running_value.clone().cpu().numpy()
        self.running_value[:] = 0
    else:
        self.value = self.running_value.copy()
        self.running_value[:] = 0


def _patched_log_to_tensorboard(self, epoch, writer, tag):
    pass


_metrics.SegmentationMetrics.report = _patched_report
_metrics.SegmentationMetrics.increment = _patched_increment
_metrics.SegmentationMetrics.save_and_reset = _patched_save_and_reset
_metrics.SegmentationMetrics.log_to_tensorboard = _patched_log_to_tensorboard


import time as _time
from blast_ct.trainer.hooks import Hook


class OwnMetricsHook(Hook):
    """Our own confusion-matrix tracker — bypasses the broken blast-ct one."""

    def __init__(self, class_names, num_classes=6):
        super().__init__()
        self.class_names = list(class_names)
        self.class_names[0] = 'Foreground'
        self.n = num_classes
        self.cm = None
        self.epoch_time = 0

    def before_epoch(self):
        self.cm = torch.zeros(self.n, self.n, dtype=torch.int64)
        self.epoch_time = _time.time()

    def after_batch(self):
        state = self.model_trainer.current_state
        t = state.get('target')
        p = state.get('pred')
        if t is None or p is None:
            return
        t_flat = t.flatten().long()
        p_flat = p.flatten().long()
        # Encode (true, pred) pairs as single index, then bincount
        indices = t_flat * self.n + p_flat
        counts = torch.bincount(indices, minlength=self.n * self.n)
        self.cm += counts.reshape(self.n, self.n)

    def after_epoch(self):
        epoch = self.model_trainer.current_state['epoch']
        num_epochs = self.model_trainer.current_state['num_epochs']
        elapsed = _time.time() - self.epoch_time
        cm = self.cm.numpy()
        total_all = int(_np.sum(cm))
        correct_all = int(_np.sum(_np.diag(cm)))
        msg = f"Training epoch {epoch}/{num_epochs-1} ({elapsed:.0f}s)  "
        msg += f"OVERALL: {correct_all}/{total_all} ({100.*correct_all/max(total_all,1):.1f}%)\n"
        for i, name in enumerate(self.class_names):
            total_i = int(_np.sum(cm[i, :]))
            correct_i = int(cm[i, i])
            msg += f"  {name.upper().ljust(16)}: {correct_i}/{total_i} ({100.*correct_i/max(total_i,1):.1f}%)\n"
        print(msg)


class BestModelSaverHook(Hook):
    """Saves model_best.torch_model whenever mean validation loss improves.

    Runs a second (no-grad) forward pass over the validation set on every
    eval epoch so it can track mean loss independently of the blast-ct hooks.
    """

    def __init__(self, valid_loader, eval_every: int = 10):
        super().__init__()
        self.valid_loader = valid_loader
        self.eval_every   = eval_every
        self.best_loss    = float("inf")

    def after_epoch(self):
        state      = self.model_trainer.current_state
        epoch      = state["epoch"]
        num_epochs = state["num_epochs"]
        is_last    = epoch == num_epochs - 1

        if not (epoch % self.eval_every == 0 or is_last) or epoch == 0:
            return

        total_loss, n = 0.0, 0
        for batch_state in self.model_trainer.step(epoch, self.valid_loader, is_training=False):
            total_loss += float(batch_state["loss"])
            n += 1

        if n == 0:
            return

        mean_loss = total_loss / n
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            import os as _os
            saved_dir = _os.path.join(self.model_trainer.job_dir, "saved_models")
            _os.makedirs(saved_dir, exist_ok=True)
            best_path = _os.path.join(saved_dir, "model_best.torch_model")
            torch.save(self.model_trainer.model.state_dict(), best_path)
            print(f"  *** New best model (epoch {epoch}, val_loss={mean_loss:.4f}) → {best_path}")
        else:
            print(f"      Val loss {mean_loss:.4f}  (best={self.best_loss:.4f})")


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
    Class 0 (background) is included at half weight (0.5).
    With 5 classes the effective denominator is 6.5 (0.5 + 3×1 + 1×2).
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
        # Initialise from probs so the grad_fn is always present even when
        # every voxel in the batch is class 0 (which is excluded from the loss).
        focal_total = probs.sum() * 0.0
        dice_total  = probs.sum() * 0.0
        denom       = logits.new_zeros(1)

        for c in range(self.n_classes):
            mask = target == c
            if not mask.any():
                continue
            w_c = self.class_weights[c] * (0.5 if c == 0 else 1.0)
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
            if src.name.endswith(".nii.gz"):
                dst = filtered_dir / (src.name[:-len(".nii.gz")] + "_filtered.nii.gz")
            else:
                dst = filtered_dir / (src.name[:-len(".nii")] + "_filtered.nii")
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

    # Pre-process: apply 5×5 median filter to images only (not targets — they're discrete labels)
    # (label remapping 3→0, 5→3 is done in-place by prepare_finetune_data.py)
    filtered_dir = job_dir / "filtered_data"
    print(f"\nPre-filtering training data (5×5 median, cached in {filtered_dir})…")
    filtered_train_csv = build_filtered_csv(train_csv, filtered_dir, ["image"])
    filtered_val_csv   = build_filtered_csv(val_csv,   filtered_dir, ["image"])

    # Data loaders
    train_loader = get_train_loader(config, model, str(filtered_train_csv), use_cuda=False)
    valid_loader = get_test_loader(config, model, str(filtered_val_csv),    use_cuda=False)
    test_loader  = get_test_loader( config, model, str(filtered_val_csv),   use_cuda=False)

    # Optimiser + scheduler
    lr_scheduler = get_optimizer(config, model)

    # Loss: focal + soft Dice, per-class balanced, SAH weighted 2x
    criterion = FocalDiceLoss(config['data']['class_names'])
    print("Loss: focal (γ=2) + soft Dice  (SAH weight = 2.0, class 0 weight = 0.5)")
    eval_every   = config["valid"]["eval_every"]
    hooks        = get_training_hooks(str(job_dir), config, device, valid_loader, test_loader)
    hooks.append(OwnMetricsHook(config['data']['class_names'], config['data']['num_classes']))
    hooks.append(BestModelSaverHook(valid_loader, eval_every))

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
