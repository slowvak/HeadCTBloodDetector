#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nnunetv2",
#   "torch",
#   "SimpleITK",
#   "pandas",
# ]
# ///
"""Train an nnUNet model on the CQ500 intracranial hemorrhage dataset.

This script:
  1. Converts the existing CQ500 data (from finetune_data/train.csv + val.csv)
     into nnUNet's expected dataset format.
  2. Runs nnUNetv2_plan_and_preprocess.
  3. Trains with nnUNetv2_train (CPU on macOS; MPS is not supported for 3-D convolutions).
  4. Optionally runs inference with nnUNetv2_predict.

nnUNet requires three environment variables:
  nnUNet_raw          — raw datasets
  nnUNet_preprocessed — preprocessed datasets written by plan_and_preprocess
  nnUNet_results      — trained model checkpoints

These default to subdirectories of --base-dir (default: ./nnunet_workspace).

Usage
-----
    uv run train_nnunet.py                        # prepare + train
    uv run train_nnunet.py --skip-prepare         # skip dataset conversion
    uv run train_nnunet.py --skip-train           # prepare only
    uv run train_nnunet.py --predict --input-dir /path/to/cases

macOS note
----------
3-D convolutions are not supported on MPS (Apple Silicon GPU), so training
always runs on CPU.  Use --config 2d to train a 2-D model, which is faster
on CPU than 3d_fullres.

Dataset
-------
  - 5 classes: 0=background, 1=IPH, 2=EDH, 3=SAH, 4=IVH
  - Single CT channel (skull-stripped, HU windowed to [-15, 100])
  - Labels sourced from *_seg.nii.gz (hand-curated) or *_prediction.nii.gz
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID   = 501                          # arbitrary ID ≥ 500 to avoid conflicts with MSD datasets
DATASET_NAME = f"Dataset{DATASET_ID:03d}_CQ500ICH"

CLASS_NAMES = {
    0: "background",
    1: "IPH",
    2: "EDH",
    3: "SAH",
    4: "IVH",
}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def set_nnunet_env(base_dir: Path) -> None:
    """Export nnUNet path variables if not already set."""
    defaults = {
        "nnUNet_raw":          str(base_dir / "raw"),
        "nnUNet_preprocessed": str(base_dir / "preprocessed"),
        "nnUNet_results":      str(base_dir / "results"),
    }
    for key, val in defaults.items():
        if key not in os.environ:
            os.environ[key] = val
            print(f"  {key} = {val}")
        else:
            print(f"  {key} = {os.environ[key]}  (from environment)")

    for key in defaults:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset conversion: CQ500 CSV → nnUNet raw format
# ---------------------------------------------------------------------------

def load_cases(train_csv: Path, val_csv: Path) -> tuple[list[dict], list[dict]]:
    train = pd.read_csv(train_csv).to_dict("records")
    val   = pd.read_csv(val_csv).to_dict("records")
    return train, val


def _copy_image(src: str | Path, dst: Path) -> None:
    """Copy a NIfTI image, rewriting via SimpleITK to ensure valid headers."""
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst))


def prepare_dataset(train_cases: list[dict], val_cases: list[dict],
                    dataset_dir: Path) -> None:
    """Write train/val cases into nnUNet raw dataset layout.

    Layout::

        DatasetXXX_Name/
          imagesTr/   CQ500_NNN_0000.nii.gz   (channel 0 = CT)
          labelsTr/   CQ500_NNN.nii.gz
          imagesTs/   CQ500_NNN_0000.nii.gz   (val used as 'test' for inference)
          labelsTs/   CQ500_NNN.nii.gz
          dataset.json
    """
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    def _copy_case(case: dict, img_dir: Path, lbl_dir: Path) -> str:
        case_id = case["id"]
        img_dst = img_dir / f"{case_id}_0000.nii.gz"
        lbl_dst = lbl_dir / f"{case_id}.nii.gz"
        if not img_dst.exists():
            print(f"  Copying image : {case_id}")
            _copy_image(case["image"], img_dst)
        if not lbl_dst.exists():
            print(f"  Copying label : {case_id}")
            _copy_image(case["target"], lbl_dst)
        return case_id

    train_ids = [_copy_case(c, images_tr, labels_tr) for c in train_cases]
    val_ids   = [_copy_case(c, images_ts, labels_ts) for c in val_cases]

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {name: idx for idx, name in CLASS_NAMES.items()},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\n  dataset.json written ({len(train_ids)} train, {len(val_ids)} val/test cases)")


# ---------------------------------------------------------------------------
# nnUNet commands
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess command, streaming output."""
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def plan_and_preprocess(dataset_id: int, config: str) -> None:
    """Run nnUNetv2_plan_and_preprocess for the given dataset."""
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
    ]
    # Only request the relevant configuration to save time
    if config == "2d":
        cmd += ["-c", "2d"]
    elif config == "3d_fullres":
        cmd += ["-c", "3d_fullres"]
    _run(cmd)


def _write_custom_trainer(epochs: int) -> str:
    """Write a minimal custom trainer into the nnunetv2 package; return class name."""
    import nnunetv2.training.nnUNetTrainer as _trainer_pkg
    trainer_name = f"nnUNetTrainer_{epochs}epochs"
    trainer_dir  = Path(_trainer_pkg.__file__).parent
    trainer_path = trainer_dir / f"{trainer_name}.py"
    trainer_path.write_text(
        f"from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n"
        f"class {trainer_name}(nnUNetTrainer):\n"
        f"    def __init__(self, plans, configuration, fold, dataset_json, device):\n"
        f"        super().__init__(plans, configuration, fold, dataset_json, device)\n"
        f"        self.num_epochs = {epochs}\n"
    )
    print(f"  Wrote trainer to: {trainer_path}")
    return trainer_name


def train(dataset_id: int, config: str, fold: int, epochs: int) -> None:
    """Run nnUNetv2_train on CPU."""
    trainer_flag = []
    if epochs != 1000:
        trainer_name = _write_custom_trainer(epochs)
        trainer_flag = ["-tr", trainer_name]
        print(f"  Using custom trainer: {trainer_name}.py ({epochs} epochs)")

    cmd = [
        "nnUNetv2_train",
        str(dataset_id), config, str(fold),
        "--npz",                # save softmax probabilities for ensembling
        "-device", "cpu",
        *trainer_flag,
    ]
    _run(cmd)


def find_best_config(dataset_id: int) -> None:
    """Print the best configuration after training all folds."""
    cmd = [
        "nnUNetv2_find_best_configuration",
        str(dataset_id),
    ]
    _run(cmd)


def predict(dataset_id: int, config: str, fold: int,
            input_dir: Path, output_dir: Path) -> None:
    """Run nnUNetv2_predict on a directory of images."""
    results_dir = Path(os.environ["nnUNet_results"])
    model_dir   = results_dir / DATASET_NAME / f"nnUNetTrainer__{config}__nnUNetPlans" / f"fold_{fold}"

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-chk", "checkpoint_best.pth",
        "-device", "cpu",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).parent / "nnunet_workspace",
        help="Root directory for nnUNet raw/preprocessed/results (default: ./nnunet_workspace)",
    )
    parser.add_argument(
        "--train-csv", type=Path, default=Path(__file__).parent / "finetune_data" / "train.csv",
        help="Training cases CSV produced by prepare_finetune_data.py",
    )
    parser.add_argument(
        "--val-csv", type=Path, default=Path(__file__).parent / "finetune_data" / "val.csv",
        help="Validation cases CSV produced by prepare_finetune_data.py",
    )
    parser.add_argument(
        "--config", choices=["2d", "3d_fullres", "3d_lowres"], default="2d",
        help=(
            "nnUNet configuration to train. Use '2d' on macOS (CPU) for speed; "
            "'3d_fullres' is more accurate but much slower on CPU. (default: 2d)"
        ),
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Cross-validation fold to train (0–4, or 'all'). Default: 0",
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs (default: 00, nnUNet's standard)",
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Skip dataset conversion (assumes it was already done)",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip nnUNetv2_plan_and_preprocess (assumes it was already done)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (prepare/preprocess only)",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Run inference after training",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory of images for inference (required with --predict). "
             "Images must be named <case_id>_0000.nii.gz.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for predictions (default: <base-dir>/predictions)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print("\nSetting nnUNet environment variables …")
    set_nnunet_env(args.base_dir)

    dataset_dir = Path(os.environ["nnUNet_raw"]) / DATASET_NAME

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    if not args.skip_prepare:
        if not args.train_csv.exists():
            raise FileNotFoundError(f"train.csv not found: {args.train_csv}")
        if not args.val_csv.exists():
            raise FileNotFoundError(f"val.csv not found: {args.val_csv}")

        print(f"\nLoading cases from CSVs …")
        train_cases, val_cases = load_cases(args.train_csv, args.val_csv)
        print(f"  {len(train_cases)} training, {len(val_cases)} validation cases")

        print(f"\nPreparing nnUNet dataset in {dataset_dir} …")
        prepare_dataset(train_cases, val_cases, dataset_dir)
    else:
        print(f"\n[skip-prepare] Using existing dataset in {dataset_dir}")

    # ------------------------------------------------------------------
    # Plan and preprocess
    # ------------------------------------------------------------------
    if not args.skip_preprocess and not args.skip_train:
        print(f"\nPlanning and preprocessing (config={args.config}) …")
        plan_and_preprocess(DATASET_ID, args.config)
    elif args.skip_preprocess:
        print("\n[skip-preprocess] Skipping plan_and_preprocess")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if not args.skip_train:
        print(f"\nTraining (config={args.config}, fold={args.fold}, epochs={args.epochs}) …")
        print("Note: training runs on CPU — this will be slow for 3d_fullres.")
        print("      Consider '--config 2d' or '--epochs 300' for a faster first run.\n")
        train(DATASET_ID, args.config, args.fold, args.epochs)
    else:
        print("\n[skip-train] Skipping training")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    if args.predict:
        if args.input_dir is None:
            parser.error("--predict requires --input-dir")
        output_dir = args.output_dir or (args.base_dir / "predictions")
        print(f"\nRunning inference: {args.input_dir} → {output_dir} …")
        predict(DATASET_ID, args.config, args.fold, args.input_dir, output_dir)
        print(f"\nPredictions written to: {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
