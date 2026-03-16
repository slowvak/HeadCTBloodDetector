#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "SimpleITK",
#   "pandas",
# ]
# ///
"""Convert CQ500 finetune data into nnUNet Dataset501_CQ500ICH format.

Reads train.csv and val.csv produced by prepare_finetune_data.py and writes:

  <nnUNet_raw>/Dataset501_CQ500ICH/
    imagesTr/   <id>_0000.nii.gz   ← skull-stripped CT (training)
    labelsTr/   <id>.nii.gz        ← segmentation label (training)
    imagesTs/   <id>_0000.nii.gz   ← skull-stripped CT (validation / test)
    labelsTs/   <id>.nii.gz        ← segmentation label (validation / test)
    dataset.json

The destination directory is resolved in order:
  1. --out-dir argument
  2. $nnUNet_raw/Dataset501_CQ500ICH
  3. ./nnunet_workspace/raw/Dataset501_CQ500ICH  (created automatically)

Images are hard-linked when source and destination are on the same filesystem
(instant, zero extra disk space); otherwise copied via SimpleITK so headers
are always valid.

Usage
-----
    uv run prepare_nnunet_data.py
    uv run prepare_nnunet_data.py --train-csv finetune_data/train.csv \\
                                  --val-csv   finetune_data/val.csv   \\
                                  --out-dir   /path/to/nnUNet_raw/Dataset501_CQ500ICH
    uv run prepare_nnunet_data.py --dry-run   # show what would be copied
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID   = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_CQ500ICH"

CLASS_NAMES = {
    0: "background",
    1: "IPH",
    2: "EDH",
    3: "SAH",
    4: "IVH",
}


# ---------------------------------------------------------------------------
# File transfer helpers
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path, dry_run: bool) -> str:
    """Hard-link if same device, otherwise copy via SimpleITK. Returns action string."""
    if dry_run:
        return "would link/copy"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        img = sitk.ReadImage(str(src))
        sitk.WriteImage(img, str(dst))
        return "copied"


def transfer_case(case_id: str,
                  image_src: Path, label_src: Path,
                  img_dir: Path, lbl_dir: Path,
                  dry_run: bool,
                  overwrite: bool) -> None:
    img_dst = img_dir / f"{case_id}_0000.nii.gz"
    lbl_dst = lbl_dir / f"{case_id}.nii.gz"

    for src, dst, kind in [
        (image_src, img_dst, "image"),
        (label_src, lbl_dst, "label"),
    ]:
        if dst.exists() and not overwrite:
            print(f"  skip  {kind:6s}: {dst.name}  (already exists)")
            continue
        if dst.exists() and overwrite and not dry_run:
            dst.unlink()
        action = _link_or_copy(src, dst, dry_run)
        print(f"  {action:6s} {kind:6s}: {src.name}  →  {dst.name}")


# ---------------------------------------------------------------------------
# dataset.json
# ---------------------------------------------------------------------------

def write_dataset_json(dataset_dir: Path, n_train: int, dry_run: bool) -> None:
    payload = {
        "channel_names": {"0": "CT"},
        "labels": {name: idx for idx, name in CLASS_NAMES.items()},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    dst = dataset_dir / "dataset.json"
    if dry_run:
        print(f"\n  would write: {dst}")
        print(f"  {json.dumps(payload, indent=4)}")
    else:
        dst.write_text(json.dumps(payload, indent=2))
        print(f"\n  wrote: {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-csv", type=Path,
        default=here / "finetune_data" / "train.csv",
        help="Training CSV from prepare_finetune_data.py (default: finetune_data/train.csv)",
    )
    parser.add_argument(
        "--val-csv", type=Path,
        default=here / "finetune_data" / "val.csv",
        help="Validation CSV from prepare_finetune_data.py (default: finetune_data/val.csv)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Destination dataset directory "
            f"(default: $nnUNet_raw/{DATASET_NAME} or ./nnunet_workspace/raw/{DATASET_NAME})"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-copy files even if they already exist in the destination",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.out_dir:
        dataset_dir = args.out_dir
    elif "nnUNet_raw" in os.environ:
        dataset_dir = Path(os.environ["nnUNet_raw"]) / DATASET_NAME
    else:
        dataset_dir = here / "nnunet_workspace" / "raw" / DATASET_NAME
        os.environ.setdefault("nnUNet_raw", str(dataset_dir.parent))

    print(f"\nDataset directory: {dataset_dir}")

    # Load CSVs
    if not args.train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {args.train_csv}")
    if not args.val_csv.exists():
        raise FileNotFoundError(f"val.csv not found: {args.val_csv}")

    train_cases = pd.read_csv(args.train_csv).to_dict("records")
    val_cases   = pd.read_csv(args.val_csv).to_dict("records")
    print(f"Cases: {len(train_cases)} training, {len(val_cases)} validation\n")

    # Create subdirectories
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    if not args.dry_run:
        for d in (images_tr, labels_tr, images_ts, labels_ts):
            d.mkdir(parents=True, exist_ok=True)

    # Training cases
    print("── Training cases ──────────────────────────────")
    missing = []
    for case in train_cases:
        img_src = Path(case["image"])
        lbl_src = Path(case["target"])
        for p in (img_src, lbl_src):
            if not p.exists():
                missing.append(str(p))
        if not missing:
            transfer_case(case["id"], img_src, lbl_src,
                          images_tr, labels_tr, args.dry_run, args.overwrite)

    # Validation cases → imagesTs / labelsTs
    print("\n── Validation cases (→ imagesTs / labelsTs) ────")
    for case in val_cases:
        img_src = Path(case["image"])
        lbl_src = Path(case["target"])
        for p in (img_src, lbl_src):
            if not p.exists():
                missing.append(str(p))
        if not missing:
            transfer_case(case["id"], img_src, lbl_src,
                          images_ts, labels_ts, args.dry_run, args.overwrite)

    if missing:
        print(f"\n⚠️  {len(missing)} missing source file(s):")
        for p in missing:
            print(f"  {p}")

    # dataset.json
    print("\n── dataset.json ────────────────────────────────")
    write_dataset_json(dataset_dir, len(train_cases), args.dry_run)

    n_ok_tr = len(list(images_tr.glob("*.nii.gz"))) if not args.dry_run else "?"
    n_ok_ts = len(list(images_ts.glob("*.nii.gz"))) if not args.dry_run else "?"
    print(f"\nDone.  imagesTr={n_ok_tr}  imagesTs={n_ok_ts}")
    print(f"\nNext step:\n  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity")


if __name__ == "__main__":
    main()
