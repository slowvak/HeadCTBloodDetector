#!/usr/bin/env python3
"""Scan ~/Desktop/CTHead for labeled exams and build train/val CSVs.

Expected file naming inside CTHead/:
  CT-X.nii.gz                  — original CT
  CT-X_stripped.nii.gz         — skull-stripped CT  (used as model input)
  CT-X_predictions.nii.gz      — single-label segmentation (0=bg,1=IPH,2=EAH,3=edema,4=IVH,5=SAH)

The skull-stripped image doubles as the sampling mask: its non-zero voxels
define the brain region where blast-ct draws training patches.

Outputs written to HeadCTBloodDetector/finetune_data/:
  train.csv
  val.csv
  summary.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def make_brain_mask(stripped_path: Path, mask_dir: Path) -> Path:
    """Create a binary brain mask from the skull-stripped CT if it doesn't exist.

    The stripped CT has 0 outside the brain and HU values inside.
    We binarise it: any non-zero voxel = brain = valid sampling location.
    """
    mask_path = mask_dir / (stripped_path.stem.replace('.nii', '') + '_brain_mask.nii.gz')
    if mask_path.exists():
        return mask_path

    img = sitk.ReadImage(str(stripped_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mask_arr = (arr != 0).astype(np.uint8)

    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(mask_path))
    print(f"  Created brain mask: {mask_path.name}")
    return mask_path


def discover_cases(data_dir: Path, mask_dir: Path) -> list[dict]:
    """Return one dict per complete case (image + stripped + prediction)."""
    # Find all originals (anything .nii.gz that is NOT a derived file)
    exclude_suffixes = ("_stripped.nii.gz", "_prediction.nii.gz", "_predictions.nii.gz",
                         "_mask.nii.gz", "_water.nii.gz", "_brain.nii.gz", "_filtered.nii.gz")
    originals = sorted(
        p for p in data_dir.glob("*.nii.gz")
        if not any(p.name.endswith(s) for s in exclude_suffixes)
    )

    cases = []
    missing = []
    for orig in originals:
        stem = orig.name[: -len(".nii.gz")]          # e.g. "CT-3"
        stripped = data_dir / f"{stem}_stripped.nii.gz"
        prediction = data_dir / f"{stem}_predictions.nii.gz"

        if not stripped.exists():
            missing.append(f"  MISSING stripped : {stripped.name}")
            continue
        if not prediction.exists():
            missing.append(f"  MISSING prediction: {prediction.name}")
            continue

        brain_mask = make_brain_mask(stripped, mask_dir)
        cases.append({
            "id": stem,
            "image": str(stripped),
            "target": str(prediction),
            "sampling_mask": str(brain_mask),  # binary: 1=brain, 0=outside
        })

    if missing:
        print("⚠️  Skipped incomplete cases:")
        for m in missing:
            print(m)

    return cases


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_cases(cases: list[dict], val_fraction: float, seed: int) -> tuple[list, list]:
    """Stratified-ish split: with <10 cases we just hold out at least 1."""
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction))
    # Never take more than half for validation
    n_val = min(n_val, len(shuffled) // 2) if len(shuffled) > 2 else 1
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "Desktop" / "HandCurated",
                        help="Directory containing CT-X / CT-X_stripped / CT-X_predictions files.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "finetune_data",
                        help="Where to write train.csv, val.csv, summary.txt.")
    parser.add_argument("--mask-dir", type=Path,
                        default=None,
                        help="Where to write generated brain masks (default: same as --out-dir).")
    parser.add_argument("--val-fraction", type=float, default=0.25,
                        help="Fraction of cases held out for validation (default 0.25).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir or args.out_dir
    mask_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.data_dir, mask_dir)
    if not cases:
        print("❌  No complete cases found. Check that each CT-X.nii.gz has a matching "
              "CT-X_stripped.nii.gz and CT-X_predictions.nii.gz in the same directory.")
        return

    train, val = split_cases(cases, args.val_fraction, args.seed)

    train_csv = args.out_dir / "train.csv"
    val_csv   = args.out_dir / "val.csv"
    pd.DataFrame(train).to_csv(train_csv, index=False)
    pd.DataFrame(val).to_csv(val_csv, index=False)

    summary = (
        f"Cases found   : {len(cases)}\n"
        f"Training set  : {len(train)} case(s)\n"
        f"Validation set: {len(val)} case(s)\n\n"
        f"Train IDs : {[c['id'] for c in train]}\n"
        f"Val IDs   : {[c['id'] for c in val]}\n\n"
        f"Train CSV : {train_csv}\n"
        f"Val CSV   : {val_csv}\n"
    )
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
