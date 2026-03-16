#!/usr/bin/env python3
"""Scan a data directory for labeled exams, preprocess predictions, and build train/val CSVs.

Expected file naming inside --data-dir:
  CQ500_NNN_prediction.nii.gz        — single-label segmentation (required)
  CQ500_NNN_stripped.nii.gz          — skull-stripped CT (optional; falls back to --source-dir)
  CQ500_NNN_stripped_brain_mask.nii.gz — brain mask (optional; generated if absent)

If a stripped image is not found in --data-dir, the script looks for it in --source-dir
using the old naming convention CQ500-CT-N_stripped.nii.gz (no leading zeros, hyphenated).

Preprocessing applied in-place to each *_prediction.nii.gz:
  1. Remove class 3 (set to 0).
  2. Remap class 5 → 3 (SAH).  Files containing class 5 are printed before remapping.

Outputs written to --out-dir:
  train.csv
  val.csv
  summary.txt
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from run_synthstrip import ensure_synthstrip, run_synthstrip as _run_synthstrip


# ---------------------------------------------------------------------------
# File renaming
# ---------------------------------------------------------------------------

_RENAME_RE = re.compile(r'^CQ500-CT-(\d+)(.*\.nii\.gz)$')


def rename_cq500_files(data_dir: Path) -> None:
    """Copy CQ500-CT-N[N]*.nii.gz → CQ500_NNN*.nii.gz in data_dir.

    The original old-format file is left in place; a new-format copy is
    created alongside it (skipped if the destination already exists).

    Examples:
      CQ500-CT-5.nii.gz            → CQ500_005.nii.gz
      CQ500-CT-53_stripped.nii.gz  → CQ500_053_stripped.nii.gz
      CQ500-CT-7_prediction.nii.gz → CQ500_007_prediction.nii.gz
    """
    for path in sorted(data_dir.glob("CQ500-CT-*.nii.gz")):
        m = _RENAME_RE.match(path.name)
        if not m:
            continue
        new_name = f"CQ500_{int(m.group(1)):03d}{m.group(2)}"
        new_path = path.parent / new_name
        if new_path.exists():
            continue
        shutil.copy2(str(path), str(new_path))
        print(f"  Copied: {path.name}  →  {new_name}")


# ---------------------------------------------------------------------------
# Filename cleanup
# ---------------------------------------------------------------------------

def clean_filenames(data_dir: Path) -> None:
    """Clean up filenames in data_dir:
      1. Remove ' copy' from any .nii.gz filename.
         If the destination already exists, delete it before renaming.
      2. Rename *_stripped_segmentation.nii.gz → *_prediction.nii.gz.
    """
    # Step 1: remove ' copy'
    for path in sorted(data_dir.glob("*.nii.gz")):
        if " copy" not in path.name:
            continue
        new_name = path.name.replace(" copy", "")
        new_path = path.parent / new_name
        if new_path.exists():
            new_path.unlink()
        path.rename(new_path)
        print(f"  Renamed: {path.name}  →  {new_name}")

    # Step 2: rename *_stripped_segmentation.nii.gz → *_prediction.nii.gz
    for path in sorted(data_dir.glob("*_stripped_segmentation.nii.gz")):
        new_name = path.name.replace("_stripped_segmentation.nii.gz", "_prediction.nii.gz")
        new_path = path.parent / new_name
        if new_path.exists():
            new_path.unlink()
        path.rename(new_path)
        print(f"  Renamed: {path.name}  →  {new_name}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def make_brain_mask(stripped_path: Path, mask_dir: Path) -> Path:
    """Create a binary brain mask from the skull-stripped CT if it doesn't exist."""
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


def _find_original(stem: str, *search_dirs: Path | None) -> Path | None:
    """Return the original (non-stripped) CT for *stem*, or None."""
    for d in search_dirs:
        if d is None:
            continue
        candidate = d / f"{stem}.nii.gz"
        if candidate.exists():
            return candidate
    return None


def _strip(original: Path, stripped_out: Path, mask_out: Path,
           synthstrip_cmd: str) -> bool:
    """Run synthstrip on *original*, writing outputs to their paths. Returns True on success."""
    print(f"  Running synthstrip: {original.name} → {stripped_out.name}")
    result = _run_synthstrip(
        input_path=original,
        output_path=stripped_out,
        mask_path=mask_out,
        synthstrip_cmd=synthstrip_cmd,
    )
    return result.returncode == 0


def discover_cases(data_dir: Path, mask_dir: Path,
                   source_dir: Path | None = None,
                   old_dir: Path | None = None,
                   synthstrip_cmd: str = "synthstrip-docker") -> list[dict]:
    """Return one dict per labeled case found in data_dir.

    Iterates over CQ500_NNN_prediction.nii.gz files as the authoritative list.
    For each case, the stripped image is resolved as follows:
      1. data_dir/CQ500_NNN_stripped.nii.gz
      2. source_dir/CQ500_NNN_stripped.nii.gz
      3. old_dir/CQ500_NNN_stripped.nii.gz
      4. If none exist but the original CT is found (any dir), run synthstrip.
    The brain mask is used from data_dir if present, otherwise generated.
    """
    pred_files = sorted(data_dir.glob("CQ500_*_prediction.nii.gz"))
    if not pred_files:
        pred_files = sorted(data_dir.glob("*_prediction.nii.gz"))

    # Only resolve synthstrip once, and only if we might need it
    _synthstrip_resolved: str | None = None

    cases = []
    missing = []

    for pred_path in pred_files:
        stem = pred_path.name[: -len("_prediction.nii.gz")]  # e.g. CQ500_053

        # Resolve stripped image — check all directories in order
        stripped = None
        for d in (data_dir, source_dir, old_dir):
            if d is None:
                continue
            candidate = d / f"{stem}_stripped.nii.gz"
            if candidate.exists():
                stripped = candidate
                break

        # Auto-strip if original CT exists
        if stripped is None:
            original = _find_original(stem, data_dir, source_dir, old_dir)
            if original:
                stripped_out = original.parent / f"{stem}_stripped.nii.gz"
                mask_out     = original.parent / f"{stem}_mask.nii.gz"
                if _synthstrip_resolved is None:
                    _synthstrip_resolved = ensure_synthstrip(synthstrip_cmd)
                if _strip(original, stripped_out, mask_out, _synthstrip_resolved):
                    stripped = stripped_out
                else:
                    missing.append(f"  SYNTHSTRIP FAILED: {stem}")
                    continue
            else:
                missing.append(f"  MISSING stripped + original: {stem}")
                continue

        # Brain mask: use existing local one or generate
        local_mask = data_dir / f"{stem}_stripped_brain_mask.nii.gz"
        if local_mask.exists():
            brain_mask = local_mask
        else:
            brain_mask = make_brain_mask(stripped, mask_dir)

        # Prefer *_seg.nii.gz as target (already remapped) if available
        target = pred_path
        for d in (data_dir, source_dir, old_dir):
            if d is None:
                continue
            seg_candidate = d / f"{stem}_seg.nii.gz"
            if seg_candidate.exists():
                target = seg_candidate
                break

        cases.append({
            "id": stem,
            "image": str(stripped),
            "target": str(target),
            "sampling_mask": str(brain_mask),
        })

    if missing:
        print(f"⚠️  Skipped {len(missing)} case(s):")
        for msg in missing:
            print(msg)

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
                        default="/Volumes/OWC Express 1M2/HandCurated",
                        help="Directory containing prediction and (optionally) stripped files.")
    parser.add_argument("--source-dir", type=Path,
                        default="/Volumes/OWC Express 1M2/CQ500_NII",
                        help="Fallback directory for stripped CT images (old CQ500-CT-N naming).")
    parser.add_argument("--old-dir", type=Path,
                        default="/Volumes/OWC Express 1M2/HandCurated_old",
                        help="Secondary fallback directory for stripped/raw CTs (e.g. HandCurated_old).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "finetune_data",
                        help="Where to write train.csv, val.csv, summary.txt.")
    parser.add_argument("--mask-dir", type=Path,
                        default=None,
                        help="Where to write generated brain masks (default: same as --out-dir).")
    parser.add_argument("--val-fraction", type=float, default=0.25,
                        help="Fraction of cases held out for validation (default 0.25).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthstrip-cmd", type=str, default="synthstrip-docker",
                        help="Path to synthstrip-docker script (used if stripped image is missing).")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    source_dir = args.source_dir if args.source_dir and args.source_dir.exists() else None
    if args.source_dir and not source_dir:
        print(f"⚠️  --source-dir not found, ignored: {args.source_dir}")

    old_dir = args.old_dir if args.old_dir and args.old_dir.exists() else None
    if args.old_dir and not old_dir:
        print(f"⚠️  --old-dir not found, ignored: {args.old_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir or args.out_dir
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRenaming CQ500-CT-N files in {args.data_dir} …")
    rename_cq500_files(args.data_dir)
    if source_dir:
        print(f"\nRenaming CQ500-CT-N files in {source_dir} …")
        rename_cq500_files(source_dir)

    print(f"\nCleaning filenames in {args.data_dir} …")
    clean_filenames(args.data_dir)

    print(f"\nDiscovering cases (fallbacks: {source_dir or 'none'}, {old_dir or 'none'}) …")
    cases = discover_cases(args.data_dir, mask_dir, source_dir, old_dir,
                           synthstrip_cmd=args.synthstrip_cmd)
    if not cases:
        print("❌  No complete cases found.")
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
