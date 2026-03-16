#!/usr/bin/env python3
"""Clean up the HandCurated directory.

1. Delete any file whose name contains 'CT' (old hyphenated format, e.g.
   CQ500-CT-0_prediction.nii.gz).

2. For every *_prediction.nii.gz that remains, remap labels in-place:
     - class 3 (edema) → 0  (remove)
     - class 5 (SAH)   → 3  (consolidate into the SAH slot)

Usage
-----
    uv run cleanup.py
    uv run cleanup.py --data-dir /path/to/HandCurated
    uv run cleanup.py --dry-run          # show what would happen without changing files
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_DATA_DIR = Path("/Volumes/OWC Express 1M2/HandCurated")


def remove_old_format_files(data_dir: Path, dry_run: bool) -> None:
    """Delete any .nii.gz file whose name contains 'CT' (old CQ500-CT-N format)."""
    to_delete = sorted(p for p in data_dir.glob("*.nii.gz") if "CT" in p.name)
    if not to_delete:
        print("  No old-format files found.")
        return
    for p in to_delete:
        print(f"  {'[dry-run] ' if dry_run else ''}Delete: {p.name}")
        if not dry_run:
            p.unlink()
    print(f"  {'Would delete' if dry_run else 'Deleted'} {len(to_delete)} file(s).")


def remap_predictions(data_dir: Path, dry_run: bool) -> None:
    """Remap labels in every *_prediction.nii.gz: class 3 → 0, class 5 → 3."""
    pred_files = sorted(data_dir.glob("*_prediction.nii.gz"))
    if not pred_files:
        print("  No *_prediction.nii.gz files found.")
        return

    changed = 0
    for pred_path in pred_files:
        img = sitk.ReadImage(str(pred_path))
        arr = sitk.GetArrayViewFromImage(img)

        has3 = bool((arr == 3).any())
        has5 = bool((arr == 5).any())

        if not has3 and not has5:
            continue  # already clean

        print(f"  {'[dry-run] ' if dry_run else ''}Remap: {pred_path.name}"
              f"  (class3={'yes' if has3 else 'no'}, class5={'yes' if has5 else 'no'})")

        if not dry_run:
            out = sitk.GetArrayFromImage(img).astype(np.int16)
            out[out == 5] = -1   # park SAH temporarily to avoid collision
            out[out == 3] = 0    # remove edema
            out[out == -1] = 3   # SAH → class 3
            new_img = sitk.GetImageFromArray(out)
            new_img.CopyInformation(img)
            sitk.WriteImage(new_img, str(pred_path))

        changed += 1

    print(f"  {'Would remap' if dry_run else 'Remapped'} {changed} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Directory to clean (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without modifying any files.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Directory not found: {args.data_dir}")

    print(f"\nStep 1 — Remove old-format files in {args.data_dir} …")
    remove_old_format_files(args.data_dir, args.dry_run)

    print(f"\nStep 2 — Remap prediction labels in {args.data_dir} …")
    remap_predictions(args.data_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
