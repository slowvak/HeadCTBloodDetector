#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["nibabel", "numpy"]
# ///
"""Fix NIfTI direction cosines in-place for every .nii.gz in a directory.

Strategy
--------
1. Extract the rotation matrix from the affine (normalize each column by voxel size).
2. Snap values within 0.1 of 0 → 0, within 0.1 of ±1 → ±1.
3. Run SVD to find the nearest exactly-orthonormal matrix (guarantees ITK compliance).
4. Rebuild the affine with the corrected rotation and original voxel sizes / translation.
5. Save back in-place only when the affine actually changed.

Usage
-----
    uv run header_cleanup.py /path/to/directory
    uv run header_cleanup.py /path/to/directory --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


SNAP_THRESHOLD = 0.1


def nearest_orthonormal(mat: np.ndarray) -> np.ndarray:
    """Return the nearest orthonormal matrix to *mat* via SVD."""
    U, _, Vt = np.linalg.svd(mat)
    return U @ Vt


def fix_affine(affine: np.ndarray) -> np.ndarray:
    """Return a corrected affine with an exactly orthonormal rotation part."""
    new_affine = affine.copy()
    rot_scale = affine[:3, :3]

    # Voxel sizes = column norms
    voxel_sizes = np.linalg.norm(rot_scale, axis=0)
    voxel_sizes[voxel_sizes == 0] = 1.0

    # Pure rotation matrix (unit columns)
    rot = rot_scale / voxel_sizes

    # Snap near-0 and near-±1 values
    snapped = rot.copy()
    snapped[np.abs(snapped) < SNAP_THRESHOLD] = 0.0
    close_to_one = np.abs(np.abs(snapped) - 1.0) < SNAP_THRESHOLD
    snapped[close_to_one] = np.sign(snapped[close_to_one])

    # SVD orthonormalization — guaranteed exact
    ortho = nearest_orthonormal(snapped)

    new_affine[:3, :3] = ortho * voxel_sizes
    return new_affine


def fix_file(path: Path, dry_run: bool) -> str:
    """Fix the affine of one NIfTI file.  Returns 'fixed', 'ok', or 'error: ...'"""
    try:
        img = nib.load(str(path))
    except Exception as e:
        return f"error loading: {e}"

    original_affine = img.affine
    new_affine = fix_affine(original_affine)

    if np.allclose(original_affine, new_affine, atol=1e-6):
        return "ok"

    if dry_run:
        return "would fix"

    try:
        data = np.asarray(img.dataobj)
        new_img = nib.Nifti1Image(data, affine=new_affine, header=img.header)
        new_img.header.set_sform(new_affine, code=int(img.header['sform_code']))
        new_img.header.set_qform(new_affine, code=int(img.header['qform_code']))
        nib.save(new_img, str(path))
    except Exception as e:
        return f"error saving: {e}"

    return "fixed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing .nii.gz files to fix")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Not a directory: {args.input_dir}")

    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        sys.exit(f"No .nii.gz files found in {args.input_dir}")

    print(f"Scanning {len(files)} file(s) in {args.input_dir} …\n")

    counts: dict[str, int] = {"fixed": 0, "ok": 0, "would fix": 0, "error": 0}
    for path in files:
        status = fix_file(path, args.dry_run)
        key = "error" if status.startswith("error") else status
        counts[key] += 1
        if status != "ok":
            print(f"  {status:12s}  {path.name}")

    print(f"\nDone.  ok={counts['ok']}  "
          + (f"would fix={counts['would fix']}  " if args.dry_run
             else f"fixed={counts['fixed']}  ")
          + f"error={counts['error']}")


if __name__ == "__main__":
    main()
