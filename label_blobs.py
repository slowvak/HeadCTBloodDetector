#!/usr/bin/env python3
"""
label_blobs.py
──────────────
Read a NIfTI image, apply a 3×3×3 median filter, threshold at 75 HU,
run 26-connected 3-D connected-component analysis, discard objects
smaller than 1 cc, and write a label map where each remaining object's
value equals its size-rank (1 = largest, 2 = next, …).

Usage
─────
    python label_blobs.py -i input.nii.gz
    python label_blobs.py -i input.nii.gz -o labels.nii.gz
    python label_blobs.py -i input.nii.gz --threshold 80 --min-cc 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter, label, sum as ndsum, center_of_mass


# 26-connected structuring element (3×3×3 cube of ones)
STRUCT_26 = np.ones((3, 3, 3), dtype=np.int32)


def volume_per_voxel_cc(affine: np.ndarray) -> float:
    """Return the volume of one voxel in cubic centimetres."""
    voxel_sizes = np.abs(np.diag(affine[:3, :3]))  # mm per voxel
    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    return voxel_vol_mm3 / 1000.0  # 1 cc = 1000 mm³


def process(
    data: np.ndarray,
    affine: np.ndarray,
    min_threshold: float = 75.0,
    max_threshold: float = 75.0,
    min_cc: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    data : 3-D ndarray
        Raw voxel intensities (e.g. HU from a CT).
    affine : 4×4 ndarray
        NIfTI affine (used to compute voxel volume).
    threshold : float
        Intensity threshold applied after the median filter.
    min_cc : float
        Minimum object volume in cubic centimetres to keep.

    Returns
    -------
    ranked_labels : 3-D ndarray (int32)
        Size-ranked label map (1 = largest, 0 = background).
    filtered : 3-D ndarray (float32)
        Median-filtered image (used later for mean density).
    binary : 3-D ndarray (int32)
        Binary mask after threshold.
    """
    # ── 1. Median filter (3×3×3) ─────────────────────────────────────────
    filtered = median_filter(data.astype(np.float32), size=3)

    # ── 2. Threshold ─────────────────────────────────────────────────────
    binary = ((filtered >= min_threshold) & (filtered <= max_threshold)).astype(np.int32)

    # ── 3. Connected-component labelling (26-connected) ──────────────────
    labelled, n_objects = label(binary, structure=STRUCT_26)
    print(f"  Components found after threshold: {n_objects}")

    if n_objects == 0:
        empty = np.zeros_like(data, dtype=np.int32)
        return empty, filtered, binary

    component_ids = np.arange(1, n_objects + 1)

    # ── 4. Measure each component's size (in voxels & cc) ────────────────
    vox_cc = volume_per_voxel_cc(affine)
    sizes_vox = ndsum(binary, labelled, component_ids).astype(int)
    sizes_cc = sizes_vox * vox_cc

    # ── 5. Remove objects < min_cc ───────────────────────────────────────
    keep_mask = sizes_cc >= min_cc
    kept_ids = component_ids[keep_mask]
    kept_sizes = sizes_vox[keep_mask]
    print(f"  Components ≥ {min_cc} cc: {len(kept_ids)}  "
          f"(removed {n_objects - len(kept_ids)})")

    if len(kept_ids) == 0:
        empty = np.zeros_like(data, dtype=np.int32)
        return empty, filtered, binary

    # ── 6. Sort remaining by size (descending) ───────────────────────────
    sort_order = np.argsort(-kept_sizes)   # largest first
    kept_ids = kept_ids[sort_order]
    kept_sizes = kept_sizes[sort_order]

    # ── 7. Build rank-labelled output ────────────────────────────────────
    output = np.zeros_like(labelled, dtype=np.int32)
    for rank, comp_id in enumerate(kept_ids, start=1):
        output[labelled == comp_id] = rank

    # Print summary table
    print(f"\n  {'Rank':>4}  {'Voxels':>8}  {'Volume (cc)':>11}")
    print(f"  {'────':>4}  {'──────':>8}  {'───────────':>11}")
    for rank, (cid, sz) in enumerate(zip(kept_ids, kept_sizes), start=1):
        print(f"  {rank:>4}  {sz:>8}  {sz * vox_cc:>11.2f}")

    return output, filtered, binary


# ---------------------------------------------------------------------------
#  Per-object statistics
# ---------------------------------------------------------------------------

def _face_areas(affine: np.ndarray) -> tuple[float, float, float]:
    """Return the area (mm²) of a single voxel face perpendicular to each axis.

    Returns (area_yz, area_xz, area_xy) corresponding to faces whose normal
    is along the X, Y, Z axes respectively.
    """
    dx, dy, dz = np.abs(np.diag(affine[:3, :3]))
    return (dy * dz, dx * dz, dx * dy)


def compute_blob_stats(
    ranked_labels: np.ndarray,
    original_data: np.ndarray,
    affine: np.ndarray,
    threshold: float,
) -> list[dict]:
    """
    Compute per-object statistics for every labelled blob.

    Parameters
    ----------
    ranked_labels : 3-D int32 array
        Output of `process()` — 1 = largest blob, 2 = next, etc.
    original_data : 3-D array
        The *original* (unfiltered) input image (for mean density and
        for distinguishing brain vs non-brain).
    affine : 4×4 array
        NIfTI affine.
    threshold : float
        The same intensity threshold used in `process()`.

    Returns
    -------
    rows : list of dict
        One dict per object, keyed by CSV column names.
    """
    dx, dy, dz = np.abs(np.diag(affine[:3, :3]))  # mm
    voxel_vol = dx * dy * dz                       # mm³
    area_yz, area_xz, area_xy = _face_areas(affine)
    shape = np.array(ranked_labels.shape, dtype=float)  # (D, H, W)

    n_labels = ranked_labels.max()
    if n_labels == 0:
        return []

    rows: list[dict] = []

    for idx in range(1, n_labels + 1):
        obj_mask = (ranked_labels == idx)
        n_vox = int(obj_mask.sum())
        volume_mm3 = n_vox * voxel_vol

        # -- Mean density (from the original, unfiltered image) ------------
        mean_density = float(original_data[obj_mask].mean())

        # -- Center of mass (fractional position) --------------------------
        com = center_of_mass(obj_mask.astype(np.float64))
        frac_x = com[2] / shape[2]   # axis-2 = columns = X
        frac_y = com[1] / shape[1]   # axis-1 = rows    = Y
        frac_z = com[0] / shape[0]   # axis-0 = slices  = Z

        # -- Surface area & contact areas ----------------------------------
        # We examine the 6-connected face-neighbours of every object voxel.
        # For each face that borders a non-object voxel (or the image edge),
        # we accumulate:
        #   total_surface   – all exposed faces
        #   contact_nonbrain – faces adjacent to voxels == 0 in original
        #   contact_brain    – faces adjacent to brain (0 < val < threshold)
        #
        # Face area depends on which axis the face is perpendicular to.

        total_surface = 0.0
        contact_nonbrain = 0.0
        contact_brain = 0.0

        # axis, direction (+1 / -1), face area for that axis
        neighbors = [
            (0, -1, area_xy),  # Z-  face ⊥ Z → area = dx*dy
            (0, +1, area_xy),  # Z+
            (1, -1, area_xz),  # Y-  face ⊥ Y → area = dx*dz
            (1, +1, area_xz),  # Y+
            (2, -1, area_yz),  # X-  face ⊥ X → area = dy*dz
            (2, +1, area_yz),  # X+
        ]

        for axis, direction, face_area in neighbors:
            # Shift the object mask in the given direction
            shifted_mask = np.roll(obj_mask, shift=direction, axis=axis)
            # Voxels at the rolled-over boundary are outside the image
            boundary_slice = [slice(None)] * 3
            if direction == -1:
                boundary_slice[axis] = slice(-1, None)   # last slice
            else:
                boundary_slice[axis] = slice(0, 1)       # first slice
            shifted_mask[tuple(boundary_slice)] = False

            # Exposed faces: object voxels whose neighbour is NOT object
            exposed = obj_mask & ~shifted_mask

            n_exposed = int(exposed.sum())
            total_surface += n_exposed * face_area

            # Classify the neighbour voxels at exposed faces
            # Get the neighbour values (shift original_data the same way)
            shifted_orig = np.roll(original_data, shift=direction, axis=axis)
            shifted_orig[tuple(boundary_slice)] = 0  # treat boundary as 0

            neighbour_vals = shifted_orig[exposed]
            n_nonbrain = int((neighbour_vals == 0).sum())
            n_brain = int(((neighbour_vals > 0) & (neighbour_vals < threshold)).sum())

            contact_nonbrain += n_nonbrain * face_area
            contact_brain += n_brain * face_area

        rows.append({
            "index":                        idx,
            "Volume":                       round(volume_mm3, 2),
            "surface area":                 round(total_surface, 2),
            "contact area with non-brain":  round(contact_nonbrain, 2),
            "contact area with brain":      round(contact_brain, 2),
            "X":                            round(frac_x, 4),
            "Y":                            round(frac_y, 4),
            "Z":                            round(frac_z, 4),
            "mean density":                 round(mean_density, 2),
        })

    return rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    """Write blob statistics to a CSV file."""
    if not rows:
        print("  No blobs to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ CSV saved to {csv_path}")


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    name = input_path.name
    if name.endswith(".nii.gz"):
        base, ext = name[: -len(".nii.gz")], ".nii.gz"
    else:
        base, ext = input_path.stem, input_path.suffix
    if output_path is None:
        return input_path.parent / f"{base}_blobs{ext}"
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D blob detection & size-ranked labelling of a NIfTI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="Input NIfTI file or directory of NIfTI files.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output label map (single-file mode only).")
    p.add_argument("--min_threshold", type=float, default=75.0,
                   help="Minimum intensity threshold (default: 75).")
    p.add_argument("--max_threshold", type=float, default=250.0,
                   help="Maximum intensity threshold (default: 250).")
    p.add_argument("--min-cc", type=float, default=1.0,
                   help="Minimum object volume in cc to keep (default: 1.0).")
    return p.parse_args(argv)
    


def process_single_file(
    input_path: Path,
    output_path: Path | None,
    min_threshold: float,
    max_threshold: float,
    min_cc: float,
) -> int:
    """Run the full pipeline on one NIfTI file. Returns 0 on success."""
    output_path = build_output_path(input_path, output_path)

    print(f"Loading {input_path.name} …")
    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj)
    affine = img.affine

    print(f"  Shape: {data.shape}   Voxel size: "
          f"{np.abs(np.diag(affine[:3,:3])).round(2)} mm")

    ranked_labels, filtered, binary = process(
        data, affine,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        min_cc=min_cc,
    )

    # Save label map
    out_img = nib.Nifti1Image(ranked_labels, affine, img.header)
    out_img.header.set_data_dtype(np.int32)
    nib.save(out_img, str(output_path))
    print(f"\n✓ Label map saved to {output_path}")

    # Compute stats & write CSV
    if ranked_labels.max() > 0:
        print("\nComputing per-object statistics …")
        rows = compute_blob_stats(
            ranked_labels, data, affine,
            threshold=threshold,
        )
        # Derive CSV name from input file basename
        stem = input_path.name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        else:
            stem = Path(stem).stem
        csv_path = input_path.parent / f"{stem}.csv"
        write_csv(rows, csv_path)

        # Print a readable summary
        print(f"\n  {'Idx':>3}  {'Vol mm³':>9}  {'SurfA':>8}  "
              f"{'NonBr':>8}  {'Brain':>8}  "
              f"{'X':>6}  {'Y':>6}  {'Z':>6}  {'Density':>8}")
        print(f"  {'───':>3}  {'───────':>9}  {'─────':>8}  "
              f"{'─────':>8}  {'─────':>8}  "
              f"{'──':>6}  {'──':>6}  {'──':>6}  {'───────':>8}")
        for r in rows:
            print(f"  {r['index']:>3}  {r['Volume']:>9.1f}  "
                  f"{r['surface area']:>8.1f}  "
                  f"{r['contact area with non-brain']:>8.1f}  "
                  f"{r['contact area with brain']:>8.1f}  "
                  f"{r['X']:>6.3f}  {r['Y']:>6.3f}  {r['Z']:>6.3f}  "
                  f"{r['mean density']:>8.1f}")

    return 0


def collect_nifti_files(directory: Path) -> list[Path]:
    """Return sorted .nii/.nii.gz files, excluding generated outputs."""
    files = sorted(
        p for p in directory.iterdir()
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz")
    )
    return [
        p for p in files
        if not any(tag in p.name for tag in ("_blobs", "_mask", "_stripped"))
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()

    # --- Single file mode -----------------------------------------------------
    if input_path.is_file():
        return process_single_file(
            input_path, args.output, args.min_threshold, args.max_threshold, args.min_cc,
        )

    # --- Directory / batch mode -----------------------------------------------
    if not input_path.is_dir():
        print(f"ERROR: Not a file or directory: {input_path}", file=sys.stderr)
        return 1

    nifti_files = collect_nifti_files(input_path)
    if not nifti_files:
        print(f"No .nii/.nii.gz files found in {input_path}", file=sys.stderr)
        return 1

    total = len(nifti_files)
    succeeded, failed = 0, 0

    for idx, nii_path in enumerate(nifti_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{total}]  {nii_path.name}")
        print(f"{'═' * 60}")
        try:
            rc = process_single_file(
                nii_path, None, args.min_threshold, args.max_threshold, args.min_cc,
            )
            if rc == 0:
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"  BATCH COMPLETE: {succeeded} ok / {failed} failed  (total {total})")
    print(f"{'═' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
