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
    python label_blobs.py -i input.nii.gz
    python label_blobs.py -i input.nii.gz -o labels.nii.gz
    python label_blobs.py -i input.nii.gz --min-cc 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter, label, sum as ndsum, center_of_mass, binary_fill_holes


# 26-connected structuring element (3×3×3 cube of ones)
STRUCT_26 = np.ones((3, 3, 3), dtype=np.int32)

MIN_COMPONENT_VOXELS = 5  # remove connected components smaller than this

# Tissue HU Thresholds (Global)
MIN_WATER = -10
MAX_WATER = 14

MIN_BRAIN = MAX_WATER
MAX_BRAIN = 45

MIN_BLOOD = MAX_BRAIN
MAX_BLOOD = 150   # User requested 170 as an upper threshold



def volume_per_voxel_cc(affine: np.ndarray) -> float:
    """Return the volume of one voxel in cubic centimetres."""
    voxel_sizes = np.abs(np.diag(affine[:3, :3]))  # mm per voxel
    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    return voxel_vol_mm3 / 1000.0  # 1 cc = 1000 mm³


def filter_small_components(
    data: np.ndarray,
    min_voxels: int = MIN_COMPONENT_VOXELS,
) -> np.ndarray:
    """Remove per-class connected components smaller than *min_voxels*.

    For every unique non-zero value (class label) in *data*, 26-connected
    component analysis is run.  Components with fewer than *min_voxels*
    voxels are zeroed out.  The remaining voxels keep their original value.

    Returns a new array (the input is not modified).
    """
    out = data.copy()
    for c in np.unique(data):
        if c == 0:
            continue
        class_mask = (data == c).astype(np.int32)
        labelled, n = label(class_mask, structure=STRUCT_26)
        if n == 0:
            continue
        component_ids = np.arange(1, n + 1)
        sizes = ndsum(class_mask, labelled, component_ids).astype(int)
        for comp_id, sz in zip(component_ids, sizes):
            if sz < min_voxels:
                out[labelled == comp_id] = 0
    return out


def process(
    data: np.ndarray,
    affine: np.ndarray,
    min_cc: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply median filter and segment into Water, Brain, and Blood.

    Parameters
    ----------
    data : 3-D ndarray
        Raw voxel intensities (HU).
    affine : 4×4 ndarray
        NIfTI affine.
    min_cc : float
        Minimum object volume (cc) to keep for the *Blood* mask.

    Returns
    -------
    ranked_blood : 3-D ndarray (int32)
        Size-ranked label map for Blood (40-90 HU).
    water_mask : 3-D ndarray (int32)
        Binary mask for Water (-10 to 20 HU).
    brain_mask : 3-D ndarray (int32)
        Binary mask for Brain (15 to 40 HU).
    filtered : 3-D ndarray (float32)
        Median-filtered image (used for stats).
    """
    # ── 1. Median filter (3×3×3) ─────────────────────────────────────────
    # ── 1. Median filter (5x5x5) ─────────────────────────────────────────
    # Clip data to avoid bone blooming, then filter
    # A. Segmentation Filter: Mask out bone to prevent blooming into blood
    seg_input = data.copy()
    
    # Mask out anything above MAX_BLOOD (e.g. 170) to avoid blooming.
    seg_input[seg_input > MAX_BLOOD] = 0  # mask bone/calcification
    filtered = median_filter(seg_input.astype(np.float32), size=5)

    # ── 2. Thresholds for specific tissues ───────────────────────────────
    water_mask = ((filtered >= MIN_WATER) & (filtered <= MAX_WATER)).astype(np.int32)
    brain_mask = ((filtered >= MIN_BRAIN) & (filtered <= MAX_BRAIN)).astype(np.int32)
    blood_mask = ((filtered >= MIN_BLOOD) & (filtered <= MAX_BLOOD)).astype(np.int32)

    # ── 2b. Filter Water: Keep only water 'contained' in brain ───────────
    # Create a "Brain Envelope" by taking Brain + Blood and filling holes (ventricles).
    # This ensures we keep internal CSF but remove external background noise.
    solid_tissue = (brain_mask | blood_mask).astype(bool)
    # Fill holes in 3D to capture ventricles surrounded by brain
    brain_envelope = binary_fill_holes(solid_tissue)
    
    # Restrict water mask to this envelope
    water_mask = (water_mask & brain_envelope).astype(np.int32)

    # ── 3. Blood: Connected-component labelling (26-connected) ───────────
    labelled, n_objects = label(blood_mask, structure=STRUCT_26)
    print(f"  Blood components found ({MIN_BLOOD}-{MAX_BLOOD} HU): {n_objects}")

    if n_objects == 0:
        empty = np.zeros_like(data, dtype=np.int32)
        return empty, water_mask, brain_mask, filtered

    component_ids = np.arange(1, n_objects + 1)

    # ── 4. Measure each component's size (in voxels & cc) ────────────────
    vox_cc = volume_per_voxel_cc(affine)
    sizes_vox = ndsum(blood_mask, labelled, component_ids).astype(int)
    sizes_cc = sizes_vox * vox_cc

    # ── 5. Remove objects < min_cc ───────────────────────────────────────
    keep_mask = sizes_cc >= min_cc
    kept_ids = component_ids[keep_mask]
    kept_sizes = sizes_vox[keep_mask]
    print(f"  Components ≥ {min_cc} cc: {len(kept_ids)}  "
          f"(removed {n_objects - len(kept_ids)})")

    if len(kept_ids) == 0:
        empty = np.zeros_like(data, dtype=np.int32)
        return empty, water_mask, brain_mask, filtered

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

    return output, water_mask, brain_mask, filtered


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
    brain_mask: np.ndarray,
    water_mask: np.ndarray,
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
    brain_mask : 3-D int32 array
        Binary mask for Brain tissue.
    water_mask : 3-D int32 array
        Binary mask for Water.

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
        contact_water = 0.0

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
            shifted_brain = np.roll(brain_mask, shift=direction, axis=axis)
            shifted_brain[tuple(boundary_slice)] = 0

            shifted_water = np.roll(water_mask, shift=direction, axis=axis)
            shifted_water[tuple(boundary_slice)] = 0

            shifted_orig = np.roll(original_data, shift=direction, axis=axis)
            shifted_orig[tuple(boundary_slice)] = 0  # treat boundary as 0

            # Get values of neighbours at the exposed faces
            vals_brain = shifted_brain[exposed]
            vals_water = shifted_water[exposed]
            vals_orig = shifted_orig[exposed]

            n_brain = int((vals_brain > 0).sum())
            n_water = int((vals_water > 0).sum())
            n_nonbrain = int((vals_orig == 0).sum())

            contact_brain += n_brain * face_area
            contact_water += n_water * face_area
            contact_nonbrain += n_nonbrain * face_area

        rows.append({
            "index":                        idx,
            "Volume":                       round(volume_mm3, 2),
            "surface area":                 round(total_surface, 2),
            "contact area with non-brain":  round(contact_nonbrain, 2),
            "contact area with brain":      round(contact_brain, 2),
            "contact area with water":      round(contact_water, 2),
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
                   help="Output blood label map (single-file mode only).")
    p.add_argument("--min-cc", type=float, default=1.0,
                   help="Minimum object volume in cc to keep (default: 1.0).")
    return p.parse_args(argv)
    


def process_single_file(
    input_path: Path,
    output_path: Path | None,
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

    ranked_blood, water_mask, brain_mask, filtered = process(
        data, affine,
        min_cc=min_cc,
    )

    # Save Blood label map
    out_img = nib.Nifti1Image(ranked_blood, affine, img.header)
    out_img.header.set_data_dtype(np.int32)
    nib.save(out_img, str(output_path))
    print(f"\n✓ Blood label map saved to {output_path}")

    # Save filtered blood label map (small components removed, values intact)
    filtered_blood = filter_small_components(ranked_blood, min_voxels=MIN_COMPONENT_VOXELS)
    filtered_name = output_path.name.replace("_blobs", "_filtered")
    filtered_path = output_path.parent / filtered_name
    filt_img = nib.Nifti1Image(filtered_blood, affine, img.header)
    filt_img.header.set_data_dtype(np.int32)
    nib.save(filt_img, str(filtered_path))
    print(f"✓ Filtered blood map saved to {filtered_path}")

    # Save Water mask
    water_path = output_path.parent / f"{output_path.stem.replace('_blobs', '')}_water.nii.gz"
    if water_path.name.endswith(".gz.nii.gz"):  # fix double ext specific case
         water_path = output_path.parent / f"{output_path.stem.replace('_blobs', '')[:-4]}_water.nii.gz"

    # Cleaner path derivation for aux files:
    base_name = output_path.name.replace("_blobs.nii.gz", "").replace("_blobs.nii", "")
    water_path = output_path.parent / f"{base_name}_water.nii.gz"
    brain_path = output_path.parent / f"{base_name}_brain.nii.gz"

    nib.save(nib.Nifti1Image(water_mask, affine, img.header), str(water_path))
    print(f"✓ Water mask saved to {water_path}")

    nib.save(nib.Nifti1Image(brain_mask, affine, img.header), str(brain_path))
    print(f"✓ Brain mask saved to {brain_path}")

    # Save Filtered image
    filt_path = output_path.parent / f"{base_name}_filtered.nii.gz"
    nib.save(nib.Nifti1Image(filtered, affine, img.header), str(filt_path))
    print(f"✓ Filtered image saved to {filt_path}")

    # Compute stats & write CSV (only for labeled blood blobs)
    if ranked_blood.max() > 0:
        print("\nComputing per-object statistics (for Blood) …")
        rows = compute_blob_stats(
            ranked_blood, data, affine,
            brain_mask=brain_mask,
            water_mask=water_mask,
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
              f"{'NonBr':>8}  {'Brain':>8}  {'Water':>8}  "
              f"{'X':>6}  {'Y':>6}  {'Z':>6}  {'Density':>8}")
        print(f"  {'───':>3}  {'───────':>9}  {'─────':>8}  "
              f"{'─────':>8}  {'─────':>8}  {'─────':>8}  "
              f"{'──':>6}  {'──':>6}  {'──':>6}  {'───────':>8}")
        for r in rows:
            print(f"  {r['index']:>3}  {r['Volume']:>9.1f}  "
                  f"{r['surface area']:>8.1f}  "
                  f"{r['contact area with non-brain']:>8.1f}  "
                  f"{r['contact area with brain']:>8.1f}  "
                  f"{r['contact area with water']:>8.1f}  "
                  f"{r['X']:>6.3f}  {r['Y']:>6.3f}  {r['Z']:>6.3f}  "
                  f"{r['mean density']:>8.1f}")

    return 0


import re

def natural_sort_key(s: str) -> list[str | int]:
    """Sort strings containing numbers naturally (e.g. 2 < 10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def collect_nifti_files(directory: Path) -> list[Path]:
    """Return sorted .nii/.nii.gz files that end with _stripped, in numerical order."""
    files = [
        p for p in directory.iterdir()
        if (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    ]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()

    # --- Single file mode -----------------------------------------------------
    if input_path.is_file():
        return process_single_file(
            input_path, args.output, args.min_cc,
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
                nii_path, None, args.min_cc,
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
