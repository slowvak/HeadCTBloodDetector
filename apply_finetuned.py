#!/usr/bin/env python3
"""Apply the finetuned DeepMedic model to CTs in a directory.

For every .nii / .nii.gz file in INPUT_DIR (excluding *_masked* files):
  - Runs the finetuned model
  - Saves prediction to OUTPUT_DIR/<stem>_predictions.nii.gz

Then prints and saves a performance summary split by:
  • skull-stripped files  (basename contains '_stripped')
  • raw CT files          (all others)

If a reference segmentation exists alongside each input
(default: matching file with suffix '_prediction.nii.gz'),
Dice / recall / precision are also computed per class.

Usage
-----
    uv run apply_finetuned.py
    uv run apply_finetuned.py --input-dir /path/to/scans --model-path finetune_runs/run_001/saved_models/model_last.torch_model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage

from blast_ct.read_config import get_model, get_test_loader
from blast_ct.nifti.savers import NiftiPatchSaver
from blast_ct.trainer.inference import ModelInference

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent

DEFAULT_INPUT_DIR = Path("/Volumes/OWC Express 1M2/CQ500_NII")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "finetuned"
DEFAULT_MODEL_PATH = REPO_ROOT / "finetune_runs" / "run_001" / "saved_models" / "model_last.torch_model"
DEFAULT_CONFIG     = REPO_ROOT / "finetune_config.json"

MIN_COMPONENT_VOXELS = 5   # connected components smaller than this many voxels are removed
MEDIAN_RADIUS    = 2     # 2D 5×5 per axial slice (radius 2 = ±2 voxels = 5 total)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_basename(p: Path) -> str:
    """Return filename without .nii or .nii.gz extension."""
    name = p.name
    return name[:-7] if name.endswith(".nii.gz") else p.stem


def collect_inputs(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        and "_stripped"     in p.stem
    )

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def prepare_image(src: Path, tmp_dir: str) -> Path | None:
    """Prepare an image for inference.

    1. If 4D, extracts the first 3D volume via numpy (avoids 4D NIfTI header bleed-through).
    2. Casts float64 → float32 (SimpleITK ResampleImageFilter limitation).
    3. Applies a 2D 5×5 median filter to each axial slice.

    Always writes a processed copy to tmp_dir.
    Returns None (skip) if the image has an unexpected dimensionality (not 3D or 4D).
    """
    img = sitk.ReadImage(str(src))

    if img.GetDimension() == 4:
        # Go through numpy to get a clean 3D image free of 4D header metadata.
        # sitk returns axes as (t, z, y, x) so arr[0] is the first volume.
        arr = sitk.GetArrayFromImage(img)          # shape: (t, z, y, x)
        arr3d = arr[0] if arr.ndim == 4 else arr   # (z, y, x)
        img3d = sitk.GetImageFromArray(arr3d)
        img3d.SetSpacing(img.GetSpacing()[:3])
        img3d.SetOrigin(img.GetOrigin()[:3])
        # Direction is a flat 4×4 matrix; take the upper-left 3×3 block.
        d = img.GetDirection()
        img3d.SetDirection((d[0], d[1], d[2], d[4], d[5], d[6], d[8], d[9], d[10]))
        img = img3d
    elif img.GetDimension() != 3:
        print(f"  SKIP {src.name}: {img.GetDimension()}D image not supported.")
        return None

    if img.GetPixelID() == sitk.sitkFloat64:
        img = sitk.Cast(img, sitk.sitkFloat32)

    # 2D 5×5 median filter per axial slice (radius in x and y; 0 = no filter in z)
    mf = sitk.MedianImageFilter()
    mf.SetRadius([MEDIAN_RADIUS, MEDIAN_RADIUS, 0])
    img = mf.Execute(img)

    dest = Path(tmp_dir) / src.name
    sitk.WriteImage(img, str(dest))
    return dest


BATCH_SIZE = 10  # files per inference batch (keeps memory bounded)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def clean_prediction(seg: np.ndarray,
                     image: nib.Nifti1Image,
                     min_value: float = 40.0,
                     max_value: float = 300.0,
                     min_size: int = 5) -> np.ndarray:
    """Return a cleaned copy of *seg* by applying HU thresholding then
    removing small 3-D connected components.

    Parameters
    ----------
    seg        : integer label array (shape Z×Y×X).
    image      : nibabel image of the source CT (HU values), same voxel grid.
    min_value  : HU lower bound; predictions below this are zeroed (default 40).
    max_value  : HU upper bound; predictions above this are zeroed (default 300).
    min_size   : connected components with fewer voxels than this are removed
                 (default 5).

    Logic
    -----
    ndimage.label(out == c) — for each foreground class c:

    1. out == c creates a boolean 3D binary mask — True where the label equals c, False everywhere else.
    2. ndimage.label(...) scans that binary mask and assigns a unique integer ID to each group of connected True voxels. Two voxels are considered
    connected if they share a face (6-connectivity by default). The result labeled is an array the same shape as seg where each distinct
    connected blob has its own integer (1, 2, 3, …), and background is 0.
    3. np.unique(labeled[labeled > 0], return_counts=True) counts how many voxels belong to each blob.
    4. Any blob whose voxel count is below min_size has all its voxels set to 0 in out — effectively erasing it.

    This runs separately per class, so a small IPH blob and a small SAH blob are each evaluated independently against min_size.


    Returns
    -------
    Cleaned copy of seg as an int32 ndarray.
    """
    ct = np.asarray(image.dataobj, dtype=np.float32)
    out = seg.copy().astype(np.int32)

    # Step 1: zero out predictions where CT HU is outside the valid range
    bad_hu = (ct < min_value) | (ct > max_value)
    out[bad_hu & (out != 0)] = 0

    # Step 2: per-class 3-D connected component analysis; drop small blobs
    for c in np.unique(out):
        if c == 0:
            continue
        labeled, _ = ndimage.label(out == c)
        labels, counts = np.unique(labeled[labeled > 0], return_counts=True)
        for lbl, cnt in zip(labels, counts):
            if cnt < min_size:
                out[labeled == lbl] = 0

    return out


def run_inference(
    input_files: list[Path],
    output_dir: Path,
    config: dict,
    model_path: Path,
    device: torch.device,
) -> dict[str, Path]:
    """Run model on all files in batches; return mapping {basename -> prediction_path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    print(f"\nRunning inference on {len(input_files)} file(s) in batches of {BATCH_SIZE}…")
    for batch_start in range(0, len(input_files), BATCH_SIZE):
        batch = input_files[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(input_files) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  Batch {batch_num}/{total_batches}  ({len(batch)} files)")

        with tempfile.TemporaryDirectory(prefix="blast_finetune_") as tmp:
            # Prepare images (cast float64 → float32, extract 3D from 4D)
            cast_dir = os.path.join(tmp, "cast")
            os.makedirs(cast_dir, exist_ok=True)

            entries: list[tuple[Path, Path]] = []
            orig_by_basename: dict[str, Path] = {}
            for orig in batch:
                prepared = prepare_image(orig, cast_dir)
                if prepared is None:
                    continue
                entries.append((orig, prepared))
                orig_by_basename[get_basename(orig)] = orig

            if not entries:
                continue

            csv_path = os.path.join(tmp, "test.csv")
            pd.DataFrame(
                [{"id": get_basename(orig), "image": str(prepared)}
                 for orig, prepared in entries]
            ).to_csv(csv_path, index=False)

            model       = get_model(config)
            test_loader = get_test_loader(config, model, csv_path, use_cuda=(device.type != "cpu"))
            saver       = NiftiPatchSaver(tmp, test_loader, write_prob_maps=False)

            ModelInference(tmp, device, model, saver, str(model_path), "segmentation")(test_loader)

            # Copy predictions out of temp dir
            pred_csv = os.path.join(tmp, "predictions", "prediction.csv")
            if os.path.exists(pred_csv):
                pred_index = pd.read_csv(pred_csv)
                for _, row in pred_index.iterrows():
                    src      = Path(str(row["prediction"]))
                    basename = src.name.replace("_prediction.nii.gz", "")
                    dest     = output_dir / f"{basename}_predictions.nii.gz"

                    # Load prediction array and clean before saving
                    pred_img  = sitk.ReadImage(str(src))
                    seg_arr   = np.round(sitk.GetArrayFromImage(pred_img)).astype(np.int32)
                    orig_path = orig_by_basename.get(basename)
                    if orig_path is not None:
                        ct_nib  = nib.load(str(orig_path))
                        seg_arr = clean_prediction(seg_arr, ct_nib)
                    else:
                        print(f"    WARNING: original CT not found for {basename}, skipping clean_prediction")

                    cleaned_img = sitk.GetImageFromArray(seg_arr)
                    cleaned_img.CopyInformation(pred_img)
                    sitk.WriteImage(cleaned_img, str(dest))
                    results[basename] = dest
                    print(f"    {dest.name}")

    return results

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

CLASS_NAMES = ["background", "IPH", "EAH", "edema", "IVH", "SAH"]


def voxel_stats(arr: np.ndarray, num_classes: int) -> dict:
    total = arr.size
    return {
        CLASS_NAMES[c]: {
            "count": int((arr == c).sum()),
            "pct":   100.0 * (arr == c).sum() / max(total, 1),
        }
        for c in range(num_classes)
    }


def compute_dice_metrics(pred_path: Path, ref_path: Path, num_classes: int) -> dict:
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(int)
    ref  = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path))).astype(int)
    if pred.shape != ref.shape:
        print(f"  WARNING: shape mismatch for {pred_path.name} vs {ref_path.name} — skipping metrics")
        return {}
    out = {}
    for c in range(1, num_classes):   # skip background
        name = CLASS_NAMES[c]
        p, r = pred == c, ref == c
        tp        = float(np.logical_and(p, r).sum())
        denom_d   = float(p.sum() + r.sum())
        out[name] = {
            "dice":      2.0 * tp / denom_d     if denom_d   > 0 else float("nan"),
            "recall":    tp / float(r.sum())     if r.sum()   > 0 else float("nan"),
            "precision": tp / float(p.sum())     if p.sum()   > 0 else float("nan"),
        }
    return out

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "  n/a "


def print_group_report(group_name: str, records: list[dict], num_classes: int) -> None:
    print(f"\n{'='*72}")
    print(f"  {group_name}  ({len(records)} file(s))")
    print(f"{'='*72}")

    if not records:
        print("  (no files in this group)")
        return

    # Class distribution
    print(f"\n  {'Class':<12}  {'Mean %':>8}  {'Std %':>8}  {'Mean voxels':>12}")
    print(f"  {'-'*48}")
    for c in range(num_classes):
        name  = CLASS_NAMES[c]
        pcts  = [r["stats"][name]["pct"]   for r in records]
        cnts  = [r["stats"][name]["count"] for r in records]
        print(f"  {name:<12}  {np.mean(pcts):8.3f}  {np.std(pcts):8.3f}  {np.mean(cnts):12.0f}")

    # Dice / recall / precision (only if any reference was found)
    met_records = [r for r in records if r["metrics"]]
    if not met_records:
        print("\n  (no reference segmentations found — Dice not computed)")
        return

    print(f"\n  Dice vs reference  ({len(met_records)}/{len(records)} files have a reference)")
    print(f"\n  {'Class':<12}  {'Dice':>8}  {'Recall':>8}  {'Precision':>10}")
    print(f"  {'-'*44}")
    for c in range(1, num_classes):
        name   = CLASS_NAMES[c]
        dices  = [r["metrics"][name]["dice"]      for r in met_records if name in r["metrics"]]
        recs   = [r["metrics"][name]["recall"]    for r in met_records if name in r["metrics"]]
        precs  = [r["metrics"][name]["precision"] for r in met_records if name in r["metrics"]]
        valid  = lambda lst: [x for x in lst if not np.isnan(x)]
        d_str  = _fmt(np.mean(valid(dices)))  if valid(dices)  else "  n/a "
        r_str  = _fmt(np.mean(valid(recs)))   if valid(recs)   else "  n/a "
        p_str  = _fmt(np.mean(valid(precs)))  if valid(precs)  else "  n/a "
        print(f"  {name:<12}  {d_str:>8}  {r_str:>8}  {p_str:>10}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",        type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir",       type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-path",       type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--config-file",      type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--reference-suffix",
        type=str,
        default="_prediction.nii.gz",
        help="Suffix appended to each input basename to find a reference segmentation "
             "in the same input directory.  Default: _prediction.nii.gz",
    )
    p.add_argument(
        "--skip-inference",
        action="store_true",
        default=False,
        help="Skip inference and only recompute the performance summary from existing predictions.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    for label, path in [("Input dir",  args.input_dir),
                         ("Model",      args.model_path),
                         ("Config",     args.config_file)]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    with open(args.config_file) as f:
        config = json.load(f)
    num_classes = config["data"]["num_classes"]

    input_files = collect_inputs(args.input_dir)
    if not input_files:
        raise SystemExit(f"No eligible .nii/.nii.gz files found in {args.input_dir}")
    print(f"Found {len(input_files)} eligible file(s).")

    # --- Inference ---
    if args.skip_inference:
        print("\nSkipping inference (--skip-inference set).")
        predictions = {
            get_basename(p): args.output_dir / f"{get_basename(p)}_predictions.nii.gz"
            for p in input_files
        }
    else:
        predictions = run_inference(
            input_files, args.output_dir, config, args.model_path, torch.device("cpu")
        )

    # --- Per-file stats ---
    stripped_records: list[dict] = []
    raw_records:      list[dict] = []

    for p in input_files:
        basename  = get_basename(p)
        pred_path = predictions.get(basename)
        if pred_path is None or not pred_path.exists():
            print(f"  WARNING: no prediction found for {basename}, skipping.")
            continue

        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(int)
        stats    = voxel_stats(pred_arr, num_classes)

        ref_path = args.input_dir / (basename + args.reference_suffix)
        metrics  = compute_dice_metrics(pred_path, ref_path, num_classes) if ref_path.exists() else {}

        record = {"basename": basename, "stats": stats, "metrics": metrics}
        (stripped_records if "_stripped" in basename else raw_records).append(record)

    print_group_report("SKULL-STRIPPED", stripped_records, num_classes)
    print_group_report("RAW CT",         raw_records,      num_classes)

    # --- Save CSV summary ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in stripped_records + raw_records:
        row = {
            "file":  rec["basename"],
            "group": "stripped" if "_stripped" in rec["basename"] else "raw",
        }
        for c in range(num_classes):
            name = CLASS_NAMES[c]
            row[f"{name}_pct"]    = round(rec["stats"][name]["pct"],   4)
            row[f"{name}_voxels"] = rec["stats"][name]["count"]
            if c > 0 and rec["metrics"]:
                m = rec["metrics"].get(name, {})
                row[f"{name}_dice"]      = m.get("dice",      float("nan"))
                row[f"{name}_recall"]    = m.get("recall",    float("nan"))
                row[f"{name}_precision"] = m.get("precision", float("nan"))
        rows.append(row)

    summary_path = args.output_dir / "performance_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()
