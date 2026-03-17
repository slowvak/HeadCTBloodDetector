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
import torch as _torch
from scipy import ndimage
from run_synthstrip import ensure_synthstrip, run_synthstrip as _run_synthstrip, build_output_paths as _build_stripped_paths
from blast_ct.read_config import get_model, get_test_loader
from blast_ct.nifti.savers import NiftiPatchSaver, reconstruct_image
from blast_ct.trainer.inference import ModelInference

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent

DEFAULT_INPUT_DIR = Path("/Volumes/OWC Express 1M2/CQ500_NII")
DEFAULT_OUTPUT_DIR = Path("/Volumes/OWC Express 1M2/CQ500_NII/finetuned")
DEFAULT_MODEL_PATH = REPO_ROOT / "finetune_runs" / "run_001" / "saved_models" / "model_best.torch_model"
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
    try:
        img = sitk.ReadImage(str(src))
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            print(f"  SKIP {src.name}: {e}")
            return None
        # Non-orthonormal direction cosines — load via nibabel and rebuild a clean SimpleITK image
        nib_img = nib.load(str(src))
        arr = np.asarray(nib_img.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]  # take first volume
        img = sitk.GetImageFromArray(arr.T)  # nibabel (X,Y,Z) → SimpleITK (Z,Y,X)
        zooms = nib_img.header.get_zooms()[:3]
        img.SetSpacing([float(z) for z in zooms])

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


class _EntropySaver(NiftiPatchSaver):
    """NiftiPatchSaver that computes per-image softmax entropy from in-memory reconstructions.

    blast-ct's save_image() calls sitk.Resample(4D_prob_map, 3D_reference) which raises
    a dimension-mismatch exception, so the prob_maps file is never actually written even
    though its path appears in prediction.csv.  This subclass bypasses that by computing
    entropy directly from the reconstruction array before save_image is called.
    """

    def __init__(self, job_dir, dataloader):
        super().__init__(job_dir, dataloader, write_prob_maps=False)
        self.entropies: dict[str, float] = {}
        self._prob_patches: list = []

    def append(self, state):
        super().append(state)  # accumulates state['pred'] (write_prob_maps=False)
        self._prob_patches += list(state['prob'].cpu().detach())

    def __call__(self, state):
        # Capture image metadata BEFORE super() increments image_index
        target_shape, center_points = self.dataset.image_mapping[self.image_index]
        patches_in_image = len(center_points)
        image_id = str(self.dataset.data_index.iloc[self.image_index].name)

        result = super().__call__(state)  # handles prediction saving + image_index advance

        if len(self._prob_patches) >= patches_in_image:
            target_patch_shape = self.dataset.patch_sampler.target_patch_size
            patches = list(_torch.stack(self._prob_patches[:patches_in_image]).numpy())
            self._prob_patches = self._prob_patches[patches_in_image:]
            recon = reconstruct_image(patches, target_shape, center_points, target_patch_shape)
            # recon shape after reconstruct_image's transpose: (Z, Y, X, C)
            probs = np.clip(recon.astype(np.float32), 1e-8, 1.0)
            h = -np.sum(probs * np.log(probs), axis=-1)  # (Z, Y, X)
            self.entropies[image_id] = float(h.mean())

        return result


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def clean_prediction(seg: np.ndarray,
                     image: nib.Nifti1Image,
                     min_value: float = 30.0,
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
    # nibabel loads as (X, Y, Z); SimpleITK seg is (Z, Y, X) — transpose to match
    ct = np.asarray(image.dataobj, dtype=np.float32).T
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
    device: _torch.device,
) -> dict[str, dict]:
    """Run model on all files in batches.

    Returns mapping {basename -> {"pred": Path, "entropy": float, "attrition": float}}
      entropy    : mean per-voxel softmax entropy over all brain voxels (nats); higher = more uncertain
      attrition  : fraction of raw foreground voxels removed by clean_prediction
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

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
            saver       = _EntropySaver(tmp, test_loader)

            ModelInference(tmp, device, model, saver, str(model_path), "segmentation")(test_loader)

            # Copy predictions out of temp dir
            pred_csv = os.path.join(tmp, "predictions", "prediction.csv")
            if os.path.exists(pred_csv):
                pred_index = pd.read_csv(pred_csv)
                for _, row in pred_index.iterrows():
                    src      = Path(str(row["prediction"]))
                    basename = src.name.replace("_prediction.nii.gz", "")
                    dest     = output_dir / f"{basename}_segmentation.nii.gz"

                    # Load prediction array and clean before saving
                    pred_img  = sitk.ReadImage(str(src))
                    seg_arr   = np.round(sitk.GetArrayFromImage(pred_img)).astype(np.int32)
                    raw_fg    = int((seg_arr != 0).sum())

                    orig_path = orig_by_basename.get(basename)
                    if orig_path is not None:
                        ct_nib  = nib.load(str(orig_path))
                        seg_arr = clean_prediction(seg_arr, ct_nib)
                    else:
                        print(f"    WARNING: original CT not found for {basename}, skipping clean_prediction")

                    cleaned_fg = int((seg_arr != 0).sum())
                    attrition  = (raw_fg - cleaned_fg) / raw_fg if raw_fg > 0 else float("nan")

                    cleaned_img = sitk.GetImageFromArray(seg_arr)
                    cleaned_img.CopyInformation(pred_img)
                    sitk.WriteImage(cleaned_img, str(dest))

                    entropy = saver.entropies.get(basename, float("nan"))

                    results[basename] = {"pred": dest, "entropy": entropy, "attrition": attrition}
                    print(f"    {dest.name}  entropy={entropy:.4f}  attrition={attrition:.2%}"
                          if not np.isnan(entropy) else f"    {dest.name}")

    return results

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def voxel_counts(arr: np.ndarray, class_names: list[str]) -> dict[str, int]:
    """Return voxel count per class (indexed by class name)."""
    return {name: int((arr == c).sum()) for c, name in enumerate(class_names)}


def compute_dice(pred_path: Path, ref_path: Path,
                 class_names: list[str]) -> dict[str, float]:
    """Return Dice score per foreground class. Returns {} on shape mismatch."""
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(int)
    ref  = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path))).astype(int)
    if pred.shape != ref.shape:
        print(f"  WARNING: shape mismatch {pred_path.name} vs {ref_path.name} — skipping Dice")
        return {}
    out = {}
    for c, name in enumerate(class_names):
        if c == 0:
            continue  # skip background
        p, r  = pred == c, ref == c
        tp    = float(np.logical_and(p, r).sum())
        denom = float(p.sum() + r.sum())
        out[name] = 2.0 * tp / denom if denom > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "  n/a "


def print_summary(records: list[dict], class_names: list[str]) -> None:
    fg = class_names[1:]  # foreground classes
    print(f"\n{'='*80}")
    print(f"  {'File':<40}  " + "  ".join(f"{n:>8}" for n in fg)
          + f"  {'entropy':>8}  {'attrtn':>7}")
    print(f"  {'-'*78}")
    for rec in sorted(records, key=lambda r: r["entropy"]
                      if not np.isnan(r["entropy"]) else -1, reverse=True):
        voxels  = "  ".join(f"{rec['voxels'].get(n, 0):>8}" for n in fg)
        ent_str = f"{rec['entropy']:>8.4f}" if not np.isnan(rec["entropy"]) else "     n/a"
        att_str = f"{rec['attrition']:>6.1%}" if not np.isnan(rec["attrition"]) else "   n/a"
        print(f"  {rec['basename']:<40}  {voxels}  {ent_str}  {att_str}")
        if rec["dice"]:
            dice_str = "  ".join(f"{_fmt(rec['dice'].get(n, float('nan'))):>8}" for n in fg)
            print(f"  {'  (Dice)':40}  {dice_str}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",        type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir",       type=Path, default=None)
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
    if args.output_dir is None:
        args.output_dir = args.input_dir / "finetuned"

    for label, path in [("Input dir",  args.input_dir),
                         ("Model",      args.model_path),
                         ("Config",     args.config_file)]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    with open(args.config_file) as f:
        config = json.load(f)
    input_files = collect_inputs(args.input_dir)
    if not input_files:
        raise SystemExit(f"No eligible .nii/.nii.gz files found in {args.input_dir}")
    print(f"Found {len(input_files)} eligible file(s).")

    # --- Inference ---
    if args.skip_inference:
        print("\nSkipping inference (--skip-inference set).")
        predictions = {
            get_basename(p): {"pred": args.output_dir / f"{get_basename(p)}_segmentation.nii.gz",
                              "entropy": float("nan"), "attrition": float("nan")}
            for p in input_files
        }
    else:
        predictions = run_inference(
            input_files, args.output_dir, config, args.model_path, _torch.device("cpu")
        )

    class_names = config["data"]["class_names"]

    # --- Per-file stats ---
    records: list[dict] = []

    for p in input_files:
        basename = get_basename(p)
        result   = predictions.get(basename)
        if result is None:
            print(f"  WARNING: no prediction found for {basename}, skipping.")
            continue
        pred_path = result["pred"]
        if not pred_path.exists():
            print(f"  WARNING: prediction file missing for {basename}, skipping.")
            continue

        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(int)
        counts   = voxel_counts(pred_arr, class_names)

        ref_path = args.input_dir / (basename + args.reference_suffix)
        dice     = compute_dice(pred_path, ref_path, class_names) if ref_path.exists() else {}

        records.append({
            "basename":  basename,
            "voxels":    counts,
            "dice":      dice,
            "entropy":   result["entropy"],
            "attrition": result["attrition"],
        })

    print_summary(records, class_names)

    # --- Save CSV summary (sorted by entropy descending = highest priority first) ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in sorted(records, key=lambda r: r["entropy"]
                      if not np.isnan(r["entropy"]) else -1, reverse=True):
        row = {
            "file":      rec["basename"],
            "entropy":   round(rec["entropy"],   6),
            "attrition": round(rec["attrition"], 4),
        }
        for c, name in enumerate(class_names):
            row[f"{name}_voxels"] = rec["voxels"].get(name, 0)
            if c > 0 and rec["dice"]:
                row[f"{name}_dice"] = rec["dice"].get(name, float("nan"))
        rows.append(row)

    summary_path = args.output_dir / "performance_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved to: {summary_path}")


def _ensure_stripped(orig: Path, synthstrip_cmd: str) -> Path | None:
    """Return the skull-stripped version of *orig*, running synthstrip if needed.

    The stripped file is written alongside the original as <stem>_stripped.nii.gz.
    Returns None if synthstrip fails.
    """
    stripped, mask = _build_stripped_paths(orig, None, None)
    if stripped.exists():
        return stripped
    print(f"  Stripping: {orig.name}")
    result = _run_synthstrip(
        input_path=orig,
        output_path=stripped,
        mask_path=mask,
        synthstrip_cmd=synthstrip_cmd,
    )
    return stripped if result.returncode == 0 else None


def new_predictions(
    input_dir: Path | str,
    output_dir: Path | str,
    model_dir: Path | str,
    config_file: Path | str = DEFAULT_CONFIG,
    synthstrip_cmd: str = "synthstrip-docker",
) -> None:
    """Apply the finetuned model to every CT .nii.gz in input_dir.

    For each file that lacks a *_stripped.nii.gz counterpart, synthstrip is
    run first.  Inference is always performed on the stripped image.
    Predictions are written to output_dir as <stem>_prediction.nii.gz.
    Files that already have a prediction in output_dir are skipped.

    Parameters
    ----------
    input_dir       : directory containing input CT scans (.nii / .nii.gz)
    output_dir      : where to write *_prediction.nii.gz files
    model_dir       : directory containing the model file; prefers
                      model_best.torch_model, then model_last.torch_model
    config_file     : path to finetune_config.json (default: repo default)
    synthstrip_cmd  : name / path of the synthstrip-docker executable
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)

    # Resolve model file
    for candidate in ["model_best.torch_model", "model_last.torch_model"]:
        if (model_dir / candidate).exists():
            model_path = model_dir / candidate
            break
    else:
        found = sorted(model_dir.glob("*.torch_model"))
        if not found:
            raise FileNotFoundError(f"No *.torch_model file found in {model_dir}")
        model_path = found[-1]
    print(f"Model: {model_path}")

    synthstrip_cmd = ensure_synthstrip(synthstrip_cmd)

    with open(config_file) as f:
        config = json.load(f)

    # Collect original CTs — exclude already-derived files
    _EXCLUDE_TAGS = ("_stripped", "_mask", "_prediction", "_seg", "_segmentation",
                     "_brain_mask", "_prob_maps")
    raw_files = sorted(
        p for p in input_dir.iterdir()
        if (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        and not any(tag in p.name for tag in _EXCLUDE_TAGS)
    )

    # Skip originals that already have a prediction
    pending = [
        p for p in raw_files
        if not (output_dir / f"{get_basename(p)}_prediction.nii.gz").exists()
    ]
    skipped = len(raw_files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} file(s) with existing prediction.")
    if not pending:
        print("All predictions already exist — nothing to do.")
        return

    # --- Skull-strip pass ---------------------------------------------------
    print(f"\nEnsuring skull-stripped versions for {len(pending)} file(s)…")
    # Maps original path → stripped path (only entries where strip succeeded)
    stripped_for: dict[Path, Path] = {}
    for orig in pending:
        stripped = _ensure_stripped(orig, synthstrip_cmd)
        if stripped is None:
            print(f"  WARNING: synthstrip failed for {orig.name}, skipping.")
        else:
            stripped_for[orig] = stripped

    if not stripped_for:
        print("No files could be stripped — nothing to run inference on.")
        return

    # Build the inference input list (stripped files), keyed back to originals
    # for HU-based post-processing
    input_files  = list(stripped_for.values())   # stripped paths (for model)
    orig_by_stripped: dict[str, Path] = {
        get_basename(s): o for o, s in stripped_for.items()
    }

    print(f"\nRunning inference on {len(input_files)} file(s)…")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _torch.device("cpu")

    for batch_start in range(0, len(input_files), BATCH_SIZE):
        batch = input_files[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(input_files) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  Batch {batch_num}/{total_batches}  ({len(batch)} files)")

        with tempfile.TemporaryDirectory(prefix="blast_newpred_") as tmp:
            cast_dir = os.path.join(tmp, "cast")
            os.makedirs(cast_dir, exist_ok=True)

            entries: list[tuple[Path, Path]] = []
            for stripped in batch:
                prepared = prepare_image(stripped, cast_dir)
                if prepared is None:
                    continue
                entries.append((stripped, prepared))

            if not entries:
                continue

            csv_path = os.path.join(tmp, "test.csv")
            pd.DataFrame(
                [{"id": get_basename(s), "image": str(prepared)}
                 for s, prepared in entries]
            ).to_csv(csv_path, index=False)

            model       = get_model(config)
            test_loader = get_test_loader(config, model, csv_path, use_cuda=False)
            saver       = _EntropySaver(tmp, test_loader)

            ModelInference(tmp, device, model, saver, str(model_path), "segmentation")(test_loader)

            pred_csv = os.path.join(tmp, "predictions", "prediction.csv")
            if not os.path.exists(pred_csv):
                print("  WARNING: no prediction.csv produced for this batch")
                continue

            pred_index = pd.read_csv(pred_csv)
            for _, row in pred_index.iterrows():
                src      = Path(str(row["prediction"]))
                basename = src.name.replace("_prediction.nii.gz", "")
                dest     = output_dir / f"{basename}_prediction.nii.gz"

                pred_img = sitk.ReadImage(str(src))
                seg_arr  = np.round(sitk.GetArrayFromImage(pred_img)).astype(np.int32)

                # Post-process against the original (non-stripped) CT for HU filtering
                orig_path = orig_by_stripped.get(basename)
                if orig_path is not None:
                    ct_nib  = nib.load(str(orig_path))
                    seg_arr = clean_prediction(seg_arr, ct_nib)

                out_img = sitk.GetImageFromArray(seg_arr)
                out_img.CopyInformation(pred_img)
                sitk.WriteImage(out_img, str(dest))
                print(f"    → {dest.name}")

    print(f"\nDone. Predictions written to: {output_dir}")


if __name__ == "__main__":
    _input_dir  = Path.home() / "Desktop" / "CQ500_NII"
    _output_dir = _input_dir / "predictions"
    _model_dir  = DEFAULT_MODEL_PATH.parent
    new_predictions(_input_dir, _output_dir, _model_dir)
