#!/usr/bin/env python3
"""Wrapper around blast-ct for brain lesion segmentation on head CT."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _venv_cmd(name: str) -> str:
    """Return the path to a command installed in the same venv as this script."""
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def build_output_path(input_path: Path, output_dir: Path | None = None) -> Path:
    """Derive a default prediction output path (.nii.gz) for a given input."""
    stem = input_path.name
    base = stem[: -len(".nii.gz")] if stem.endswith(".nii.gz") else input_path.stem
    parent = output_dir if output_dir else input_path.parent
    return parent / f"{base}_prediction.nii.gz"


def collect_nifti_files(directory: Path) -> list[Path]:
    """Return sorted list of .nii / .nii.gz files in *directory*."""
    files = sorted(
        p for p in directory.iterdir()
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz")
    )
    return [p for p in files if not any(tag in p.name for tag in ("_prediction", "_stripped", "_mask"))]


def run_single(
    input_path: Path,
    output_path: Path,
    device: str,
    ensemble: bool,
    do_localisation: bool,
) -> int:
    """Run blast-ct on a single NIfTI file."""
    cmd = [
        _venv_cmd("blast-ct"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--device", device,
    ]
    if ensemble:
        cmd.append("--ensemble")
    if do_localisation:
        cmd.append("--do-localisation")

    print(f"{'─' * 60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'─' * 60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nERROR: blast-ct exited with code {result.returncode}", file=sys.stderr)
    else:
        print(f"\nDone. Prediction saved to: {output_path}")

    return result.returncode


def run_batch(
    nifti_files: list[Path],
    job_dir: Path,
    device: str,
    ensemble: bool,
    do_localisation: bool,
    overwrite: bool,
) -> int:
    """Run blast-ct-inference on a list of NIfTI files via a generated CSV."""
    job_dir.mkdir(parents=True, exist_ok=True)
    csv_path = job_dir / "test.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image"])
        for p in nifti_files:
            stem = p.name
            scan_id = stem[: -len(".nii.gz")] if stem.endswith(".nii.gz") else p.stem
            writer.writerow([scan_id, str(p)])

    cmd = [
        _venv_cmd("blast-ct-inference"),
        "--job-dir", str(job_dir),
        "--test-csv-path", str(csv_path),
        "--device", device,
        "--overwrite", "True" if overwrite else "False",
    ]
    if ensemble:
        cmd += ["--ensemble", "True"]
    if do_localisation:
        cmd += ["--do-localisation", "True"]

    print(f"{'─' * 60}")
    print(f"  Job dir : {job_dir}")
    print(f"  CSV     : {csv_path}")
    print(f"  Files   : {len(nifti_files)}")
    print(f"  Command : {' '.join(cmd)}")
    print(f"{'─' * 60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nERROR: blast-ct-inference exited with code {result.returncode}", file=sys.stderr)
    else:
        print(f"\nDone. Predictions saved in: {job_dir / 'predictions'}/")

    return result.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply blast-ct lesion segmentation to a head CT (single file or directory).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output labels: 1=IPH, 2=EAH, 3=Perilesional oedema, 4=IVH\n"
            "\nExamples:\n"
            "  %(prog)s -i scan.nii.gz\n"
            "  %(prog)s -i scan.nii.gz -o prediction.nii.gz --device 0 --ensemble\n"
            "  %(prog)s -i /scans/dir/ -o /output/job/ --device cpu\n"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input NIfTI file (.nii / .nii.gz) or directory of NIfTI files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Single-file mode: path for the output prediction (.nii.gz). "
            "Directory mode: job-dir where predictions and logs are written "
            "(default: <input_dir>/blast_ct_output/)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device: 'cpu' or integer GPU index (default: cpu).",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Use ensemble of 12 models — slower but more accurate (recommended with GPU).",
    )
    parser.add_argument(
        "--do-localisation",
        action="store_true",
        default=False,
        help="Calculate lesion volume per brain region (adds a registration step).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="(Batch mode) Overwrite an existing job-dir. Omit to resume a previous run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()

    # --- Single-file mode ---
    if input_path.is_file():
        output_path = args.output.resolve() if args.output else build_output_path(input_path)
        if not str(output_path).endswith(".nii.gz"):
            output_path = output_path.parent / f"{output_path.stem}_prediction.nii.gz"
        return run_single(
            input_path=input_path,
            output_path=output_path,
            device=args.device,
            ensemble=args.ensemble,
            do_localisation=args.do_localisation,
        )

    # --- Directory / batch mode ---
    if not input_path.is_dir():
        print(f"ERROR: Not a file or directory: {input_path}", file=sys.stderr)
        return 1

    nifti_files = collect_nifti_files(input_path)
    if not nifti_files:
        print(f"No .nii / .nii.gz files found in {input_path}", file=sys.stderr)
        return 1

    job_dir = args.output.resolve() if args.output else (input_path / "blast_ct_output")

    print(f"\n{'═' * 60}")
    print(f"  BATCH MODE: {len(nifti_files)} file(s) found in {input_path}")
    print(f"{'═' * 60}")

    return run_batch(
        nifti_files=nifti_files,
        job_dir=job_dir,
        device=args.device,
        ensemble=args.ensemble,
        do_localisation=args.do_localisation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
