#!/usr/bin/env python3
"""
run_synthstrip.py
─────────────────
Read a NIfTI file and apply FreeSurfer SynthStrip (via Docker) to perform
skull-stripping.

Prerequisites
─────────────
1. Docker must be installed and running.
2. Download the synthstrip-docker wrapper once:
       curl -O https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/synthstrip-docker
       chmod +x synthstrip-docker
   Place it somewhere on your PATH, or pass its location with --synthstrip-cmd.

Usage
─────
    python run_synthstrip.py -i /path/to/input.nii.gz
    python run_synthstrip.py -i input.nii.gz -o stripped.nii.gz -m mask.nii.gz
    python run_synthstrip.py -i input.nii.gz --gpu          # use GPU mode
    python run_synthstrip.py -i input.nii.gz --no-csf       # exclude CSF
    python run_synthstrip.py --input-dir /path/to/folder    # batch mode
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_synthstrip(cmd: str) -> str:
    """Return the resolved path to the synthstrip-docker script, or exit."""
    resolved = shutil.which(cmd)
    if resolved is None:
        # Try current working directory as a fallback
        local = Path.cwd() / cmd
        if local.is_file() and os.access(local, os.X_OK):
            return str(local)
        print(
            f"ERROR: Could not find '{cmd}'.\n"
            "Download it with:\n"
            "  curl -O https://raw.githubusercontent.com/freesurfer/freesurfer/"
            "dev/mri_synthstrip/synthstrip-docker\n"
            "  chmod +x synthstrip-docker\n"
            "Then place it on your PATH or pass --synthstrip-cmd /path/to/synthstrip-docker",
            file=sys.stderr,
        )
        sys.exit(1)
    return resolved


def build_output_paths(
    input_path: Path,
    output_path: Path | None,
    mask_path: Path | None,
) -> tuple[Path, Path]:
    """Derive default output and mask paths when not explicitly provided."""
    stem = input_path.name
    # Handle double extensions like .nii.gz
    if stem.endswith(".nii.gz"):
        base = stem[: -len(".nii.gz")]
        ext = ".nii.gz"
    else:
        base = input_path.stem
        ext = input_path.suffix

    parent = input_path.parent

    if output_path is None:
        output_path = parent / f"{base}_stripped{ext}"
    if mask_path is None:
        mask_path = parent / f"{base}_mask{ext}"

    return output_path, mask_path


def run_synthstrip(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    synthstrip_cmd: str = "synthstrip-docker",
    gpu: bool = False,
    no_csf: bool = False,
    border: int | None = None,
) -> subprocess.CompletedProcess:
    """
    Call synthstrip-docker on the given NIfTI file.

    Parameters
    ----------
    input_path : Path
        Input NIfTI file (.nii or .nii.gz).
    output_path : Path
        Where to write the skull-stripped image.
    mask_path : Path
        Where to write the binary brain mask.
    synthstrip_cmd : str
        Path or name of the synthstrip-docker executable.
    gpu : bool
        If True, pass --gpu to enable GPU acceleration.
    no_csf : bool
        If True, pass --no-csf to exclude CSF from the brain mask.
    border : int or None
        If set, pass --border <mm> to control mask boundary.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = [
        synthstrip_cmd,
        "-i", str(input_path),
        "-o", str(output_path),
        "-m", str(mask_path),
    ]

    if gpu:
        cmd.append("--gpu")
    if no_csf:
        cmd.append("--no-csf")
    if border is not None:
        cmd.extend(["--border", str(border)])

    print(f"{'─' * 60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Mask   : {mask_path}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'─' * 60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(
            f"\nERROR: synthstrip-docker exited with code {result.returncode}",
            file=sys.stderr,
        )
        if result.returncode == 137:
            print(
                "  → Container ran out of memory. "
                "Try increasing RAM in Docker preferences.",
                file=sys.stderr,
            )
    else:
        print("\n✓ Skull-stripping complete.")
        print(f"  Stripped image : {output_path}")
        print(f"  Brain mask     : {mask_path}")

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skull-strip a NIfTI file using FreeSurfer SynthStrip (Docker).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_synthstrip.py -i brain.nii.gz\n"
            "  python run_synthstrip.py -i brain.nii.gz -o stripped.nii.gz -m mask.nii.gz\n"
            "  python run_synthstrip.py -i brain.nii.gz --gpu --no-csf\n"
            "  python run_synthstrip.py -i /path/to/nifti_folder\n"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input NIfTI file or directory of NIfTI files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path for the skull-stripped output (single-file mode only).",
    )
    parser.add_argument(
        "-m", "--mask",
        type=Path,
        default=None,
        help="Path for the brain-mask output (single-file mode only).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (requires nvidia-docker).",
    )
    parser.add_argument(
        "--no-csf",
        action="store_true",
        help="Exclude CSF from the brain border.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=None,
        help="Mask border threshold in mm (default: 1).",
    )
    parser.add_argument(
        "--synthstrip-cmd",
        type=str,
        default="synthstrip-docker",
        help="Path to the synthstrip-docker script (default: synthstrip-docker).",
    )
    return parser.parse_args(argv)


def collect_nifti_files(directory: Path) -> list[Path]:
    """Return sorted list of .nii / .nii.gz files in *directory*."""
    files = sorted(
        p for p in directory.iterdir()
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz")
    )
    # Exclude already-produced outputs (_stripped, _mask)
    files = [
        p for p in files
        if not any(tag in p.name for tag in ("_stripped", "_mask"))
    ]
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Resolve synthstrip command -------------------------------------------
    synthstrip_cmd = ensure_synthstrip(args.synthstrip_cmd)

    input_path = args.input.resolve()

    # --- Single-file mode -----------------------------------------------------
    if input_path.is_file():
        output_path, mask_path = build_output_paths(
            input_path, args.output, args.mask
        )
        result = run_synthstrip(
            input_path=input_path,
            output_path=output_path,
            mask_path=mask_path,
            synthstrip_cmd=synthstrip_cmd,
            gpu=args.gpu,
            no_csf=args.no_csf,
            border=args.border,
        )
        return result.returncode

    # --- Batch / directory mode -----------------------------------------------
    if not input_path.is_dir():
        print(f"ERROR: Not a file or directory: {input_path}", file=sys.stderr)
        return 1

    nifti_files = collect_nifti_files(input_path)
    if not nifti_files:
        print(f"No .nii / .nii.gz files found in {input_path}", file=sys.stderr)
        return 1

    total = len(nifti_files)
    succeeded, failed = 0, 0

    for idx, nii_path in enumerate(nifti_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{total}]  {nii_path.name}")
        print(f"{'═' * 60}")

        out_path, msk_path = build_output_paths(nii_path, None, None)

        result = run_synthstrip(
            input_path=nii_path,
            output_path=out_path,
            mask_path=msk_path,
            synthstrip_cmd=synthstrip_cmd,
            gpu=args.gpu,
            no_csf=args.no_csf,
            border=args.border,
        )
        if result.returncode == 0:
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"  BATCH COMPLETE: {succeeded} ok / {failed} failed  (total {total})")
    print(f"{'═' * 60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
