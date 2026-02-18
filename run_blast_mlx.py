#!/usr/bin/env python3
"""Run blast-ct-mlx on all clean .nii.gz files in a folder.

Selects files ending in .nii.gz that have NO underscore in the filename.
Predictions are written to <folder>/blast_output/.

Usage:
    uv run python run_blast_mlx.py /path/to/folder
    uv run python run_blast_mlx.py ~/Desktop/CQ500_NII
"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder>")
        return 1

    folder = Path(sys.argv[1]).expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}")
        return 1

    files = sorted(
        p for p in folder.iterdir()
        if p.name.endswith(".nii.gz") and "_" not in p.stem.replace(".nii", "")
    )

    if not files:
        print(f"No matching .nii.gz files (without underscores) found in {folder}")
        return 1

    output_dir = folder / "blast_output"
    output_dir.mkdir(exist_ok=True)

    total = len(files)
    print(f"Found {total} file(s) → output: {output_dir}\n")

    failed = []
    for i, nii in enumerate(files, 1):
        stem = nii.name[: -len(".nii.gz")]
        out = output_dir / f"{stem}_prediction.nii.gz"
        print(f"[{i}/{total}] {nii.name}")

        result = subprocess.run(
            ["uv", "run", "blast-ct-mlx",
             "--input", str(nii),
             "--output", str(out)],
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode})\n")
            failed.append(nii.name)
        else:
            print(f"  → {out.name}\n")

    print(f"Done: {total - len(failed)}/{total} succeeded.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
