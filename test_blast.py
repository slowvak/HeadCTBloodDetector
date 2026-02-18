#!/usr/bin/env python3
# TEMPORARY TEST — run blast-ct on all CQ500-CT-*.nii.gz scans

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from blast import main

DATA_DIR = Path("/Users/bje01/Desktop/CQ500_NII")
OUTPUT_DIR = DATA_DIR / "blast_ct_output"

if __name__ == "__main__":
    sys.exit(main([
        "--input", str(DATA_DIR),
        "--output", str(OUTPUT_DIR),
        "--device", "cpu",
    ]))
