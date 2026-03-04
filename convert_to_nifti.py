#!/usr/bin/env python3
"""
Convert DICOM to NIfTI

For each subfolder in the input directory, looks for a 'NII' subdirectory
containing DICOM files. Converts those DICOMs to a single NIfTI file
using the dicom2nifti library.

Output filename: <subfolder_name>.nii.gz
Output location: user-specified output folder

Usage:
    python convert_to-nifti.py <input_folder> <output_folder>

Requirements:
    pip install dicom2nifti
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    import pydicom
except ImportError:
    print("Error: pydicom is required. Install with: pip install pydicom")
    sys.exit(1)

try:
    import dicom2nifti
    import dicom2nifti.settings as settings
except ImportError:
    print("Error: dicom2nifti is required. Install with: pip install dicom2nifti")
    sys.exit(1)


def fix_orientation_and_direction(nifti_path: str):
    """Reorient a NIfTI image to RAS and ensure orthonormal direction cosines.
    
    1. Reorients image data and affine to RAS (Right-Anterior-Superior)
       so that axes map correctly: X=R/L, Y=A/P, Z=S/I.
    2. Orthonormalises the 3×3 rotation via SVD so ITK-based tools
       (ITK-SNAP, ANTs, FreeSurfer, etc.) accept the file.
    """
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

    img = nib.load(nifti_path)

    # --- Step 1: Reorient to RAS ---
    orig_ornt = io_orientation(img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    img = img.as_reoriented(transform)

    # --- Step 2: Orthonormalise direction cosines ---
    affine = img.affine.copy()
    rot_scale = affine[:3, :3]
    voxel_sizes = np.sqrt(np.sum(rot_scale ** 2, axis=0))  # column norms

    rotation = rot_scale / voxel_sizes

    # Nearest orthonormal matrix via SVD:  R_orth = U @ V^T
    U, _, Vt = np.linalg.svd(rotation)
    rotation_orth = U @ Vt

    # Preserve original handedness (det should stay the same sign)
    if np.linalg.det(rotation_orth) < 0:
        U[:, -1] *= -1
        rotation_orth = U @ Vt

    affine[:3, :3] = rotation_orth * voxel_sizes

    new_img = nib.Nifti1Image(img.get_fdata(), affine, img.header)
    nib.save(new_img, nifti_path)



def convert_dicom_manual(dicom_dir: str, output_path: str):
    """Fallback DICOM-to-NIfTI converter using pydicom + nibabel.

    Handles cases where dicom2nifti fails (e.g. signed pixel data stored
    as uint16 with RescaleIntercept = -1024).  Reads every DICOM slice,
    applies RescaleSlope/Intercept, builds the 3-D volume, constructs a
    proper affine from the DICOM geometry tags, and saves as NIfTI.
    """
    # --- Read and sort slices ---
    slices = []
    for f in Path(dicom_dir).iterdir():
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, 'ImagePositionPatient'):
                slices.append(ds)
        except Exception:
            continue

    if len(slices) < 2:
        raise ValueError(f"Only {len(slices)} usable DICOM slices found")

    # Sort by slice position along the normal
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # --- Build pixel volume with Hounsfield units ---
    ref = slices[0]
    rows = int(ref.Rows)
    cols = int(ref.Columns)
    volume = np.zeros((rows, cols, len(slices)), dtype=np.float32)

    for i, s in enumerate(slices):
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, 'RescaleSlope', 1))
        intercept = float(getattr(s, 'RescaleIntercept', 0))
        volume[:, :, i] = arr * slope + intercept

    # Store as int16 (sufficient for CT Hounsfield range -1024 to +3071)
    volume = np.clip(volume, -32768, 32767).astype(np.int16)

    # --- Build affine from DICOM geometry ---
    ipp = np.array([float(v) for v in ref.ImagePositionPatient])
    iop = [float(v) for v in ref.ImageOrientationPatient]
    row_cos = np.array(iop[:3])
    col_cos = np.array(iop[3:])

    pixel_spacing = [float(v) for v in ref.PixelSpacing]
    dr, dc = pixel_spacing[0], pixel_spacing[1]

    # Slice direction from first-to-last position
    last_ipp = np.array([float(v) for v in slices[-1].ImagePositionPatient])
    slice_vec = (last_ipp - ipp) / max(len(slices) - 1, 1)

    affine = np.eye(4)
    affine[:3, 0] = row_cos * dr
    affine[:3, 1] = col_cos * dc
    affine[:3, 2] = slice_vec
    affine[:3, 3] = ipp

    # DICOM is LPS, NIfTI is RAS -> negate first two rows
    affine[0, :] = -affine[0, :]
    affine[1, :] = -affine[1, :]

    img = nib.Nifti1Image(volume, affine)
    nib.save(img, output_path)


def convert_folder(input_dir: str, output_dir: str):
    """Convert DICOM files in NII subfolders to NIfTI format.
    
    Args:
        input_dir: Parent directory containing exam subfolders, each with a 'NII' subfolder.
        output_dir: Directory to write the output .nii.gz files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Disable some validations that can fail on clinical data
    settings.disable_validate_slice_increment()
    settings.disable_validate_orthogonal()
    
    subs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    total = len(subs)
    converted = 0
    skipped = 0
    failed = 0
    
    print(f"Scanning {total} subfolders in: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    for idx, sub_dir in enumerate(subs, 1):
        nii_dir = sub_dir / 'NII'
        exam_name = sub_dir.name
        output_file = output_path / f"{exam_name}.nii.gz"
        
        if not nii_dir.is_dir():
            print(f"  [{idx}/{total}] {exam_name}: No NII folder found — skipping")
            skipped += 1
            continue
        
        # Check that NII folder has files
        dicom_files = [f for f in nii_dir.iterdir() if f.is_file()]
        if not dicom_files:
            print(f"  [{idx}/{total}] {exam_name}: NII folder is empty — skipping")
            skipped += 1
            continue
        
        # Skip if output already exists
        # if output_file.exists():
        #     print(f"  [{idx}/{total}] {exam_name}: Output already exists — skipping")
        #     skipped += 1
        #     continue
        
        try:
            print(f"  [{idx}/{total}] {exam_name}: Converting {len(dicom_files)} DICOMs...", end=" ", flush=True)
            try:
                dicom2nifti.dicom_series_to_nifti(str(nii_dir), str(output_file), reorient_nifti=True)
            except Exception:
                # Fallback: manual conversion (handles uint16 / signed-pixel issues)
                convert_dicom_manual(str(nii_dir), str(output_file))
            # Fix direction cosines so ITK-based tools accept the file
            fix_orientation_and_direction(str(output_file))
            print("OK")
            converted += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python convert_to-nifti.py <input_folder> <output_folder>")
    #     print("")
    #     print("  input_folder:  Parent directory with exam subfolders (each containing a 'NII' subfolder)")
    #     print("  output_folder: Directory to write .nii.gz output files")
    #     sys.exit(1)
    
    # input_folder = sys.argv[1]
    # output_folder = sys.argv[2]

    input_folder = '/Volumes/OWC Express 1M2/Images/CQ500_Brain_Hemorrhage_Dataset/CT_DICOMs'
    output_folder = '/Users/bje01/Desktop/CQ500_NII'
        
    convert_folder(input_folder, output_folder)
