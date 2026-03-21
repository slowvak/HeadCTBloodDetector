import argparse
import nibabel as nib
import numpy as np
import json
import sys

def compute_volumes(input_file):
    """
    Computes the volume of each unique label in a NIfTI file.
    """
    try:
        # Load the NIfTI image
        img = nib.load(input_file)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)
        
    # Get voxel spacing (x, y, z dimensions)
    # img.header.get_zooms() usually returns a tuple of voxel sizes in mm
    zooms = img.header.get_zooms()
    
    # Calculate volume of a single voxel (typically in cubic millimeters, mm^3)
    if len(zooms) >= 3:
        voxel_volume = zooms[0] * zooms[1] * zooms[2]
    else:
        # Fallback if less than 3D
        voxel_volume = np.prod(zooms)
        
    # Get image data as integers (labels are typically integers)
    data = img.get_fdata().astype(int)
    
    # Find unique labels and their corresponding voxel counts
    unique_labels, counts = np.unique(data, return_counts=True)
    
    results = {}
    for label, count in zip(unique_labels, counts):
        # Calculate total volume for this label
        volume_mm3 = count * voxel_volume
        # Convert mm^3 to cm^3 (cc)
        volume_cm3 = volume_mm3 / 1000.0
        
        results[str(label)] = {
            "voxel_count": int(count),
            "volume_mm3": float(volume_mm3),
            "volume_cm3": float(volume_cm3)
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compute the volume of each unique label in a NIfTI segmentation file (*.nii.gz)."
    )
    parser.add_argument("input_file", help="Path to the input NIfTI file")
    parser.add_argument("--json", "-j", help="Optional path to output the results as a JSON file", default=None)
    parser.add_argument("--csv", "-c", help="Optional path to output the results as a CSV file", default=None)
    
    args = parser.parse_args()
    
    stats = compute_volumes(args.input_file)
    
    # Print results in a formatted table
    print(f"\nImage: {args.input_file}")
    print(f"{'Label':<10} | {'Voxel Count':<15} | {'Volume (mm³)':<15} | {'Volume (cm³ / cc)':<15}")
    print("-" * 65)
    
    # Sort labels for display (convert string keys back to int for proper sorting)
    sorted_labels = sorted(stats.keys(), key=lambda x: int(x))
    
    for label in sorted_labels:
        data = stats[label]
        print(f"{label:<10} | {data['voxel_count']:<15} | {data['volume_mm3']:<15.2f} | {data['volume_cm3']:<15.4f}")
        
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nResults saved to {args.json}")
        
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'Voxel Count', 'Volume (mm³)', 'Volume (cm³ / cc)'])
            for label in sorted_labels:
                data = stats[label]
                writer.writerow([label, data['voxel_count'], f"{data['volume_mm3']:.2f}", f"{data['volume_cm3']:.4f}"])
        print(f"\nResults saved to {args.csv}")

if __name__ == "__main__":
    main()
