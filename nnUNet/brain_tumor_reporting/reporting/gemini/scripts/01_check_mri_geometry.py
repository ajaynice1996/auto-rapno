#!/usr/bin/env python3

"""
Inspect MRI geometry for a single BraTS-PEDs subject.

Checks:
- File existence
- Image dimensions
- Orientation
- Voxel spacing
- Data type
- Shape consistency
- Affine consistency

This script does not modify any input files.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[5]

SUBJECT = "BraTSPED_00002"

DATASET_DIR = (
    PROJECT_ROOT
    / "nnUNet"
    / "nnUNet_raw"
    / "Dataset202_BraTS"
)

MRI_DIR = DATASET_DIR / "imagesTr_all"
MASK_DIR = DATASET_DIR / "labelsTr_all"


FILES = {
    "T1n": MRI_DIR / f"{SUBJECT}_0000.nii.gz",
    "T1c": MRI_DIR / f"{SUBJECT}_0001.nii.gz",
    "T2f": MRI_DIR / f"{SUBJECT}_0002.nii.gz",
    "T2w": MRI_DIR / f"{SUBJECT}_0003.nii.gz",
    "Tumor mask": MASK_DIR / f"{SUBJECT}.nii.gz",
}


# ============================================================
# GEOMETRY INSPECTION
# ============================================================

def main():

    print("=" * 70)
    print("MRI GEOMETRY INSPECTION")
    print("=" * 70)

    print(f"Subject: {SUBJECT}")
    print(f"Dataset: {DATASET_DIR}")

    reference_shape = None
    reference_affine = None

    results = []

    for name, path in FILES.items():

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        if not path.exists():
            print(f"ERROR: File not found: {path}")
            results.append((name, False, False))
            continue

        image = nib.load(str(path))

        shape = image.shape
        affine = image.affine

        orientation = nib.aff2axcodes(affine)
        voxel_sizes = nib.affines.voxel_sizes(affine)

        print(f"File: {path}")
        print(f"Shape: {shape}")
        print(f"Orientation: {orientation}")
        print(f"Voxel sizes (mm): {voxel_sizes}")
        print(f"Data type: {image.get_data_dtype()}")

        if reference_shape is None:

            reference_shape = shape
            reference_affine = affine

            same_shape = True
            same_affine = True

            print("Reference geometry established.")

        else:

            same_shape = shape == reference_shape
            same_affine = np.allclose(
                affine,
                reference_affine,
                atol=1e-5
            )

            print(f"Same shape as reference: {same_shape}")
            print(f"Same affine as reference: {same_affine}")

        results.append((name, same_shape, same_affine))

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("GEOMETRY SUMMARY")
    print("=" * 70)

    for name, same_shape, same_affine in results:

        print(
            f"{name:15s} | "
            f"Shape match: {same_shape!s:5s} | "
            f"Affine match: {same_affine!s:5s}"
        )

    all_files_present = all(
        path.exists() for path in FILES.values()
    )

    all_geometry_match = (
        all_files_present
        and all(same_shape and same_affine for _, same_shape, same_affine in results)
    )

    print("\n" + "=" * 70)

    if all_geometry_match:
        print("RESULT: All files share matching geometry.")
        print("Direct corresponding slice selection is appropriate.")
    else:
        print("RESULT: Geometry differences or missing files detected.")
        print("Check image registration/resampling before montage generation.")

    print("=" * 70)


if __name__ == "__main__":
    main()
