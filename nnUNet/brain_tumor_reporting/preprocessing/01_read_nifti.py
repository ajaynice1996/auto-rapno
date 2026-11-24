
import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brain_tumor_reporting.config.dataset_config import (
    IMAGES_DIR,
    LABELS_DIR,
    CHANNELS,
    SEQUENCE_DESCRIPTIONS,
    LABELS,
    FILE_ENDING,
    MANIFEST_DIR,
)


def get_subject_id(filename):
    """
    Extract subject ID from an nnUNet image filename.

    Example:
        BraTSPED_00003_0000.nii.gz
        -> BraTSPED_00003
    """

    name = filename

    if name.endswith(".nii.gz"):
        name = name[:-7]

    match = re.match(r"^(.*)_(000[0-3])$", name)

    if match is None:
        return None

    return match.group(1)


def get_nifti_information(filepath):
    """
    Read NIfTI metadata without loading the complete image
    into memory.
    """

    image = nib.load(str(filepath))

    return {
        "shape": list(image.shape),
        "ndim": len(image.shape),
        "voxel_spacing": [
            float(x) for x in image.header.get_zooms()[:3]
        ],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
        "datatype": str(image.get_data_dtype()),
        "file_size_mb": round(
            filepath.stat().st_size / (1024 ** 2), 3
        ),
    }


def inspect_segmentation(filepath):
    """
    Identify labels present in the segmentation mask.
    """

    image = nib.load(str(filepath))
    data = np.asanyarray(image.dataobj)

    unique_values = np.unique(data)

    integer_values = np.all(
        np.isclose(unique_values, np.round(unique_values))
    )

    labels_found = [
        int(x) for x in unique_values
        if np.isclose(x, round(float(x)))
    ]

    unexpected_labels = [
        x for x in labels_found
        if x not in LABELS
    ]

    return {
        "labels_found": labels_found,
        "unexpected_labels": unexpected_labels,
        "integer_labels": bool(integer_values),
    }


def validate_geometry(reference, current, name):
    """
    Compare dimensions, affine and voxel spacing.
    """

    errors = []

    if reference["shape"] != current["shape"]:
        errors.append(f"{name}: dimension mismatch")

    if not np.allclose(
        reference["affine"],
        current["affine"],
        atol=1e-4,
        rtol=1e-5,
    ):
        errors.append(f"{name}: affine mismatch")

    if not np.allclose(
        reference["voxel_spacing"],
        current["voxel_spacing"],
        atol=1e-4,
    ):
        errors.append(f"{name}: voxel spacing mismatch")

    if reference["orientation"] != current["orientation"]:
        errors.append(f"{name}: orientation mismatch")

    return errors


def discover_subjects(images_dir):
    """
    Discover subjects using the four nnUNet channel suffixes.
    """

    subjects = {}

    for filepath in sorted(images_dir.glob(f"*{FILE_ENDING}")):

        subject_id = get_subject_id(filepath.name)

        if subject_id is None:
            print(f"[WARNING] Unrecognized image filename: {filepath.name}")
            continue

        suffix = filepath.name[:-7].split("_")[-1]

        if suffix not in CHANNELS:
            continue

        sequence = CHANNELS[suffix]

        if subject_id not in subjects:
            subjects[subject_id] = {}

        subjects[subject_id][sequence] = filepath

    return subjects


def process_subject(subject_id, sequence_files, labels_dir):

    print(f"\nProcessing: {subject_id}")

    result = {
        "subject_id": subject_id,
        "images": {},
        "segmentation": None,
        "validation": {
            "missing_sequences": [],
            "errors": [],
            "warnings": [],
            "valid": False,
        },
    }

    # --------------------------------------------------
    # Check required sequences
    # --------------------------------------------------

    for sequence in CHANNELS.values():

        if sequence not in sequence_files:
            result["validation"]["missing_sequences"].append(
                sequence
            )

    if result["validation"]["missing_sequences"]:

        result["validation"]["errors"].append(
            "Missing MRI sequences"
        )

    # --------------------------------------------------
    # Read MRI sequence metadata
    # --------------------------------------------------

    reference_geometry = None

    for sequence, filepath in sequence_files.items():

        try:

            info = get_nifti_information(filepath)

            if info["ndim"] != 3:
                result["validation"]["errors"].append(
                    f"{sequence}: expected 3D image, "
                    f"found {info['ndim']} dimensions"
                )

            if any(x <= 0 for x in info["voxel_spacing"]):
                result["validation"]["errors"].append(
                    f"{sequence}: invalid voxel spacing"
                )

            result["images"][sequence] = {
                "path": str(filepath.resolve()),
                "description": SEQUENCE_DESCRIPTIONS[sequence],
                **info,
            }

            if reference_geometry is None:
                reference_geometry = info

            else:
                errors = validate_geometry(
                    reference_geometry,
                    info,
                    sequence,
                )

                result["validation"]["errors"].extend(errors)

        except Exception as error:

            result["validation"]["errors"].append(
                f"{sequence}: failed to read image: {error}"
            )

    # --------------------------------------------------
    # Read corresponding segmentation
    # --------------------------------------------------

    segmentation_path = labels_dir / (
        subject_id + FILE_ENDING
    )

    if not segmentation_path.exists():

        result["validation"]["errors"].append(
            "Segmentation mask missing"
        )

    else:

        try:

            seg_info = get_nifti_information(
                segmentation_path
            )

            seg_labels = inspect_segmentation(
                segmentation_path
            )

            result["segmentation"] = {
                "path": str(segmentation_path.resolve()),
                **seg_info,
                **seg_labels,
            }

            if seg_info["ndim"] != 3:
                result["validation"]["errors"].append(
                    "Segmentation must be a 3D image"
                )

            if not seg_labels["integer_labels"]:
                result["validation"]["errors"].append(
                    "Segmentation contains non-integer labels"
                )

            if seg_labels["unexpected_labels"]:
                result["validation"]["errors"].append(
                    "Unexpected segmentation labels found: "
                    + str(seg_labels["unexpected_labels"])
                )

            if reference_geometry is not None:

                errors = validate_geometry(
                    reference_geometry,
                    seg_info,
                    "segmentation",
                )

                result["validation"]["errors"].extend(errors)

        except Exception as error:

            result["validation"]["errors"].append(
                f"Segmentation reading failed: {error}"
            )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    result["validation"]["valid"] = (
        len(result["validation"]["errors"]) == 0
    )

    status = (
        "VALID"
        if result["validation"]["valid"]
        else "CHECK REQUIRED"
    )

    print(f"Status: {status}")

    if result["validation"]["errors"]:

        for error in result["validation"]["errors"]:
            print("  ERROR:", error)

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Read and validate Dataset202_BraTS MRI data"
    )

    parser.add_argument(
        "--images_dir",
        type=Path,
        default=IMAGES_DIR,
    )

    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=LABELS_DIR,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=MANIFEST_DIR,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Validate directories
    # --------------------------------------------------

    if not args.images_dir.exists():
        raise FileNotFoundError(args.images_dir)

    if not args.labels_dir.exists():
        raise FileNotFoundError(args.labels_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Discover subjects
    # --------------------------------------------------

    subjects = discover_subjects(args.images_dir)

    print("\n===================================")
    print("Dataset202_BraTS")
    print("===================================")

    print("Discovered subjects:", len(subjects))

    # --------------------------------------------------
    # Process all subjects
    # --------------------------------------------------

    results = []

    for subject_id, sequence_files in subjects.items():

        result = process_subject(
            subject_id,
            sequence_files,
            args.labels_dir,
        )

        results.append(result)

    # --------------------------------------------------
    # Save JSON manifest
    # --------------------------------------------------

    manifest_path = (
        args.output_dir / "dataset_manifest.json"
    )

    with open(manifest_path, "w") as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Save CSV validation report
    # --------------------------------------------------

    rows = []

    for result in results:

        row = {
            "subject_id": result["subject_id"],
            "valid": result["validation"]["valid"],
            "missing_sequences": ",".join(
                result["validation"]["missing_sequences"]
            ),
            "errors": " | ".join(
                result["validation"]["errors"]
            ),
        }

        for sequence in CHANNELS.values():

            row[f"{sequence}_available"] = (
                sequence in result["images"]
            )

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    csv_path = (
        args.output_dir / "dataset_validation.csv"
    )

    dataframe.to_csv(csv_path, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    valid_count = sum(
        r["validation"]["valid"] for r in results
    )

    invalid_count = len(results) - valid_count

    print("\n===================================")
    print("DATASET SUMMARY")
    print("===================================")

    print("Total subjects:", len(results))
    print("Valid subjects:", valid_count)
    print("Subjects requiring review:", invalid_count)

    print("\nSaved:")
    print(manifest_path)
    print(csv_path)


if __name__ == "__main__":
    main()
