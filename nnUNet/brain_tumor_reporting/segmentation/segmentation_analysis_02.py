
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from brain_tumor_reporting.config.dataset_config import (
    LABELS_DIR,
    OUTPUT_DIR,
    LABELS,
)


# ============================================================
# 1. CASE IDENTIFICATION
# ============================================================

def get_case_id(filename):
    """Extract subject ID from NIfTI filename or nnU-Net split entry."""

    name = str(filename)

    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    # Remove nnU-Net channel suffix, e.g. _0000.
    if len(name) >= 5 and name[-5] == "_" and name[-4:].isdigit():
        name = name[:-5]

    return name


# ============================================================
# 2. SUBJECT SELECTION
# ============================================================

def select_subjects(
    segmentation_files,
    subject_id=None,
    fold=None,
    split_file=None,
    max_subjects=None,
):
    """Select one subject, fold validation subjects, or all masks."""

    files_by_id = {
        get_case_id(path.name): path
        for path in segmentation_files
    }

    if subject_id is not None:

        requested_id = get_case_id(subject_id)

        if requested_id not in files_by_id:
            raise FileNotFoundError(
                f"Subject not found: {requested_id}"
            )

        selected = [files_by_id[requested_id]]

    elif fold is not None:

        if split_file is None or not split_file.exists():
            raise FileNotFoundError(
                f"nnU-Net split file not found: {split_file}"
            )

        with open(split_file, "r") as file:
            splits = json.load(file)

        if fold < 0 or fold >= len(splits):
            raise ValueError(
                f"Invalid fold {fold}. "
                f"Available folds: 0 to {len(splits) - 1}"
            )

        validation_ids = splits[fold].get("val", [])

        selected = []
        missing_ids = []

        for case in validation_ids:

            case_id = get_case_id(case)

            if case_id in files_by_id:
                selected.append(files_by_id[case_id])
            else:
                missing_ids.append(case_id)

        if missing_ids:
            print(
                f"\nWARNING: {len(missing_ids)} validation "
                "subjects have no corresponding mask."
            )
            print("Missing examples:", missing_ids[:10])

        print(f"\nFold {fold} validation cases: {len(validation_ids)}")
        print(f"Matching segmentation masks: {len(selected)}")

    else:
        selected = sorted(segmentation_files)

    if max_subjects is not None:

        if max_subjects < 1:
            raise ValueError("--max_subjects must be greater than zero.")

        selected = selected[:max_subjects]

    return selected


# ============================================================
# 3. TUMOR COMPONENT MASKS
# ============================================================

def get_component_masks(segmentation):
    """
    Dataset label mapping:
        0 = Background
        1 = ET
        2 = NET
        3 = CC
        4 = ED

    TC = ET + NET + CC
    WT = ET + NET + CC + ED
    """

    return {
        "ET": segmentation == 1,
        "NET": segmentation == 2,
        "CC": segmentation == 3,
        "ED": segmentation == 4,
        "TC": np.isin(segmentation, [1, 2, 3]),
        "WT": np.isin(segmentation, [1, 2, 3, 4]),
    }


# ============================================================
# 4. PHYSICAL MEASUREMENTS
# ============================================================

def calculate_voxel_volume(affine):
    """Calculate physical voxel volume in mm3."""

    return abs(float(np.linalg.det(affine[:3, :3])))


def calculate_centroid(mask, affine):
    """Calculate tumor centroid in NIfTI RAS+ world coordinates."""

    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return None

    voxel_centroid = coordinates.mean(axis=0)

    world_centroid = nib.affines.apply_affine(
        affine,
        voxel_centroid
    )

    return {
        "x_RAS_mm": float(world_centroid[0]),
        "y_RAS_mm": float(world_centroid[1]),
        "z_RAS_mm": float(world_centroid[2]),
    }


def calculate_bounding_box(mask, affine):
    """
    Calculate bounding-box dimensions along the three voxel axes.

    These are voxel-axis dimensions in millimeters, not the
    maximum Euclidean tumor diameter.
    """

    coordinates = np.where(mask)

    if len(coordinates[0]) == 0:
        return None

    minimum = np.array([
        axis.min() for axis in coordinates
    ])

    maximum = np.array([
        axis.max() for axis in coordinates
    ])

    voxel_dimensions = maximum - minimum + 1

    axis_spacing = np.linalg.norm(
        affine[:3, :3],
        axis=0
    )

    dimensions_mm = voxel_dimensions * axis_spacing

    return {
        "dimension_1_mm": float(dimensions_mm[0]),
        "dimension_2_mm": float(dimensions_mm[1]),
        "dimension_3_mm": float(dimensions_mm[2]),
    }


# ============================================================
# 5. CONNECTED COMPONENT ANALYSIS
# ============================================================

def count_connected_components(mask):
    """Count 3D connected components using 26-connectivity."""

    structure = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    _, number = ndimage.label(
        mask,
        structure=structure
    )

    return int(number)


# ============================================================
# 6. ANALYZE ONE SUBJECT
# ============================================================

def analyze_subject(segmentation_path):

    subject_id = get_case_id(segmentation_path.name)

    print(f"\nProcessing: {subject_id}")

    # Load NIfTI segmentation.
    image = nib.load(str(segmentation_path))

    segmentation = np.asanyarray(image.dataobj)
    affine = image.affine

    # Basic data validation.
    if segmentation.ndim != 3:
        raise ValueError(
            f"Expected 3D segmentation, got {segmentation.shape}"
        )

    if not np.all(np.isfinite(segmentation)):
        raise ValueError(
            "Segmentation contains non-finite values."
        )

    if not np.allclose(
        segmentation,
        np.round(segmentation)
    ):
        raise ValueError(
            "Segmentation contains non-integer labels."
        )

    segmentation = np.rint(
        segmentation
    ).astype(np.int16)

    # Validate label values.
    observed_labels = np.unique(segmentation)

    allowed_labels = np.array(
        list(LABELS.keys())
    )

    unexpected = np.setdiff1d(
        observed_labels,
        allowed_labels
    )

    if len(unexpected) > 0:
        raise ValueError(
            f"Unexpected labels: {unexpected.tolist()}"
        )

    # Calculate voxel volume.
    voxel_volume = calculate_voxel_volume(affine)

    if voxel_volume <= 0:
        raise ValueError("Invalid voxel volume.")

    # Generate masks.
    masks = get_component_masks(segmentation)

    # Calculate component voxel counts.
    counts = {
        name: int(mask.sum())
        for name, mask in masks.items()
    }

    # Internal arithmetic validation.
    expected_tc = counts["ET"] + counts["NET"] + counts["CC"]
    expected_wt = expected_tc + counts["ED"]

    if counts["TC"] != expected_tc:
        raise ValueError("TC volume arithmetic check failed.")

    if counts["WT"] != expected_wt:
        raise ValueError("WT volume arithmetic check failed.")

    # Physical whole tumor volume.
    whole_tumor_volume_mm3 = (
        counts["WT"] * voxel_volume
    )

    # Subject-level results.
    subject_results = {
        "subject_id": subject_id,
        "segmentation_file": str(
            segmentation_path.resolve()
        ),
        "shape": list(segmentation.shape),
        "orientation": list(nib.aff2axcodes(affine)),
        "voxel_volume_mm3": voxel_volume,
        "observed_labels": observed_labels.tolist(),
        "tumor_present": bool(counts["WT"] > 0),
        "component_arithmetic_valid": True,
        "components": {},
    }

    csv_records = []

    # Calculate measurements for each component.
    for component_name, mask in masks.items():

        voxel_count = counts[component_name]

        volume_mm3 = voxel_count * voxel_volume
        volume_cm3 = volume_mm3 / 1000.0

        percentage_of_wt = (
            100.0 * volume_mm3 / whole_tumor_volume_mm3
            if whole_tumor_volume_mm3 > 0
            else 0.0
        )

        centroid = calculate_centroid(
            mask,
            affine
        )

        bounding_box = calculate_bounding_box(
            mask,
            affine
        )

        connected_components = count_connected_components(
            mask
        )

        component_results = {
            "voxel_count": voxel_count,
            "volume_mm3": float(volume_mm3),
            "volume_cm3": float(volume_cm3),
            "percentage_of_WT": float(percentage_of_wt),
            "connected_components": connected_components,
            "centroid": centroid,
            "bounding_box": bounding_box,
        }

        subject_results["components"][
            component_name
        ] = component_results

        csv_records.append({
            "subject_id": subject_id,
            "component": component_name,
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "volume_cm3": volume_cm3,
            "percentage_of_WT": percentage_of_wt,
            "connected_components": connected_components,

            "centroid_x_RAS_mm": (
                centroid["x_RAS_mm"]
                if centroid else None
            ),

            "centroid_y_RAS_mm": (
                centroid["y_RAS_mm"]
                if centroid else None
            ),

            "centroid_z_RAS_mm": (
                centroid["z_RAS_mm"]
                if centroid else None
            ),

            "dimension_1_mm": (
                bounding_box["dimension_1_mm"]
                if bounding_box else None
            ),

            "dimension_2_mm": (
                bounding_box["dimension_2_mm"]
                if bounding_box else None
            ),

            "dimension_3_mm": (
                bounding_box["dimension_3_mm"]
                if bounding_box else None
            ),
        })

    print(
        f"WT volume: {whole_tumor_volume_mm3 / 1000:.3f} cm3"
    )

    return subject_results, csv_records


# ============================================================
# 7. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Quantitative nnU-Net brain tumor segmentation analysis"
        )
    )

    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=LABELS_DIR,
        help="Directory containing segmentation NIfTI files"
    )

    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Analyze one subject, e.g. BraTSPED_00003"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Analyze validation subjects from a zero-based nnU-Net fold"
    )

    parser.add_argument(
        "--split_file",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "nnUNet_preprocessed_all"
            / "Dataset202_BraTS"
            / "splits_final.json"
        ),
        help="Path to nnU-Net splits_final.json"
    )

    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Maximum number of selected subjects to process"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional custom output directory"
    )

    args = parser.parse_args()

    if args.subject_id is not None and args.fold is not None:
        parser.error(
            "Use either --subject_id or --fold, not both."
        )

    if not args.labels_dir.exists():
        raise FileNotFoundError(
            f"Segmentation directory not found: {args.labels_dir}"
        )

    segmentation_files = sorted(
        args.labels_dir.glob("*.nii.gz")
    )

    if not segmentation_files:
        raise FileNotFoundError(
            f"No NIfTI masks found in {args.labels_dir}"
        )

    # Select subjects.
    selected_files = select_subjects(
        segmentation_files=segmentation_files,
        subject_id=args.subject_id,
        fold=args.fold,
        split_file=args.split_file,
        max_subjects=args.max_subjects
    )

    if not selected_files:
        raise RuntimeError(
            "No subjects selected for analysis."
        )

    # Determine output directory.
    if args.output_dir is not None:

        output_dir = args.output_dir

    elif args.subject_id is not None:

        output_dir = (
            OUTPUT_DIR
            / "segmentation_analysis"
            / f"subject_{get_case_id(args.subject_id)}"
        )

    elif args.fold is not None:

        output_dir = (
            OUTPUT_DIR
            / "segmentation_analysis"
            / f"fold_{args.fold}_validation"
        )

    elif args.max_subjects is not None:

        output_dir = (
            OUTPUT_DIR
            / "segmentation_analysis"
            / f"first_{args.max_subjects}_subjects"
        )

    else:

        output_dir = (
            OUTPUT_DIR / "segmentation_analysis"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n===================================")
    print("SEGMENTATION ANALYSIS")
    print("===================================")

    print(f"Selected subjects: {len(selected_files)}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {output_dir}")

    # Process subjects.
    all_subject_results = []
    all_csv_records = []
    failed_subjects = []

    for segmentation_path in selected_files:

        try:

            result, records = analyze_subject(
                segmentation_path
            )

            all_subject_results.append(result)
            all_csv_records.extend(records)

        except Exception as error:

            print(
                f"ERROR: {segmentation_path.name}: {error}"
            )

            failed_subjects.append({
                "subject_id": get_case_id(
                    segmentation_path.name
                ),
                "error": str(error),
            })

    # Save detailed JSON.
    json_path = (
        output_dir / "segmentation_metrics.json"
    )

    with open(json_path, "w") as file:

        json.dump(
            all_subject_results,
            file,
            indent=4
        )

    # Save component-level CSV.
    csv_path = (
        output_dir / "segmentation_metrics.csv"
    )

    columns = [
        "subject_id",
        "component",
        "voxel_count",
        "volume_mm3",
        "volume_cm3",
        "percentage_of_WT",
        "connected_components",
        "centroid_x_RAS_mm",
        "centroid_y_RAS_mm",
        "centroid_z_RAS_mm",
        "dimension_1_mm",
        "dimension_2_mm",
        "dimension_3_mm",
    ]

    pd.DataFrame(
        all_csv_records,
        columns=columns
    ).to_csv(
        csv_path,
        index=False
    )

    # Save failures.
    failure_path = (
        output_dir / "failed_subjects.json"
    )

    with open(failure_path, "w") as file:

        json.dump(
            failed_subjects,
            file,
            indent=4
        )

    # Save execution summary.
    summary = {
        "selected_subjects": len(selected_files),
        "successful_subjects": len(all_subject_results),
        "failed_subjects": len(failed_subjects),
        "tumor_present_subjects": sum(
            result["tumor_present"]
            for result in all_subject_results
        ),
        "subjects_without_tumor": sum(
            not result["tumor_present"]
            for result in all_subject_results
        ),
    }

    summary_path = output_dir / "summary.json"

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=4)

    print("\n===================================")
    print("ANALYSIS SUMMARY")
    print("===================================")

    print(f"Selected subjects: {summary['selected_subjects']}")
    print(f"Successful: {summary['successful_subjects']}")
    print(f"Failed: {summary['failed_subjects']}")
    print(f"Tumor present: {summary['tumor_present_subjects']}")
    print(f"No tumor: {summary['subjects_without_tumor']}")

    print("\nSaved files:")
    print(json_path)
    print(csv_path)
    print(failure_path)
    print(summary_path)


if __name__ == "__main__":
    main()
