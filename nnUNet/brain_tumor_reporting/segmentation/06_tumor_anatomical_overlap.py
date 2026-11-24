

#!/usr/bin/env python3
"""
Tumor–Anatomy Spatial Overlap Analysis using SynthSeg.

Features:
1. ET, NET, CC, ED, TC and WT analysis.
2. Main anatomical structures and cortical parcellation.
3. Nearest-neighbour alignment to original T1 reference grid.
4. Overlap volume in mm3 and mL.
5. Percentage of anatomical region occupied by tumor.
6. Percentage of tumor component overlapping each anatomical region.
7. Top 5 and Top 10 affected regions for each tumor component:
   - Ranked by absolute overlap volume (mm3).
   - Ranked by relative anatomical involvement (%).
8. Subject-level CSV and JSON reports.

Run from nnUNet project root:

Single subject:
python brain_tumor_reporting/segmentation/06_tumor_anatomical_overlap.py \
    --subject BraTSPED_00002

All subjects:
python brain_tumor_reporting/segmentation/06_tumor_anatomical_overlap.py --all

IMPORTANT:
Spatial overlap does not establish histological tumor infiltration.
Visual quality control is required, particularly in tumor-distorted anatomy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

T1_DIR = (
    PROJECT_ROOT
    / "nnUNet_raw"
    / "Dataset202_BraTS"
    / "imagesTr_all"
)

TUMOR_DIR = (
    PROJECT_ROOT
    / "nnUNet_raw"
    / "Dataset202_BraTS"
    / "labelsTr_all"
)

SYNTHSEG_DIR = (
    PROJECT_ROOT
    / "brain_tumor_reporting"
    / "outputs"
    / "synthseg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "brain_tumor_reporting"
    / "outputs"
    / "tumor_anatomical_overlap"
)

MAIN_LUT = (
    PROJECT_ROOT
    / "external"
    / "SynthSeg"
    / "data"
    / "labels table.txt"
)

CORTICAL_LUT = (
    PROJECT_ROOT
    / "external"
    / "SynthSeg"
    / "data"
    / "synthseg_cortical_labels.txt"
)


# ============================================================
# 2. TUMOR LABEL DEFINITIONS
# ============================================================

TUMOR_COMPONENTS = {
    "ET": (1,),
    "NET": (2,),
    "CC": (3,),
    "ED": (4,),
    "TC": (1, 2, 3),
    "WT": (1, 2, 3, 4),
}


# ============================================================
# 3. READ LOOKUP TABLES
# ============================================================



# ============================================================
# 3. READ LOOKUP TABLES
# ============================================================

def parse_main_lut(path: Path) -> dict[int, str]:
    """
    Parse SynthSeg main LUT.

    Format:
    Label_ID    Structure_Name

    Structure names may contain spaces.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Main lookup table not found: {path}"
        )

    labels = {}

    with path.open("r", encoding="utf-8") as file:

        for line_number, raw_line in enumerate(file, start=1):

            line = raw_line.strip()

            if not line:
                continue

            fields = line.split(maxsplit=1)

            if len(fields) != 2:
                continue

            try:
                label_id = int(fields[0])
            except ValueError:
                # Skip explanatory text and table headers.
                continue

            label_name = fields[1].strip()

            if label_id in labels and labels[label_id] != label_name:
                raise ValueError(
                    f"Conflicting main LUT label ID {label_id} "
                    f"in {path}, line {line_number}"
                )

            labels[label_id] = label_name

    return labels


def parse_cortical_lut(path: Path) -> dict[int, str]:
    """
    Parse FreeSurfer-style cortical LUT.

    Format:
    Label_ID  Label_Name  R  G  B  A
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Cortical lookup table not found: {path}"
        )

    labels = {}

    with path.open("r", encoding="utf-8") as file:

        for line_number, raw_line in enumerate(file, start=1):

            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 6:
                continue

            try:
                label_id = int(fields[0])
                rgba = [int(value) for value in fields[-4:]]
            except ValueError:
                continue

            label_name = " ".join(fields[1:-4])

            if not label_name:
                continue

            if label_id in labels and labels[label_id] != label_name:
                raise ValueError(
                    f"Conflicting cortical LUT label ID {label_id} "
                    f"in {path}, line {line_number}"
                )

            labels[label_id] = label_name

    return labels


def load_anatomy_lut() -> dict[int, str]:
    """
    Combine main SynthSeg anatomical labels and cortical labels.
    Exclude unknown cortical labels 1000 and 2000.
    """

    anatomy = parse_main_lut(MAIN_LUT)
    cortical = parse_cortical_lut(CORTICAL_LUT)

    for label_id, label_name in cortical.items():

        if (
            label_id in anatomy
            and anatomy[label_id] != label_name
        ):
            raise ValueError(
                f"Label conflict for ID {label_id}: "
                f"{anatomy[label_id]} vs {label_name}"
            )

        anatomy[label_id] = label_name

    anatomy.pop(1000, None)
    anatomy.pop(2000, None)

    return anatomy



# ============================================================
# 4. FIND INPUT FILES
# ============================================================

def find_one_file(
    directory: Path,
    exact_name: str,
    pattern: str,
    description: str
) -> Path:

    exact_path = directory / exact_name

    if exact_path.is_file():
        return exact_path

    matches = sorted(directory.glob(pattern))

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {description} in {directory}.\n"
            f"Expected: {exact_name}"
        )

    raise RuntimeError(
        f"Multiple possible {description} files found:\n"
        + "\n".join(str(p) for p in matches)
    )


# ============================================================
# 5. ALIGN LABEL MAPS TO T1 REFERENCE
# ============================================================

def load_on_reference_grid(
    path: Path,
    reference,
    description: str
) -> np.ndarray:

    img = nib.load(str(path))

    if len(img.shape) != 3:
        raise ValueError(
            f"{description} must be 3D. "
            f"Received shape: {img.shape}"
        )

    same_shape = img.shape == reference.shape

    same_affine = np.allclose(
        img.affine,
        reference.affine,
        atol=1e-4
    )

    if same_shape and same_affine:

        aligned = img

    else:

        print(
            f"  Resampling {description}: "
            f"{img.shape} -> {reference.shape}"
        )

        aligned = resample_from_to(
            img,
            (reference.shape, reference.affine),
            order=0
        )

    data = np.asarray(aligned.dataobj)

    return np.rint(data).astype(np.int32)


# ============================================================
# 6. TOP AFFECTED REGION RANKING
# ============================================================

def get_top_affected_regions(rows):
    """
    Rank regions independently for each tumor component.

    Ranking A:
        Absolute overlap volume (mm3).

    Ranking B:
        Relative anatomical involvement (% of anatomical
        region occupied by tumor).

    Only regions with nonzero overlap are included.
    """

    results = {}

    for component in TUMOR_COMPONENTS:

        component_rows = [
            row for row in rows
            if row["tumor_component"] == component
            and row["overlap_voxels"] > 0
        ]

        # ----------------------------------------
        # Ranking A: Absolute overlap volume
        # ----------------------------------------

        by_overlap_volume = sorted(
            component_rows,
            key=lambda x: x["overlap_volume_mm3"],
            reverse=True
        )

        # ----------------------------------------
        # Ranking B: Relative anatomical involvement
        # ----------------------------------------

        by_region_percentage = sorted(
            component_rows,
            key=lambda x: x["percent_anatomy_occupied"],
            reverse=True
        )

        results[component] = {

            "top_5_by_overlap_volume": (
                by_overlap_volume[:5]
            ),

            "top_10_by_overlap_volume": (
                by_overlap_volume[:10]
            ),

            "top_5_by_percent_anatomy_occupied": (
                by_region_percentage[:5]
            ),

            "top_10_by_percent_anatomy_occupied": (
                by_region_percentage[:10]
            ),
        }

    return results


# ============================================================
# 7. EXPORT RANKED CSV
# ============================================================

def save_top_regions_csv(
    top_regions: dict,
    output_path: Path
):

    fields = [
        "tumor_component",
        "ranking_metric",
        "rank",
        "anatomy_label_id",
        "anatomy_label_name",
        "overlap_volume_mm3",
        "overlap_volume_ml",
        "percent_tumor_in_anatomy",
        "percent_anatomy_occupied",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for component, rankings in top_regions.items():

            ranking_groups = [
                (
                    "absolute_overlap_volume_mm3",
                    rankings["top_10_by_overlap_volume"]
                ),
                (
                    "relative_anatomical_involvement_percent",
                    rankings[
                        "top_10_by_percent_anatomy_occupied"
                    ]
                ),
            ]

            for metric_name, ranked_regions in ranking_groups:

                for rank, region in enumerate(
                    ranked_regions,
                    start=1
                ):

                    writer.writerow({

                        "tumor_component": component,

                        "ranking_metric": metric_name,

                        "rank": rank,

                        "anatomy_label_id": (
                            region["anatomy_label_id"]
                        ),

                        "anatomy_label_name": (
                            region["anatomy_label_name"]
                        ),

                        "overlap_volume_mm3": (
                            region["overlap_volume_mm3"]
                        ),

                        "overlap_volume_ml": (
                            region["overlap_volume_ml"]
                        ),

                        "percent_tumor_in_anatomy": (
                            region["percent_tumor_in_anatomy"]
                        ),

                        "percent_anatomy_occupied": (
                            region["percent_anatomy_occupied"]
                        ),
                    })

def json_safe(obj):
    """Convert NumPy types into standard Python JSON-compatible types."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    return obj

# ============================================================
# 8. PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    subject_id: str,
    anatomy_lut: dict[int, str]
):

    print("\n" + "=" * 72)
    print(f"SUBJECT: {subject_id}")
    print("=" * 72)

    # ----------------------------------------
    # Input files
    # ----------------------------------------

    t1_path = find_one_file(
        T1_DIR,
        f"{subject_id}_0000.nii.gz",
        f"{subject_id}_0000.nii*",
        "T1 image"
    )

    tumor_path = find_one_file(
        TUMOR_DIR,
        f"{subject_id}.nii.gz",
        f"{subject_id}*.nii*",
        "Tumor segmentation"
    )

    synthseg_path = (
        SYNTHSEG_DIR
        / f"{subject_id}_synthseg.nii.gz"
    )

    if not synthseg_path.is_file():
        raise FileNotFoundError(
            f"SynthSeg prediction not found: {synthseg_path}"
        )

    # ----------------------------------------
    # T1 reference
    # ----------------------------------------

    reference = nib.load(str(t1_path))

    if len(reference.shape) != 3:
        raise ValueError(
            f"Expected 3D T1 image, got {reference.shape}"
        )

    voxel_volume_mm3 = float(
        np.asarray(
            abs(np.linalg.det(reference.affine[:3, :3]))
        ).item()
        )

    if (
        not np.isfinite(voxel_volume_mm3)
        or voxel_volume_mm3 <= 0
    ):
        raise ValueError("Invalid T1 voxel volume.")

    # ----------------------------------------
    # Load aligned segmentations
    # ----------------------------------------

    tumor = load_on_reference_grid(
        tumor_path,
        reference,
        "Tumor segmentation"
    )

    anatomy = load_on_reference_grid(
        synthseg_path,
        reference,
        "SynthSeg segmentation"
    )

    # ----------------------------------------
    # Validate tumor labels
    # ----------------------------------------

    tumor_values = set(np.unique(tumor).tolist())

    unexpected_values = sorted(
        tumor_values - {0, 1, 2, 3, 4}
    )

    if unexpected_values:

        print(
            "WARNING: Unexpected tumor labels:",
            unexpected_values
        )

    # ----------------------------------------
    # Identify anatomical labels
    # ----------------------------------------

    present_anatomy_ids = sorted(
        int(x)
        for x in np.unique(anatomy)
        if int(x) != 0
    )

    unknown_ids = [
        label_id
        for label_id in present_anatomy_ids
        if label_id not in anatomy_lut
    ]

    if unknown_ids:

        print(
            "WARNING: SynthSeg labels absent from LUT:",
            unknown_ids
        )

    regions = [
        (label_id, anatomy_lut[label_id])
        for label_id in present_anatomy_ids
        if label_id in anatomy_lut
    ]

    # ----------------------------------------
    # Calculate overlap
    # ----------------------------------------

    rows = []

    component_summary = {}

    for component, tumor_label_values in TUMOR_COMPONENTS.items():

        tumor_mask = np.isin(
            tumor,
            tumor_label_values
        )

        tumor_voxels = int(
            np.count_nonzero(tumor_mask)
        )

        tumor_volume = (
            tumor_voxels * voxel_volume_mm3
        )

        component_summary[component] = {

            "tumor_label_values": list(
                tumor_label_values
            ),

            "voxel_count": tumor_voxels,

            "volume_mm3": round(
                tumor_volume, 3
            ),

            "volume_ml": round(
                tumor_volume / 1000.0, 4
            ),
        }

        for anatomy_id, anatomy_name in regions:

            anatomy_mask = (
                anatomy == anatomy_id
            )

            anatomy_voxels = int(
                np.count_nonzero(anatomy_mask)
            )

            overlap_voxels = int(
                np.count_nonzero(
                    tumor_mask & anatomy_mask
                )
            )

            overlap_volume = (
                overlap_voxels * voxel_volume_mm3
            )

            anatomy_volume = (
                anatomy_voxels * voxel_volume_mm3
            )

            percent_anatomy_occupied = (
                100.0 * overlap_voxels / anatomy_voxels
                if anatomy_voxels else 0.0
            )

            percent_tumor_in_anatomy = (
                100.0 * overlap_voxels / tumor_voxels
                if tumor_voxels else 0.0
            )

            rows.append({

                "subject_id": subject_id,

                "tumor_component": component,

                "tumor_label_values": ";".join(
                    map(str, tumor_label_values)
                ),

                "anatomy_label_id": anatomy_id,

                "anatomy_label_name": anatomy_name,

                "tumor_voxels": tumor_voxels,

                "tumor_volume_mm3": round(
                    tumor_volume, 3
                ),

                "anatomy_voxels": anatomy_voxels,

                "anatomy_volume_mm3": round(
                    anatomy_volume, 3
                ),

                "overlap_voxels": overlap_voxels,

                "overlap_volume_mm3": round(
                    overlap_volume, 3
                ),

                "overlap_volume_ml": round(
                    overlap_volume / 1000.0, 6
                ),

                "percent_anatomy_occupied": round(
                    percent_anatomy_occupied, 4
                ),

                "percent_tumor_in_anatomy": round(
                    percent_tumor_in_anatomy, 4
                ),
            })

    # ========================================================
    # 9. OUTPUT DIRECTORY
    # ========================================================

    subject_out = OUTPUT_DIR / subject_id

    subject_out.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Full overlap CSV
    # ----------------------------------------

    csv_path = (
        subject_out
        / f"{subject_id}_tumor_anatomical_overlap.csv"
    )

    if rows:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    # ========================================================
    # 10. TOP 5 AND TOP 10 REGIONS
    # ========================================================

    top_regions = get_top_affected_regions(rows)

    # ----------------------------------------
    # Top regions JSON
    # ----------------------------------------

    top_json_path = (
        subject_out
        / f"{subject_id}_top_affected_regions.json"
    )

    top_json_path.write_text(
    json.dumps(
        json_safe(top_regions),
        indent=2
    ),
    encoding="utf-8"
    )

    # ----------------------------------------
    # Top regions CSV
    # ----------------------------------------

    top_csv_path = (
        subject_out
        / f"{subject_id}_top_affected_regions.csv"
    )

    save_top_regions_csv(
        top_regions,
        top_csv_path
    )

    # ========================================================
    # 11. MAIN SUBJECT JSON REPORT
    # ========================================================

    report = {

        "subject_id": subject_id,

        "input_files": {

            "t1_reference": str(t1_path),

            "tumor_segmentation": str(tumor_path),

            "synthseg_segmentation": str(synthseg_path),

            "main_anatomical_lut": str(MAIN_LUT),

            "cortical_parcellation_lut": str(CORTICAL_LUT),
        },

        "reference_grid": {

            "shape": list(reference.shape),

            "voxel_volume_mm3": voxel_volume_mm3,

            "voxel_spacing_mm": list(
                reference.header.get_zooms()[:3]
            ),
        },

        "synthseg_label_ids_present": (
            present_anatomy_ids
        ),

        "synthseg_label_ids_missing_from_lut": (
            unknown_ids
        ),

        "anatomical_regions_in_report": [

            {
                "label_id": label_id,
                "label_name": label_name
            }

            for label_id, label_name in regions
        ],

        "tumor_components": component_summary,

        "top_affected_regions": top_regions,

        "interpretation_note": (
            "These are voxel-wise spatial overlaps after "
            "nearest-neighbour alignment to the T1 reference grid. "
            "Spatial overlap does not establish histological tumor "
            "infiltration. Edema and tumor-related mass effect may "
            "distort anatomical segmentation boundaries. Visual "
            "quality control is required."
        ),
    }

    json_path = (
        subject_out
        / f"{subject_id}_tumor_anatomical_overlap.json"
    )

    json_path.write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8"
    )

    # ========================================================
    # 12. TERMINAL SUMMARY
    # ========================================================

    print("\nTUMOR COMPONENT VOLUMES")

    for component, values in component_summary.items():

        print(
            f"  {component:5s}: "
            f"{values['volume_ml']:.3f} mL"
        )

    print("\nTOP 5 REGIONS BY ABSOLUTE OVERLAP VOLUME")

    for component, rankings in top_regions.items():

        print(f"\n{component}:")

        for rank, region in enumerate(
            rankings["top_5_by_overlap_volume"],
            start=1
        ):

            print(
                f"  {rank:2d}. "
                f"{region['anatomy_label_name']:35s} "
                f"{region['overlap_volume_mm3']:10.2f} mm3 "
                f"({region['percent_anatomy_occupied']:.2f}% region)"
            )

    print("\nTOP 5 REGIONS BY RELATIVE ANATOMICAL INVOLVEMENT")

    for component, rankings in top_regions.items():

        print(f"\n{component}:")

        for rank, region in enumerate(
            rankings[
                "top_5_by_percent_anatomy_occupied"
            ],
            start=1
        ):

            print(
                f"  {rank:2d}. "
                f"{region['anatomy_label_name']:35s} "
                f"{region['percent_anatomy_occupied']:8.2f}% "
                f"({region['overlap_volume_mm3']:.2f} mm3)"
            )

    print("\nOUTPUT FILES")

    print(f"  Full overlap CSV: {csv_path}")
    print(f"  Main JSON report: {json_path}")
    print(f"  Top regions CSV:  {top_csv_path}")
    print(f"  Top regions JSON: {top_json_path}")

    return report


# ============================================================
# 13. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate tumor overlap with SynthSeg "
            "anatomical and cortical regions."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--subject",
        type=str,
        help="Subject ID, e.g. BraTSPED_00002"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Process all available subjects"
    )

    args = parser.parse_args()

    print("\nTUMOR–ANATOMY SPATIAL OVERLAP ANALYSIS")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Main LUT: {MAIN_LUT}")
    print(f"Cortical LUT: {CORTICAL_LUT}")
    print(f"Output: {OUTPUT_DIR}")

    # ----------------------------------------
    # Load lookup tables
    # ----------------------------------------

    try:

        anatomy_lut = load_anatomy_lut()

    except Exception as exc:

        print(
            f"\nERROR loading lookup tables: {exc}",
            file=sys.stderr
        )

        return 1

    print(
        f"\nLoaded {len(anatomy_lut)} "
        "anatomical/cortical label definitions."
    )

    # ----------------------------------------
    # Select subjects
    # ----------------------------------------

    if args.all:

        subjects = sorted(

            path.name[:-len("_0000.nii.gz")]

            for path in T1_DIR.glob(
                "*_0000.nii.gz"
            )
        )

    else:

        subjects = [args.subject]

    if not subjects:

        print(
            f"No T1 subjects found in {T1_DIR}",
            file=sys.stderr
        )

        return 1

    print(f"Selected subjects: {len(subjects)}")

    # ----------------------------------------
    # Process subjects
    # ----------------------------------------

    successful = []
    failed = []

    for subject_id in subjects:

        try:

            process_subject(
                subject_id,
                anatomy_lut
            )

            successful.append(subject_id)

        except Exception as exc:

            print(
                f"\n[FAILED] {subject_id}: {exc}",
                file=sys.stderr
            )

            failed.append({

                "subject_id": subject_id,

                "error": str(exc)
            })

    # ----------------------------------------
    # Batch summary
    # ----------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        OUTPUT_DIR / "batch_summary.json"
    )

    summary = {

        "successful_count": len(successful),

        "failed_count": len(failed),

        "successful_subjects": successful,

        "failed_subjects": failed,
    }

    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 72)

    print("PIPELINE SUMMARY")

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    print(f"Summary: {summary_path}")

    return 0 if not failed else 2


if __name__ == "__main__":

    raise SystemExit(main())
