#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_ROOT = SCRIPT_DIR.parent
NNUNET_ROOT = SCRIPT_DIR.parents[3]
REPORTING_ROOT = NNUNET_ROOT / "brain_tumor_reporting"


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_text(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def relative_to_case(path, case_dir):
    return str(path.relative_to(case_dir))


def find_image(case_dir, filename):
    matches = list(case_dir.rglob(filename))
    return matches[0] if matches else None


def make_image_entry(path, case_dir, category, reason, **details):
    return {
        "path": relative_to_case(path, case_dir),
        "category": category,
        "reason_for_selection": reason,
        **details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare structured case data and image inventory for Gemini."
    )
    parser.add_argument(
        "--subject",
        default="BraTSPED_00002",
        help="Subject identifier (default: BraTSPED_00002)",
    )
    args = parser.parse_args()

    subject = args.subject
    case_dir = GEMINI_ROOT / subject
    input_dir = case_dir / "llm_inputs"
    prepared_dir = case_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    quantitative_dir = input_dir / "04_quantitative_data"
    original_dir = input_dir / "01_original_mri"
    segmentation_dir = input_dir / "02_tumor_segmentation"
    anatomy_dir = input_dir / "03_anatomical_visualizations"

    overlap_json_path = (
        quantitative_dir / f"{subject}_tumor_anatomical_overlap.json"
    )
    top_regions_json_path = (
        quantitative_dir / f"{subject}_top_affected_regions.json"
    )
    overlap_csv_path = (
        quantitative_dir / f"{subject}_tumor_anatomical_overlap.csv"
    )
    top_regions_csv_path = (
        quantitative_dir / f"{subject}_top_affected_regions.csv"
    )
    qc_csv_path = quantitative_dir / f"{subject}_qc.csv"
    volumes_csv_path = quantitative_dir / f"{subject}_volumes.csv"
    validation_path = quantitative_dir / f"{subject}_validation_report.txt"

    required_files = [
        overlap_json_path,
        top_regions_json_path,
        overlap_csv_path,
        qc_csv_path,
        validation_path,
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required staged input files are missing:\n"
            + "\n".join(missing)
        )

    overlap_data = load_json(overlap_json_path)
    top_regions_data = load_json(top_regions_json_path)
    overlap_rows = load_csv(overlap_csv_path)
    top_regions_rows = load_csv(top_regions_csv_path)
    qc_rows = load_csv(qc_csv_path)
    validation_text = load_text(validation_path)

    # ---------------------------------------------------------
    # Tumor component volumes from the overlap JSON.
    # ---------------------------------------------------------
    component_data = overlap_data.get("tumor_components", {})
    component_summary = {}

    for component, values in component_data.items():
        component_summary[component] = {
            "label_values": values.get("tumor_label_values", []),
            "voxel_count": values.get("voxel_count"),
            "volume_mm3": values.get("volume_mm3"),
            "volume_ml": values.get("volume_ml"),
        }

    # Basic arithmetic checks for the segmentation labels.
    consistency_checks = {}

    if all(k in component_summary for k in ("ET", "NET", "CC", "TC", "ED", "WT")):
        et = component_summary["ET"]["volume_mm3"] or 0
        net = component_summary["NET"]["volume_mm3"] or 0
        cc = component_summary["CC"]["volume_mm3"] or 0
        ed = component_summary["ED"]["volume_mm3"] or 0
        tc = component_summary["TC"]["volume_mm3"] or 0
        wt = component_summary["WT"]["volume_mm3"] or 0

        consistency_checks = {
            "TC_equals_ET_plus_NET_plus_CC": abs(tc - (et + net + cc)) < 1e-6,
            "WT_equals_TC_plus_ED": abs(wt - (tc + ed)) < 1e-6,
            "note": "TC and WT are composite totals; do not add them to their component volumes.",
        }

    # ---------------------------------------------------------
    # Top anatomical overlaps, retaining the two percentage
    # definitions as separate fields.
    # ---------------------------------------------------------
    top_overlap_summary = {}

    for component in component_summary:
        rows = [
            row for row in overlap_rows
            if row.get("tumor_component") == component
        ]

        def numeric(row, key):
            try:
                return float(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        rows.sort(
            key=lambda r: numeric(r, "overlap_volume_ml"),
            reverse=True,
        )

        top_overlap_summary[component] = [
            {
                "anatomy_label_id": row.get("anatomy_label_id"),
                "anatomy_label_name": row.get("anatomy_label_name"),
                "overlap_volume_ml": numeric(row, "overlap_volume_ml"),
                "percent_tumor_in_anatomy": numeric(
                    row, "percent_tumor_in_anatomy"
                ),
                "percent_anatomy_occupied": numeric(
                    row, "percent_anatomy_occupied"
                ),
            }
            for row in rows[:5]
            if numeric(row, "overlap_volume_ml") > 0
        ]

    # Preserve the precomputed top-region JSON as an additional source.
    precomputed_top_regions = {}
    for component, values in top_regions_data.items():
        precomputed_top_regions[component] = values.get(
            "top_5_by_overlap_volume", []
        )

    # ---------------------------------------------------------
    # SynthSeg regional QC scores.
    # ---------------------------------------------------------
    qc_summary = None
    if qc_rows:
        # The file is expected to contain one row for this subject.
        matching_rows = [
            row for row in qc_rows
            if subject in str(row.get("subject", ""))
        ]
        selected_qc = matching_rows[0] if matching_rows else qc_rows[0]
        qc_summary = {
            "source_file": relative_to_case(qc_csv_path, case_dir),
            "subject_row": selected_qc,
            "interpretation": (
                "These are SynthSeg regional quality-control scores, "
                "not diagnostic measures or tumor-confidence scores."
            ),
        }

    # ---------------------------------------------------------
    # Alignment and slice metadata.
    # ---------------------------------------------------------
    metadata_files = [
        original_dir / f"{subject}_alignment_and_slice_metadata.json",
        segmentation_dir / f"{subject}_alignment_and_slice_metadata.json",
    ]

    alignment_metadata = {}
    for path in metadata_files:
        if path.exists():
            alignment_metadata[path.parent.name] = {
                "source_file": relative_to_case(path, case_dir),
                "data": load_json(path),
            }

    # ---------------------------------------------------------
    # Image inventory: include every PNG under llm_inputs.
    # ---------------------------------------------------------
    image_manifest = []

    for path in sorted(input_dir.rglob("*.png")):
        rel = path.relative_to(input_dir)
        parts = rel.parts

        entry = {
            "path": relative_to_case(path, case_dir),
            "filename": path.name,
            "category": parts[0] if parts else "other",
            "file_size_bytes": path.stat().st_size,
        }

        if len(parts) >= 2 and parts[0] == "02_tumor_segmentation":
            entry["component"] = parts[1]
        if len(parts) >= 3 and parts[0] == "02_tumor_segmentation":
            entry["sequence"] = parts[2]

        image_manifest.append(entry)

    # ---------------------------------------------------------
    # Curated initial image set.
    # All images remain in the inventory, but only a subset
    # is selected for the first Gemini report-generation run.
    # ---------------------------------------------------------
    selected_images = []

    original_image = find_image(
        case_dir,
        f"{subject}_matched_axial_comparison_v3.png",
    )
    if original_image:
        selected_images.append(
            make_image_entry(
                original_image,
                case_dir,
                "original_mri",
                "Matched axial comparison across the four MRI sequences.",
            )
        )

    anatomy_images = [
        (
            f"{subject}_combined_tumor_anatomy.png",
            "Combined tumor and anatomical overview.",
        ),
        (
            f"{subject}_synthseg_validation.png",
            "SynthSeg anatomical segmentation validation overview.",
        ),
        (
            f"{subject}_visual_qc.png",
            "Multiplanar visual QC of tumor and anatomical segmentation.",
        ),
    ]

    for filename, reason in anatomy_images:
        path = find_image(case_dir, filename)
        if path:
            selected_images.append(
                make_image_entry(
                    path,
                    case_dir,
                    "anatomical_visualization",
                    reason,
                )
            )

    # Select T1c axial and multiplanar overlays for non-empty
    # primary components. CC is excluded because its measured
    # volume is zero. TC is omitted because it is a composite.
    selected_components = ["ET", "NET", "ED", "WT"]
    selected_views = ["axial_overlay.png", "multiplanar_3x3_overlay.png"]

    for component in selected_components:
        volume = component_summary.get(component, {}).get("volume_ml", 0) or 0
        if volume <= 0:
            continue

        for view_suffix in selected_views:
            filename = f"{subject}_{component}_T1c_{view_suffix}"
            path = find_image(case_dir, filename)

            if path:
                selected_images.append(
                    make_image_entry(
                        path,
                        case_dir,
                        "tumor_segmentation",
                        (
                            f"{component} segmentation overlay on T1c; "
                            f"measured component volume {volume:.3f} mL."
                        ),
                        component=component,
                        sequence="T1c",
                        view=view_suffix.replace("_overlay.png", ""),
                    )
                )

    # ---------------------------------------------------------
    # Case summary with provenance.
    # ---------------------------------------------------------
    case_summary = {
        "subject_id": subject,
        "purpose": (
            "Research-draft brain MRI report generation using four MRI "
            "sequences, tumor segmentation, anatomical overlap, and QC data."
        ),
        "demographics": {
            "age": None,
            "sex": None,
            "note": "No verified subject-level demographic data supplied.",
        },
        "segmentation_label_definitions": {
            "0": "Background",
            "1": "ET",
            "2": "NET",
            "3": "CC",
            "4": "ED",
            "TC": "Union of labels 1, 2, and 3",
            "WT": "Union of labels 1, 2, 3, and 4",
        },
        "tumor_component_summary": component_summary,
        "component_volume_consistency_checks": consistency_checks,
        "top_anatomical_overlaps_from_csv": top_overlap_summary,
        "precomputed_top_affected_regions": precomputed_top_regions,
        "anatomical_overlap_interpretation_note": overlap_data.get(
            "interpretation_note"
        ),
        "reference_grid": overlap_data.get("reference_grid"),
        "synthseg_label_ids_present": overlap_data.get(
            "synthseg_label_ids_present"
        ),
        "synthseg_label_ids_missing_from_lut": overlap_data.get(
            "synthseg_label_ids_missing_from_lut"
        ),
        "synthseg_regional_qc": qc_summary,
        "alignment_and_slice_metadata": alignment_metadata,
        "validation_report_text": validation_text,
        "source_files": {
            "tumor_anatomical_overlap_json": relative_to_case(
                overlap_json_path, case_dir
            ),
            "tumor_anatomical_overlap_csv": relative_to_case(
                overlap_csv_path, case_dir
            ),
            "top_affected_regions_json": relative_to_case(
                top_regions_json_path, case_dir
            ),
            "top_affected_regions_csv": relative_to_case(
                top_regions_csv_path, case_dir
            ),
            "synthseg_qc_csv": relative_to_case(qc_csv_path, case_dir),
            "synthseg_volumes_csv": (
                relative_to_case(volumes_csv_path, case_dir)
                if volumes_csv_path.exists()
                else None
            ),
            "validation_report": relative_to_case(
                validation_path, case_dir
            ),
        },
        "reporting_guardrails": [
            "Do not infer tumor histology, molecular subtype, or grade.",
            "Do not infer enhancement solely from the ET segmentation label.",
            "Do not describe anatomical spatial overlap as histological invasion or infiltration.",
            "Distinguish tumor component volumes from composite TC and WT volumes.",
            "Do not interpret SynthSeg regional QC scores as diagnostic confidence.",
            "Do not invent age, sex, symptoms, clinical history, or pathology.",
            "Treat the output as a research draft requiring radiologist review.",
        ],
    }

    # ---------------------------------------------------------
    # Write outputs.
    # ---------------------------------------------------------
    summary_path = prepared_dir / "case_summary.json"
    manifest_path = prepared_dir / "image_manifest.json"
    selected_path = prepared_dir / "gemini_selected_images.json"

    summary_path.write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": subject,
                "image_count": len(image_manifest),
                "images": image_manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected_path.write_text(
        json.dumps(
            {
                "subject_id": subject,
                "selection_note": (
                    "Curated initial image set. The complete image inventory "
                    "is stored separately in image_manifest.json."
                ),
                "selected_image_count": len(selected_images),
                "images": selected_images,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("CASE DATA PREPARATION COMPLETE")
    print("=" * 65)
    print(f"Subject:                 {subject}")
    print(f"Component groups:        {len(component_summary)}")
    print(f"Images inventoried:      {len(image_manifest)}")
    print(f"Images selected:         {len(selected_images)}")
    print(f"TC/WT consistency:       {consistency_checks}")
    print("\nTumor component volumes:")
    for name, values in component_summary.items():
        print(
            f"  {name:4s}: "
            f"{values.get('volume_ml', 0):.3f} mL "
            f"({values.get('voxel_count', 0)} voxels)"
        )

    print("\nCreated:")
    print(f"  {summary_path}")
    print(f"  {manifest_path}")
    print(f"  {selected_path}")

    print("\nReview the selected image list before sending anything to Gemini.")


if __name__ == "__main__":
    main()
