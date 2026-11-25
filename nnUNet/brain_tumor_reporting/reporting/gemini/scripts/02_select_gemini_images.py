#!/usr/bin/env python3

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_ROOT = SCRIPT_DIR.parent

SUBJECT = "BraTSPED_00002"
CASE_DIR = GEMINI_ROOT / SUBJECT
INPUT_DIR = CASE_DIR / "llm_inputs"
PREPARED_DIR = CASE_DIR / "prepared"

OUTPUT_FILE = PREPARED_DIR / "gemini_selected_images.json"

SEQUENCES = ["T1n", "T1c", "T2f", "T2w"]

# Main tumor components with multi-sequence selection.
MAIN_COMPONENTS = ["ET", "NET", "WT"]

# Small-volume component: retain cautiously for visual review.
SMALL_COMPONENT = "ED"


def add_image(selected, path, category, reason, **details):
    if not path.exists():
        return False

    selected.append({
        "path": str(path.relative_to(CASE_DIR)),
        "filename": path.name,
        "category": category,
        "reason_for_selection": reason,
        "file_size_bytes": path.stat().st_size,
        **details,
    })
    return True


def find_named_image(filename):
    matches = list(INPUT_DIR.rglob(filename))
    return matches[0] if matches else None


def main():
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    selected = []
    missing = []

    # --------------------------------------------------
    # 1. Original MRI images WITHOUT tumor overlays.
    # Include all PNGs in 01_original_mri.
    # --------------------------------------------------
    original_dir = INPUT_DIR / "01_original_mri"

    original_images = sorted(original_dir.rglob("*.png"))

    for path in original_images:
        add_image(
            selected,
            path,
            "original_unsegmented_mri",
            "Original MRI visualization without tumor segmentation overlay.",
        )

    if not original_images:
        missing.append(
            "No PNG images found in 01_original_mri/"
        )

    # --------------------------------------------------
    # 2. Overall anatomical visualizations and QC.
    # --------------------------------------------------
    overview_images = [
        (
            f"{SUBJECT}_combined_tumor_anatomy.png",
            "Combined tumor and anatomical overview.",
        ),
        (
            f"{SUBJECT}_synthseg_validation.png",
            "SynthSeg anatomical segmentation validation.",
        ),
        (
            f"{SUBJECT}_visual_qc.png",
            "Multiplanar tumor and anatomical segmentation QC.",
        ),
    ]

    for filename, reason in overview_images:
        path = find_named_image(filename)
        if path:
            add_image(
                selected,
                path,
                "anatomical_visualization",
                reason,
            )
        else:
            missing.append(filename)

    # --------------------------------------------------
    # 3. ET, NET and WT:
    # axial overlays for all four MRI sequences,
    # plus T1c multiplanar view.
    # --------------------------------------------------
    for component in MAIN_COMPONENTS:
        for sequence in SEQUENCES:
            filename = (
                f"{SUBJECT}_{component}_{sequence}_axial_overlay.png"
            )
            path = find_named_image(filename)

            if path:
                add_image(
                    selected,
                    path,
                    "tumor_segmentation_overlay",
                    (
                        f"{component} axial segmentation overlay on "
                        f"{sequence}."
                    ),
                    component=component,
                    sequence=sequence,
                    view="axial",
                )
            else:
                missing.append(filename)

        filename = (
            f"{SUBJECT}_{component}_T1c_multiplanar_3x3_overlay.png"
        )
        path = find_named_image(filename)

        if path:
            add_image(
                selected,
                path,
                "tumor_segmentation_overlay",
                f"{component} multiplanar segmentation overlay on T1c.",
                component=component,
                sequence="T1c",
                view="multiplanar_3x3",
            )
        else:
            missing.append(filename)

    # --------------------------------------------------
    # 4. ED: small-volume component.
    # Keep T1c axial and multiplanar views.
    # --------------------------------------------------
    for view_suffix, view_name in [
        ("axial_overlay.png", "axial"),
        ("multiplanar_3x3_overlay.png", "multiplanar_3x3"),
    ]:
        filename = (
            f"{SUBJECT}_{SMALL_COMPONENT}_T1c_{view_suffix}"
        )
        path = find_named_image(filename)

        if path:
            add_image(
                selected,
                path,
                "small_component_segmentation_overlay",
                (
                    "Small-volume ED component; interpret cautiously "
                    "and verify against the original MRI."
                ),
                component=SMALL_COMPONENT,
                sequence="T1c",
                view=view_name,
            )
        else:
            missing.append(filename)

    # --------------------------------------------------
    # 5. Remove duplicate paths, if any.
    # --------------------------------------------------
    unique = []
    seen = set()

    for item in selected:
        if item["path"] not in seen:
            unique.append(item)
            seen.add(item["path"])

    selected = unique

    # --------------------------------------------------
    # 6. Write selection manifest.
    # --------------------------------------------------
    result = {
        "subject_id": SUBJECT,
        "selection_strategy": {
            "original_unsegmented_mri": (
                "All PNG files found in 01_original_mri."
            ),
            "main_components": MAIN_COMPONENTS,
            "main_component_views": (
                "Axial overlays on T1n, T1c, T2f and T2w; "
                "plus T1c multiplanar views."
            ),
            "small_component": (
                "ED retained with T1c axial and multiplanar views."
            ),
            "excluded_components": {
                "CC": "Measured volume is zero.",
                "TC": "Composite of ET, NET and CC; WT is retained as the overall extent.",
            },
        },
        "selected_image_count": len(selected),
        "original_unsegmented_image_count": sum(
            x["category"] == "original_unsegmented_mri"
            for x in selected
        ),
        "segmentation_overlay_count": sum(
            "segmentation_overlay" in x["category"]
            for x in selected
        ),
        "missing_expected_files": missing,
        "images": selected,
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("GEMINI IMAGE SELECTION")
    print("=" * 65)
    print(f"Subject:                       {SUBJECT}")
    print(f"Total selected images:         {len(selected)}")
    print(
        "Original unsegmented images:  "
        f"{result['original_unsegmented_image_count']}"
    )
    print(
        "Segmentation overlay images:  "
        f"{result['segmentation_overlay_count']}"
    )
    print(f"Missing expected files:        {len(missing)}")

    print("\nSELECTED IMAGES")
    for item in selected:
        print(f"[{item['category']}] {item['path']}")

    if missing:
        print("\nMISSING EXPECTED FILES")
        for filename in missing:
            print("  -", filename)

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
