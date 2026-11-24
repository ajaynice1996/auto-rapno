

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from brain_tumor_reporting.config.dataset_config import (
    IMAGES_DIR,
    LABELS_DIR,
    OUTPUT_DIR,
)


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCES = {
    "0000": "T1n",
    "0001": "T1c",
    "0002": "T2-FLAIR",
    "0003": "T2w",
}

LABEL_NAMES = {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}

COMPONENTS = {
    "WT": [1, 2, 3, 4],
    "TC": [1, 2, 3],
    "ET": [1],
    "NET": [2],
    "CC": [3],
    "ED": [4],
}

# Colors: ET, NET, CC, ED
LABEL_COLORS = [
    "red",
    "orange",
    "lime",
    "cyan",
]

SEGMENTATION_CMAP = ListedColormap(LABEL_COLORS)

SEGMENTATION_NORM = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5],
    SEGMENTATION_CMAP.N
)

# Three anatomical orientations.
# Canonical NIfTI axes:
# 0 = left-right, 1 = posterior-anterior, 2 = inferior-superior.
ORIENTATIONS = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}


# =========================================================
# NIFTI UTILITIES
# =========================================================

def load_nifti(path):

    image = nib.load(str(path))

    # Convert to closest canonical orientation.
    image = nib.as_closest_canonical(image)

    data = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D NIfTI: {path.name}; "
            f"received {data.shape}"
        )

    return image, data


def normalize_intensity(data):

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    nonzero = data[data != 0]

    if nonzero.size == 0:
        return data

    low, high = np.percentile(
        nonzero,
        [1, 99]
    )

    if high <= low:
        return data

    data = np.clip(data, low, high)

    return (data - low) / (high - low)


def extract_slice(volume, axis, index):

    if axis == 0:
        result = volume[index, :, :]

    elif axis == 1:
        result = volume[:, index, :]

    else:
        result = volume[:, :, index]

    return np.rot90(result)


# =========================================================
# SUBJECT LOADING
# =========================================================

def load_subject(subject_id):

    segmentation_path = (
        LABELS_DIR / f"{subject_id}.nii.gz"
    )

    if not segmentation_path.exists():
        raise FileNotFoundError(
            f"Segmentation not found: {segmentation_path}"
        )

    seg_image, segmentation = load_nifti(
        segmentation_path
    )

    if not np.all(np.isfinite(segmentation)):
        raise ValueError("Segmentation contains non-finite values")

    rounded = np.rint(segmentation)

    if not np.allclose(
        segmentation,
        rounded,
        atol=1e-5
    ):
        raise ValueError(
            "Segmentation contains non-integer label values"
        )

    segmentation = rounded.astype(np.uint8)

    observed = set(np.unique(segmentation).tolist())

    if not observed.issubset({0, 1, 2, 3, 4}):
        raise ValueError(
            f"Unexpected segmentation labels: {observed}"
        )

    modalities = {}

    for suffix, sequence_name in SEQUENCES.items():

        image_path = (
            IMAGES_DIR
            / f"{subject_id}_{suffix}.nii.gz"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {sequence_name}: {image_path}"
            )

        image, data = load_nifti(image_path)

        if data.shape != segmentation.shape:
            raise ValueError(
                f"Shape mismatch for {sequence_name}: "
                f"{data.shape} versus {segmentation.shape}"
            )

        if not np.allclose(
            image.affine,
            seg_image.affine,
            atol=1e-4
        ):
            raise ValueError(
                f"Affine mismatch for {sequence_name}"
            )

        modalities[sequence_name] = normalize_intensity(
            data
        )

    return (
        seg_image,
        segmentation,
        modalities
    )


# =========================================================
# COMPONENT MASKS AND SLICE SELECTION
# =========================================================

def create_component_mask(segmentation, component_name):

    labels = COMPONENTS[component_name]

    return np.isin(
        segmentation,
        labels
    )


def select_maximum_area_slice(mask, axis):

    """
    Select the slice with the largest cross-sectional
    component area along the specified voxel axis.

    Returns:
        slice_index, maximum_area_voxels
    """

    if not np.any(mask):
        return (
            mask.shape[axis] // 2,
            0
        )

    reduce_axes = tuple(
        i for i in range(3)
        if i != axis
    )

    areas = np.sum(
        mask,
        axis=reduce_axes
    )

    index = int(np.argmax(areas))

    return (
        index,
        int(areas[index])
    )


def get_component_slices(mask):

    indices = {}
    areas = {}

    for orientation, axis in ORIENTATIONS.items():

        index, area = select_maximum_area_slice(
            mask,
            axis
        )

        indices[orientation] = index
        areas[orientation] = area

    return indices, areas


# =========================================================
# IMAGE COLLAGE GENERATION
# =========================================================

def create_collage(
    subject_id,
    component_name,
    segmentation,
    modalities,
    component_mask,
    slice_indices,
    output_path,
    show_overlay=False,
    dpi=220
):

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 13)
    )

    for row, (orientation, axis) in enumerate(
        ORIENTATIONS.items()
    ):

        slice_index = slice_indices[orientation]

        for col, sequence_name in enumerate(
            SEQUENCES.values()
        ):

            ax = axes[row, col]

            image_volume = modalities[sequence_name]

            image_slice = extract_slice(
                image_volume,
                axis,
                slice_index
            )

            ax.imshow(
                image_slice,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )

            if show_overlay:

                segmentation_slice = extract_slice(
                    segmentation,
                    axis,
                    slice_index
                )

                overlay = np.ma.masked_where(
                    segmentation_slice == 0,
                    segmentation_slice
                )

                ax.imshow(
                    overlay,
                    cmap=SEGMENTATION_CMAP,
                    norm=SEGMENTATION_NORM,
                    alpha=0.48,
                    interpolation="nearest"
                )

            # Outline the target component separately.
            target_slice = extract_slice(
                component_mask.astype(np.uint8),
                axis,
                slice_index
            )

            if np.any(target_slice):
                ax.contour(
                    target_slice,
                    levels=[0.5],
                    colors=["white"],
                    linewidths=0.7
                )

            if row == 0:
                ax.set_title(
                    sequence_name,
                    fontsize=13,
                    fontweight="bold"
                )

            if col == 0:
                ax.set_ylabel(
                    orientation,
                    fontsize=13,
                    fontweight="bold"
                )

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel(
                f"Slice index: {slice_index}",
                fontsize=8
            )

    if show_overlay:

        legend_handles = [
            Patch(
                facecolor=LABEL_COLORS[label - 1],
                label=LABEL_NAMES[label]
            )
            for label in LABEL_NAMES
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=10
        )

    else:

        fig.text(
            0.5,
            0.025,
            "Original MRI without segmentation-label overlay",
            ha="center",
            fontsize=10
        )

    overlay_text = (
        "WITH SEGMENTATION OVERLAY"
        if show_overlay
        else "WITHOUT SEGMENTATION OVERLAY"
    )

    fig.suptitle(
        f"{subject_id} | {component_name} | {overlay_text}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0.07, 1, 0.95]
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# SUBJECT PROCESSING
# =========================================================

def process_subject(subject_id, dpi=220):

    print(f"\nProcessing subject: {subject_id}")

    (
        seg_image,
        segmentation,
        modalities
    ) = load_subject(subject_id)

    subject_output = (
        OUTPUT_DIR
        / "visualizations"
        / subject_id
    )

    subject_output.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "subject_id": subject_id,
        "input_segmentation": str(
            LABELS_DIR / f"{subject_id}.nii.gz"
        ),
        "sequence_mapping": SEQUENCES,
        "label_mapping": LABEL_NAMES,
        "voxel_spacing_mm": (
            nib.affines.voxel_sizes(
                seg_image.affine
            ).tolist()
        ),
        "components": {}
    }

    for component_name in COMPONENTS:

        print(f"  Generating: {component_name}")

        component_mask = create_component_mask(
            segmentation,
            component_name
        )

        voxel_count = int(
            np.count_nonzero(component_mask)
        )

        indices, areas = get_component_slices(
            component_mask
        )

        prefix = (
            f"{subject_id}_{component_name}"
        )

        raw_path = (
            subject_output
            / f"{prefix}_raw.png"
        )

        overlay_path = (
            subject_output
            / f"{prefix}_overlay.png"
        )

        # First image: original MRI only.
        create_collage(
            subject_id=subject_id,
            component_name=component_name,
            segmentation=segmentation,
            modalities=modalities,
            component_mask=component_mask,
            slice_indices=indices,
            output_path=raw_path,
            show_overlay=False,
            dpi=dpi
        )

        # Second image: same MRI slices with all labels.
        create_collage(
            subject_id=subject_id,
            component_name=component_name,
            segmentation=segmentation,
            modalities=modalities,
            component_mask=component_mask,
            slice_indices=indices,
            output_path=overlay_path,
            show_overlay=True,
            dpi=dpi
        )

        manifest["components"][component_name] = {
            "present": voxel_count > 0,
            "voxel_count": voxel_count,
            "maximum_area_slice_indices": indices,
            "maximum_area_voxels": areas,
            "raw_image": str(raw_path),
            "overlay_image": str(overlay_path)
        }

    manifest_path = (
        subject_output
        / "llm_review_manifest.json"
    )

    with open(manifest_path, "w") as file:
        json.dump(
            manifest,
            file,
            indent=4
        )

    print(f"\nSaved visualizations in: {subject_output}")
    print(f"Saved manifest: {manifest_path}")

    return manifest


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate MRI PNG collages for multimodal "
            "LLM review, with and without segmentation overlays."
        )
    )

    parser.add_argument(
        "--subject_id",
        type=str,
        required=True,
        help="Subject ID, e.g. BraTSPED_00003"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MRI VISUALIZATION FOR LLM REVIEW")
    print("=" * 60)

    try:

        process_subject(
            subject_id=args.subject_id,
            dpi=args.dpi
        )

        print("\nVisualization completed successfully.")

    except Exception as error:

        print("\nVisualization failed.")
        print(f"Reason: {error}")

        raise


if __name__ == "__main__":
    main()
