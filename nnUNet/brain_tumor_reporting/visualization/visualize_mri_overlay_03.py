

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from brain_tumor_reporting.config.dataset_config import (
    IMAGES_DIR,
    LABELS_DIR,
    OUTPUT_DIR,
    CHANNELS,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

# Colors correspond to labels 1, 2, 3, 4 respectively.
LABEL_COLORS = [
    "red",
    "orange",
    "lime",
    "cyan",
]

SEGMENTATION_CMAP = ListedColormap(LABEL_COLORS)


# ---------------------------------------------------------
# File utilities
# ---------------------------------------------------------

def remove_nii_extension(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def get_subject_ids():
    """Identify subjects using segmentation filenames."""

    mask_files = sorted(LABELS_DIR.glob("*.nii.gz"))

    return [
        remove_nii_extension(path.name)
        for path in mask_files
    ]


def load_nifti(path):
    """Load NIfTI and convert it to closest canonical orientation."""

    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    data = np.asarray(image.dataobj, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D image: {path.name}; "
            f"received shape {data.shape}"
        )

    return image, data


# ---------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------

def normalize_intensity(image):
    """Robust display normalization using nonzero voxels."""

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    nonzero = image[image != 0]

    if nonzero.size == 0:
        return image

    low, high = np.percentile(nonzero, [1, 99])

    if high <= low:
        return image

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)

    return image


# ---------------------------------------------------------
# Slice selection
# ---------------------------------------------------------

def get_slice_indices(segmentation, mode="tumor_center"):
    """
    Return representative axial, coronal and sagittal slices.

    Canonical orientation:
        Axis 0: Left-Right
        Axis 1: Posterior-Anterior
        Axis 2: Inferior-Superior
    """

    tumor_coordinates = np.argwhere(segmentation > 0)

    if mode == "tumor_center" and tumor_coordinates.size > 0:
        center = np.round(
            tumor_coordinates.mean(axis=0)
        ).astype(int)

    else:
        center = np.array(segmentation.shape) // 2

    center = np.minimum(
        center,
        np.array(segmentation.shape) - 1
    )

    return {
        "sagittal": int(center[0]),
        "coronal": int(center[1]),
        "axial": int(center[2]),
    }


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def plot_subject(subject_id, slice_mode="tumor_center", dpi=200):

    print(f"\nProcessing: {subject_id}")

    # Load segmentation
    segmentation_path = LABELS_DIR / f"{subject_id}.nii.gz"

    if not segmentation_path.exists():
        raise FileNotFoundError(
            f"Segmentation not found: {segmentation_path}"
        )

    seg_img, segmentation = load_nifti(segmentation_path)

    # Validate segmentation labels
    observed_labels = set(np.unique(segmentation).tolist())
    allowed_labels = {0, 1, 2, 3, 4}

    if not observed_labels.issubset(allowed_labels):
        raise ValueError(
            f"Unexpected segmentation labels: {observed_labels}"
        )

    segmentation = np.rint(segmentation).astype(np.uint8)

    # Load four MRI sequences
    modalities = {}

    for suffix, sequence_name in SEQUENCES.items():

        image_path = (
            IMAGES_DIR / f"{subject_id}_{suffix}.nii.gz"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {sequence_name}: {image_path}"
            )

        image, data = load_nifti(image_path)

        # Ensure that image and segmentation share geometry.
        if data.shape != segmentation.shape:
            raise ValueError(
                f"Shape mismatch for {sequence_name}: "
                f"{data.shape} versus {segmentation.shape}"
            )

        if not np.allclose(
            image.affine,
            seg_img.affine,
            atol=1e-4
        ):
            raise ValueError(
                f"Affine mismatch for {sequence_name}"
            )

        modalities[sequence_name] = normalize_intensity(data)

    # Choose representative slices
    indices = get_slice_indices(
        segmentation,
        mode=slice_mode
    )

    # Axes and corresponding array extraction
    views = {
        "Axial": (
            lambda x, i: np.rot90(x[:, :, i]),
            indices["axial"],
        ),
        "Coronal": (
            lambda x, i: np.rot90(x[:, i, :]),
            indices["coronal"],
        ),
        "Sagittal": (
            lambda x, i: np.rot90(x[i, :, :]),
            indices["sagittal"],
        ),
    }

    # Four columns = four MRI sequences.
    # Three rows = three anatomical orientations.
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(17, 12)
    )

    for row, (view_name, (extract_slice, slice_index)) in enumerate(
        views.items()
    ):

        for col, (sequence_name, image_data) in enumerate(
            modalities.items()
        ):

            ax = axes[row, col]

            mri_slice = extract_slice(
                image_data,
                slice_index
            )

            mask_slice = extract_slice(
                segmentation,
                slice_index
            )

            ax.imshow(
                mri_slice,
                cmap="gray",
                interpolation="nearest"
            )

            # Transparent background: display only tumor labels.
            overlay = np.ma.masked_where(
                mask_slice == 0,
                mask_slice
            )

            ax.imshow(
                overlay,
                cmap=SEGMENTATION_CMAP,
                vmin=1,
                vmax=4,
                alpha=0.45,
                interpolation="nearest"
            )

            if row == 0:
                ax.set_title(
                    sequence_name,
                    fontsize=13,
                    fontweight="bold"
                )

            if col == 0:
                ax.set_ylabel(
                    view_name,
                    fontsize=13,
                    fontweight="bold"
                )

            ax.set_xticks([])
            ax.set_yticks([])

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
        fontsize=11
    )

    fig.suptitle(
        f"{subject_id} - MRI and Segmentation Overlay",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    # Output directory
    subject_output = (
        OUTPUT_DIR
        / "visualizations"
        / subject_id
    )

    subject_output.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        subject_output
        / f"{subject_id}_mri_segmentation_overlay.png"
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

    return output_path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Visualize four MRI sequences with tumor segmentation."
    )

    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Subject ID, e.g. BraTSPED_00003"
    )

    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Process only the first N subjects."
    )

    parser.add_argument(
        "--slice_mode",
        choices=["tumor_center", "middle"],
        default="tumor_center"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200
    )

    args = parser.parse_args()

    if args.subject_id:

        subjects = [args.subject_id]

    else:

        subjects = get_subject_ids()

    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]

    print("=" * 55)
    print("MRI SEGMENTATION VISUALIZATION")
    print("=" * 55)

    print(f"Subjects selected: {len(subjects)}")

    successful = []
    failed = []

    for subject_id in subjects:

        try:

            output = plot_subject(
                subject_id,
                slice_mode=args.slice_mode,
                dpi=args.dpi
            )

            successful.append(str(output))

        except Exception as error:

            print(f"FAILED: {subject_id}")
            print(f"Reason: {error}")

            failed.append({
                "subject_id": subject_id,
                "error": str(error)
            })

    summary = {
        "selected_subjects": len(subjects),
        "successful_subjects": len(successful),
        "failed_subjects": failed,
        "outputs": successful,
    }

    summary_dir = OUTPUT_DIR / "visualizations"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / "visualization_summary.json"

    with open(summary_path, "w") as file:
        import json
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("VISUALIZATION SUMMARY")
    print("=" * 55)

    print(f"Selected: {len(subjects)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
