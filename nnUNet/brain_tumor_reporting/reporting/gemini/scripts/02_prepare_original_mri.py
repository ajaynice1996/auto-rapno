#!/usr/bin/env python3

from pathlib import Path
import json
import sys

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
NNUNET_DIR = PROJECT_ROOT / "nnUNet"

SUBJECT = "BraTSPED_00002"

IMAGE_DIR = (
    NNUNET_DIR
    / "nnUNet_raw"
    / "Dataset202_BraTS"
    / "imagesTr_all"
)

LABEL_DIR = (
    NNUNET_DIR
    / "nnUNet_raw"
    / "Dataset202_BraTS"
    / "labelsTr_all"
)

OUTPUT_DIR = (
    NNUNET_DIR
    / "brain_tumor_reporting"
    / "reporting"
    / "gemini"
    / SUBJECT
    / "llm_inputs"
    / "01_original_mri"
)

SEQUENCES = {
    "T1n": f"{SUBJECT}_0000.nii.gz",
    "T1c": f"{SUBJECT}_0001.nii.gz",
    "T2f": f"{SUBJECT}_0002.nii.gz",
    "T2w": f"{SUBJECT}_0003.nii.gz",
}

MASK_FILE = LABEL_DIR / f"{SUBJECT}.nii.gz"

N_AXIAL_SLICES = 15
MARGIN_FRACTION = 0.10


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_canonical(path):
    """Load a NIfTI image in canonical RAS orientation."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    data = np.asarray(image.dataobj)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D image: {path.name}; got shape {data.shape}"
        )

    return image, data


def get_display_limits(data):
    """Calculate robust intensity limits, excluding background zeros."""
    values = data[np.isfinite(data) & (data != 0)]

    if values.size == 0:
        return 0, 1

    low, high = np.percentile(values, [1, 99])

    if low >= high:
        low = float(np.min(values))
        high = float(np.max(values)) + 1e-6

    return float(low), float(high)


def axial_view(data, z):
    """Return an axial image in a consistent radiological display."""
    return np.fliplr(data[:, :, z].T)


def coronal_view(data, y):
    """Return a coronal image."""
    return np.fliplr(data[:, y, :].T)


def sagittal_view(data, x):
    """Return a sagittal image."""
    return np.fliplr(data[x, :, :].T)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------
# Load MRI sequences and segmentation
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading MRI sequences...")

    images = {}
    arrays = {}
    display_limits = {}

    reference_shape = None
    reference_affine = None

    for sequence, filename in SEQUENCES.items():

        path = IMAGE_DIR / filename
        image, data = load_canonical(path)

        if reference_shape is None:
            reference_shape = data.shape
            reference_affine = image.affine
        else:
            if data.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {sequence}: "
                    f"{data.shape} vs {reference_shape}"
                )

            if not np.allclose(
                image.affine,
                reference_affine,
                atol=1e-4
            ):
                raise ValueError(
                    f"Affine mismatch for sequence {sequence}"
                )

        images[sequence] = image
        arrays[sequence] = data
        display_limits[sequence] = get_display_limits(data)

        print(
            f"{sequence}: shape={data.shape}, "
            f"display range={display_limits[sequence]}"
        )

    print("\nLoading tumor segmentation...")

    mask_image, mask = load_canonical(MASK_FILE)

    if mask.shape != reference_shape:
        raise ValueError(
            f"Segmentation shape mismatch: {mask.shape} "
            f"vs MRI {reference_shape}"
        )

    if not np.allclose(
        mask_image.affine,
        reference_affine,
        atol=1e-4
    ):
        raise ValueError(
            "Segmentation affine does not match MRI geometry."
        )

    tumor = mask > 0

    if not np.any(tumor):
        raise ValueError("No tumor voxels found in segmentation.")

    # -----------------------------------------------------
    # Determine tumor bounding box and slice positions
    # -----------------------------------------------------

    coordinates = np.where(tumor)

    x_min, x_max = int(coordinates[0].min()), int(coordinates[0].max())
    y_min, y_max = int(coordinates[1].min()), int(coordinates[1].max())
    z_min, z_max = int(coordinates[2].min()), int(coordinates[2].max())

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    z_center = (z_min + z_max) // 2

    tumor_z_extent = z_max - z_min + 1
    margin = max(2, int(round(tumor_z_extent * MARGIN_FRACTION)))

    z_start = max(0, z_min - margin)
    z_end = min(reference_shape[2] - 1, z_max + margin)

    # Select unique, approximately equally spaced axial slices.
    z_indices = np.linspace(
        z_start,
        z_end,
        min(N_AXIAL_SLICES, z_end - z_start + 1)
    ).round().astype(int)

    z_indices = np.unique(z_indices)

    print("\nTumor bounding box in canonical RAS voxel coordinates:")
    print(f"X: {x_min} to {x_max}")
    print(f"Y: {y_min} to {y_max}")
    print(f"Z: {z_min} to {z_max}")
    print(f"\nSelected axial slices: {len(z_indices)}")
    print(z_indices.tolist())

    # -----------------------------------------------------
    # Convert selected voxel positions to RAS coordinates
    # -----------------------------------------------------

    slice_metadata = []

    for z in z_indices:
        ras = nib.affines.apply_affine(
            reference_affine,
            [x_center, y_center, int(z)]
        )

        slice_metadata.append({
            "z_voxel_index": int(z),
            "ras_coordinates_mm": {
                "R": round(float(ras[0]), 2),
                "A": round(float(ras[1]), 2),
                "S": round(float(ras[2]), 2)
            }
        })

    center_ras = nib.affines.apply_affine(
        reference_affine,
        [x_center, y_center, z_center]
    )

    # -----------------------------------------------------
    # 1–4. Individual 15-slice collages
    # -----------------------------------------------------

    print("\nGenerating individual sequence collages...")

    for sequence, data in arrays.items():

        fig, axes = plt.subplots(
            3, 5,
            figsize=(17, 11)
        )

        axes = axes.flatten()
        vmin, vmax = display_limits[sequence]

        for i, ax in enumerate(axes):

            if i < len(z_indices):
                z = int(z_indices[i])
                view = axial_view(data, z)

                ax.imshow(
                    view,
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    origin="upper"
                )

                ax.set_title(f"Axial Z = {z}", fontsize=10)

            ax.axis("off")

        fig.suptitle(
            f"{SUBJECT} | {sequence} | Original MRI | "
            f"{len(z_indices)} Matched Axial Slices",
            fontsize=15
        )

        output_path = (
            OUTPUT_DIR
            / f"{SUBJECT}_{sequence}_axial_15slice_collage.png"
        )

        save_figure(fig, output_path)

    # -----------------------------------------------------
    # 5. Matched four-sequence axial comparison
    # -----------------------------------------------------

    print("\nGenerating matched four-sequence comparison...")

    sequence_names = list(SEQUENCES.keys())

    fig, axes = plt.subplots(
        len(sequence_names),
        len(z_indices),
        figsize=(max(24, len(z_indices) * 2.0), 9)
    )

    if len(z_indices) == 1:
        axes = axes.reshape(len(sequence_names), 1)

    for row, sequence in enumerate(sequence_names):

        data = arrays[sequence]
        vmin, vmax = display_limits[sequence]

        for col, z in enumerate(z_indices):

            ax = axes[row, col]

            ax.imshow(
                axial_view(data, int(z)),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="upper"
            )

            if row == 0:
                ax.set_title(f"Z = {z}", fontsize=9)

            if col == 0:
                ax.set_ylabel(
                    sequence,
                    fontsize=11,
                    fontweight="bold"
                )

            ax.axis("off")

    fig.suptitle(
        f"{SUBJECT} | Matched Axial MRI Comparison "
        f"(T1n, T1c, T2f, T2w)",
        fontsize=15
    )

    output_path = (
        OUTPUT_DIR
        / f"{SUBJECT}_all_sequences_matched_axial_collage.png"
    )

    save_figure(fig, output_path)

    # -----------------------------------------------------
    # 6. Multiplanar overview: 3 views x 4 sequences
    # -----------------------------------------------------

    print("\nGenerating multiplanar overview...")

    fig, axes = plt.subplots(
        3, 4,
        figsize=(17, 12)
    )

    view_functions = [
        ("Axial", axial_view, z_center),
        ("Coronal", coronal_view, y_center),
        ("Sagittal", sagittal_view, x_center)
    ]

    for col, sequence in enumerate(sequence_names):

        data = arrays[sequence]
        vmin, vmax = display_limits[sequence]

        for row, (view_name, view_function, index) in enumerate(
            view_functions
        ):

            ax = axes[row, col]

            view = view_function(data, index)

            ax.imshow(
                view,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="upper"
            )

            if row == 0:
                ax.set_title(sequence, fontsize=13, fontweight="bold")

            if col == 0:
                ax.set_ylabel(
                    view_name,
                    fontsize=12,
                    fontweight="bold"
                )

            ax.axis("off")

    fig.suptitle(
        f"{SUBJECT} | Original MRI Multiplanar Overview | "
        f"Tumor Bounding-box Center",
        fontsize=15
    )

    output_path = (
        OUTPUT_DIR
        / f"{SUBJECT}_multiplanar_original_overview.png"
    )

    save_figure(fig, output_path)

    # -----------------------------------------------------
    # Save slice selection metadata
    # -----------------------------------------------------

    metadata = {
        "subject": SUBJECT,
        "description": (
            "Original MRI visualization using matched axial slices "
            "selected across the tumor superior-inferior extent."
        ),
        "source_directory": str(IMAGE_DIR),
        "segmentation_file": str(MASK_FILE),
        "canonical_orientation": "RAS",
        "sequence_order": sequence_names,
        "mri_shape_voxels": list(reference_shape),
        "tumor_voxel_bounding_box": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max
        },
        "tumor_center_voxel": {
            "x": x_center,
            "y": y_center,
            "z": z_center
        },
        "tumor_center_ras_mm": {
            "R": round(float(center_ras[0]), 2),
            "A": round(float(center_ras[1]), 2),
            "S": round(float(center_ras[2]), 2)
        },
        "axial_slice_count": len(z_indices),
        "axial_slice_positions": slice_metadata,
        "display_note": (
            "Images are original MRI intensities without tumor overlays. "
            "Intensity windows are independently calculated for each "
            "sequence using the 1st and 99th percentiles of nonzero "
            "finite voxels. Slice positions are matched across sequences."
        )
    }

    metadata_path = OUTPUT_DIR / f"{SUBJECT}_slice_selection.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved metadata: {metadata_path}")

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("ORIGINAL MRI COLLAGE GENERATION COMPLETED")
    print("=" * 65)

    print(f"Subject: {SUBJECT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Axial slices per sequence: {len(z_indices)}")
    print(f"Total axial sequence panels: {len(z_indices) * 4}")
    print("Expected PNG collages: 6")
    print("Metadata JSON: 1")
    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.glob(f"{SUBJECT}_*")):
        print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
