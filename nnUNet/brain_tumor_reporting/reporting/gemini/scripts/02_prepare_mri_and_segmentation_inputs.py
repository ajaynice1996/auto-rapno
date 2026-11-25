#!/usr/bin/env python3

from pathlib import Path
import json
import warnings

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from nibabel.processing import resample_from_to


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
NNUNET_ROOT = PROJECT_ROOT / "nnUNet"

SUBJECT = "BraTSPED_00002"

DATASET_DIR = NNUNET_ROOT / "nnUNet_raw" / "Dataset202_BraTS"
IMAGE_DIR = DATASET_DIR / "imagesTr_all"
MASK_PATH = DATASET_DIR / "labelsTr_all" / f"{SUBJECT}.nii.gz"

BASE_DIR = (
    NNUNET_ROOT
    / "brain_tumor_reporting"
    / "reporting"
    / "gemini"
    / SUBJECT
    / "llm_inputs"
)

ORIGINAL_DIR = BASE_DIR / "01_original_mri"
SEGMENTATION_DIR = BASE_DIR / "02_tumor_segmentation"

SEQUENCES = {
    "T1n": "0000",
    "T1c": "0001",
    "T2f": "0002",
    "T2w": "0003",
}

COMPONENTS = {
    "ET": [1],
    "NET": [2],
    "CC": [3],
    "ED": [4],
    "TC": [1, 2, 3],
    "WT": [1, 2, 3, 4],
}

COLORS = {
    "ET": "red",
    "NET": "lime",
    "CC": "dodgerblue",
    "ED": "yellow",
    "TC": "magenta",
    "WT": "cyan",
}

# Axial: 24 slices (4 x 6)
# Coronal: 20 slices (4 x 5)
# Sagittal: 20 slices (4 x 5)
SLICE_COUNTS = {
    "axial": 24,
    "coronal": 20,
    "sagittal": 20,
}

# Multiplanar: 3 slices per plane.
MULTIPLANAR_COUNT = 3

# Minimal panel gaps.
PANEL_WSPACE = 0.015
PANEL_HSPACE = 0.035

# Overlay appearance.
OVERLAY_ALPHA = 0.30
CONTOUR_WIDTH = 0.9

# Preferred display sequences (all four will still be generated).
PREFERRED_SEQUENCE = {
    "ET": "T1c",
    "NET": "T2f",
    "CC": "T1c",
    "ED": "T2f",
    "TC": "T1c",
    "WT": "T2f",
}


# ============================================================
# 2. NIFTI LOADING AND PHYSICAL ALIGNMENT
# ============================================================

def load_nifti(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return nib.load(str(path))


def canonical_image(image):
    return nib.as_closest_canonical(image)


def same_geometry(image_a, image_b):
    return (
        image_a.shape == image_b.shape
        and np.allclose(
            image_a.affine,
            image_b.affine,
            atol=1e-4
        )
    )


def resample_to_reference(image, reference, order):
    """
    Resample an image to the exact reference grid.

    order=0: nearest-neighbour for discrete segmentation labels.
    order=1: linear interpolation for MRI intensities.
    """
    if same_geometry(image, reference):
        return image

    print(
        f"  Resampling {image.shape} to reference "
        f"{reference.shape}; interpolation order={order}"
    )

    return resample_from_to(
        image,
        (reference.shape, reference.affine),
        order=order
    )


def load_aligned_data():
    """
    T1n is the common reference grid.

    All MRI sequences are linearly resampled to T1n when necessary.
    Tumour labels are nearest-neighbour resampled to T1n.
    """

    print("\nLoading T1n as reference geometry...")

    t1n_path = IMAGE_DIR / f"{SUBJECT}_{SEQUENCES['T1n']}.nii.gz"
    reference = load_nifti(t1n_path)

    print("Reference shape:", reference.shape)
    print("Reference orientation:", nib.aff2axcodes(reference.affine))

    volumes = {}

    for sequence, channel in SEQUENCES.items():
        path = IMAGE_DIR / f"{SUBJECT}_{channel}.nii.gz"

        image = load_nifti(path)
        image = resample_to_reference(image, reference, order=1)

        volumes[sequence] = np.asarray(
            image.dataobj,
            dtype=np.float32
        )

        print(
            f"{sequence}: shape={image.shape}; "
            f"orientation={nib.aff2axcodes(image.affine)}"
        )

    print("\nLoading tumour segmentation...")

    mask_image = load_nifti(MASK_PATH)

    # Crucial: discrete labels must use nearest-neighbour interpolation.
    mask_image = resample_to_reference(
        mask_image,
        reference,
        order=0
    )

    label_map = np.rint(
        np.asarray(mask_image.dataobj)
    ).astype(np.int16)

    # Final strict geometry validation.
    for sequence in volumes:
        if volumes[sequence].shape != label_map.shape:
            raise RuntimeError(
                f"Shape mismatch after alignment: {sequence}"
            )

    print("\nAll sequences and segmentation are now on the T1n reference grid.")
    print("Unique segmentation labels:", np.unique(label_map))

    return reference, volumes, label_map


# ============================================================
# 3. DISPLAY ORIENTATION
# ============================================================

def display_slice(volume, plane, index):
    """
    Input volume is canonical RAS:
      X = Right
      Y = Anterior
      Z = Superior

    Display conventions:
      Axial: anterior at top, radiological left-right.
      Coronal: superior at top, radiological left-right.
      Sagittal: superior at top, anterior on displayed left.

    The identical transformation is applied to MRI and masks.
    """

    if plane == "axial":
        image = volume[:, :, index].T

    elif plane == "coronal":
        image = volume[:, index, :].T

    elif plane == "sagittal":
        image = volume[index, :, :].T

    else:
        raise ValueError(f"Unknown plane: {plane}")

    return np.flipud(np.fliplr(image))


def robust_limits(volume):
    valid = volume[np.isfinite(volume) & (volume != 0)]

    if valid.size == 0:
        return 0.0, 1.0

    low, high = np.percentile(valid, [1, 99])

    if high <= low:
        high = low + 1

    return float(low), float(high)


# ============================================================
# 4. TUMOUR BOUNDING BOX AND SLICE SELECTION
# ============================================================

def get_tumour_bbox(label_map):
    coordinates = np.argwhere(label_map > 0)

    if coordinates.size == 0:
        raise ValueError("Tumour segmentation is empty.")

    return coordinates.min(axis=0), coordinates.max(axis=0)


def select_slices(low, high, requested, dimension):
    """
    Select evenly distributed, unique slices across the tumour extent,
    with a small margin outside the bounding box.
    """

    margin = max(
        1,
        int(round((high - low + 1) * 0.05))
    )

    start = max(0, low - margin)
    stop = min(dimension - 1, high + margin)

    available = stop - start + 1
    count = min(requested, available)

    selected = np.linspace(start, stop, count)
    return np.unique(np.rint(selected).astype(int)).tolist()


def select_multiplanar_slices(low, high, dimension):
    center = int(round((low + high) / 2))
    offset = max(1, int(round(max(2, high - low) / 8)))

    return sorted(set([
        max(0, min(dimension - 1, center - offset)),
        center,
        max(0, min(dimension - 1, center + offset)),
    ]))


def get_ras_coordinate(affine, xyz):
    point = nib.affines.apply_affine(
        affine,
        np.asarray(xyz, dtype=float)
    )

    return [round(float(v), 2) for v in point]


def create_slice_selection(label_map):
    bbox_min, bbox_max = get_tumour_bbox(label_map)

    xdim, ydim, zdim = label_map.shape

    selected = {
        "axial": select_slices(
            int(bbox_min[2]), int(bbox_max[2]),
            SLICE_COUNTS["axial"], zdim
        ),
        "coronal": select_slices(
            int(bbox_min[1]), int(bbox_max[1]),
            SLICE_COUNTS["coronal"], ydim
        ),
        "sagittal": select_slices(
            int(bbox_min[0]), int(bbox_max[0]),
            SLICE_COUNTS["sagittal"], xdim
        ),
    }

    multiplanar = {
        "axial": select_multiplanar_slices(
            int(bbox_min[2]), int(bbox_max[2]), zdim
        ),
        "coronal": select_multiplanar_slices(
            int(bbox_min[1]), int(bbox_max[1]), ydim
        ),
        "sagittal": select_multiplanar_slices(
            int(bbox_min[0]), int(bbox_max[0]), xdim
        ),
    }

    center = np.rint(
        (bbox_min + bbox_max) / 2
    ).astype(int).tolist()

    return bbox_min, bbox_max, selected, multiplanar, center


# ============================================================
# 5. FIGURE LAYOUT
# ============================================================

def create_axes(rows, columns, cell_size=2.5):
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * cell_size, rows * cell_size),
        squeeze=False
    )

    for ax in axes.flat:
        ax.set_axis_off()

    return fig, axes


def finish_figure(fig, output_path, title, title_color="black"):
    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        color=title_color,
        y=0.99
    )

    # Reduce unnecessary margins and spacing.
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.94,
        wspace=PANEL_WSPACE,
        hspace=PANEL_HSPACE
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close(fig)

    print("Saved:", output_path)


def add_slice_title(ax, plane, index):
    ax.set_title(
        f"{plane.capitalize()} {index}",
        fontsize=7,
        pad=2
    )


# ============================================================
# 6. ORIGINAL MRI COLLAGES
# ============================================================

def save_original_plane(
    volume, sequence, plane, indices, output_path
):
    columns = 6 if plane == "axial" else 5
    rows = 4

    fig, axes = create_axes(rows, columns)
    vmin, vmax = robust_limits(volume)

    for n, index in enumerate(indices):
        ax = axes.flat[n]

        image = display_slice(volume, plane, index)

        ax.imshow(
            image,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper"
        )

        add_slice_title(ax, plane, index)

    finish_figure(
        fig,
        output_path,
        f"{SUBJECT} | {sequence} | Original MRI | {plane.capitalize()}"
    )


def save_original_multiplanar(
    volume, sequence, multiplanar, output_path
):
    fig, axes = create_axes(3, 3, cell_size=3)
    vmin, vmax = robust_limits(volume)

    for row, plane in enumerate(
        ["axial", "coronal", "sagittal"]
    ):
        for col, index in enumerate(multiplanar[plane]):
            image = display_slice(volume, plane, index)

            axes[row, col].imshow(
                image,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="upper"
            )

            add_slice_title(axes[row, col], plane, index)

    finish_figure(
        fig,
        output_path,
        f"{SUBJECT} | {sequence} | Multiplanar Original MRI"
    )


def save_matched_axial(volumes, axial_indices, output_path):
    rows = len(volumes)
    columns = len(axial_indices)

    fig, axes = create_axes(rows, columns, cell_size=1.4)

    for row, (sequence, volume) in enumerate(volumes.items()):
        vmin, vmax = robust_limits(volume)

        for col, index in enumerate(axial_indices):
            image = display_slice(volume, "axial", index)

            axes[row, col].imshow(
                image,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="upper"
            )

            if row == 0:
                axes[row, col].set_title(
                    f"Z={index}", fontsize=7
                )

            if col == 0:
                axes[row, col].set_ylabel(
                    sequence,
                    fontsize=10
                )

    finish_figure(
        fig,
        output_path,
        f"{SUBJECT} | Matched Axial MRI Comparison"
    )


# ============================================================
# 7. SEGMENTATION OVERLAYS
# ============================================================

def overlay_slice(
    ax,
    volume,
    component_mask,
    plane,
    index,
    component,
    vmin,
    vmax
):
    """
    Both MRI and mask are already aligned to the T1n reference grid.
    The exact same slice extraction and orientation transform is used.
    """

    image = display_slice(volume, plane, index)

    mask_2d = display_slice(
        component_mask.astype(np.uint8),
        plane,
        index
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        origin="upper"
    )

    if np.any(mask_2d):
        color = COLORS[component]

        rgba = np.zeros(
            (mask_2d.shape[0], mask_2d.shape[1], 4),
            dtype=float
        )

        rgba[mask_2d > 0] = to_rgba(
            color,
            alpha=OVERLAY_ALPHA
        )

        ax.imshow(
            rgba,
            origin="upper"
        )

        # Contour is calculated from the same displayed mask.
        # ax.contour(
        #     mask_2d.astype(float),
        #     levels=[0.5],
        #     colors=[color],
        #     linewidths=CONTOUR_WIDTH,
        #     origin="upper"
        # )

    ax.set_xticks([])
    ax.set_yticks([])


def save_component_plane(
    volume,
    component_mask,
    sequence,
    component,
    plane,
    indices,
    output_path
):
    columns = 6 if plane == "axial" else 5
    rows = 4

    fig, axes = create_axes(rows, columns)

    vmin, vmax = robust_limits(volume)
    voxel_count = int(component_mask.sum())

    present = voxel_count > 0

    for n, index in enumerate(indices):
        ax = axes.flat[n]

        overlay_slice(
            ax,
            volume,
            component_mask,
            plane,
            index,
            component,
            vmin,
            vmax
        )

        add_slice_title(ax, plane, index)

    # Absent-component status is in the title, never over the MRI.
    if present:
        title = (
            f"{SUBJECT} | {component} on {sequence} | "
            f"{plane.capitalize()} | {voxel_count:,} voxels"
        )
        title_color = "black"
    else:
        title = (
            f"{SUBJECT} | {component} NOT PRESENT IN SEGMENTATION | "
            f"{sequence} | {plane.capitalize()}"
        )
        title_color = "darkred"

    finish_figure(
        fig,
        output_path,
        title,
        title_color
    )


def save_component_multiplanar(
    volume,
    component_mask,
    sequence,
    component,
    multiplanar,
    output_path
):
    fig, axes = create_axes(3, 3, cell_size=3)

    vmin, vmax = robust_limits(volume)
    voxel_count = int(component_mask.sum())

    for row, plane in enumerate(
        ["axial", "coronal", "sagittal"]
    ):
        for col, index in enumerate(multiplanar[plane]):
            ax = axes[row, col]

            overlay_slice(
                ax,
                volume,
                component_mask,
                plane,
                index,
                component,
                vmin,
                vmax
            )

            add_slice_title(ax, plane, index)

    if voxel_count > 0:
        title = (
            f"{SUBJECT} | {component} on {sequence} | "
            f"Multiplanar | {voxel_count:,} voxels"
        )
        title_color = "black"
    else:
        title = (
            f"{SUBJECT} | {component} NOT PRESENT IN SEGMENTATION | "
            f"{sequence} | Multiplanar"
        )
        title_color = "darkred"

    finish_figure(
        fig,
        output_path,
        title,
        title_color
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MRI + TUMOUR SEGMENTATION INPUT PREPARATION")
    print("Subject:", SUBJECT)
    print("=" * 75)

    # --------------------------------------------------------
    # Load and physically align all data.
    # --------------------------------------------------------

    reference, volumes, label_map = load_aligned_data()

    (
        bbox_min,
        bbox_max,
        selected,
        multiplanar,
        center
    ) = create_slice_selection(label_map)

    print("\nTumour bounding box:")
    print("Minimum:", bbox_min.tolist())
    print("Maximum:", bbox_max.tolist())
    print("Centre:", center)

    print("\nSelected slice counts:")
    for plane, indices in selected.items():
        print(f"  {plane}: {len(indices)}")

    print("\nMultiplanar positions:")
    print(multiplanar)

    # --------------------------------------------------------
    # Original MRI
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("A. GENERATING ORIGINAL MRI COLLAGES")
    print("=" * 75)

    for sequence, volume in volumes.items():

        sequence_dir = ORIGINAL_DIR / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)

        for plane, indices in selected.items():

            output = (
                sequence_dir
                / f"{SUBJECT}_{sequence}_{plane}_collage.png"
            )

            save_original_plane(
                volume,
                sequence,
                plane,
                indices,
                output
            )

        output = (
            sequence_dir
            / f"{SUBJECT}_{sequence}_multiplanar_3x3.png"
        )

        save_original_multiplanar(
            volume,
            sequence,
            multiplanar,
            output
        )

    # Preserve existing matched axial image.
    # New output has a different filename.
    matched_output = (
        ORIGINAL_DIR
        / f"{SUBJECT}_matched_axial_comparison_v3.png"
    )

    save_matched_axial(
        volumes,
        selected["axial"],
        matched_output
    )

    # --------------------------------------------------------
    # Segmentation component overlays
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("B. GENERATING SEGMENTATION OVERLAYS")
    print("=" * 75)

    component_metadata = {}

    for component, labels in COMPONENTS.items():

        component_mask = np.isin(
            label_map,
            labels
        ).astype(np.uint8)

        voxel_count = int(component_mask.sum())
        present = voxel_count > 0

        component_metadata[component] = {
            "source_labels": labels,
            "voxel_count": voxel_count,
            "present": present,
            "preferred_sequence": PREFERRED_SEQUENCE[component],
            "colour": COLORS[component],
        }

        print(
            f"\n{component}: {voxel_count:,} voxels | "
            f"{'PRESENT' if present else 'NOT PRESENT'}"
        )

        for sequence, volume in volumes.items():

            output_dir = (
                SEGMENTATION_DIR
                / component
                / sequence
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Axial, coronal and sagittal.
            for plane, indices in selected.items():

                output = (
                    output_dir
                    / f"{SUBJECT}_{component}_{sequence}_{plane}_overlay.png"
                )

                save_component_plane(
                    volume,
                    component_mask,
                    sequence,
                    component,
                    plane,
                    indices,
                    output
                )

            # Three slices per plane.
            output = (
                output_dir
                / f"{SUBJECT}_{component}_{sequence}_multiplanar_3x3_overlay.png"
            )

            save_component_multiplanar(
                volume,
                component_mask,
                sequence,
                component,
                multiplanar,
                output
            )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "subject": SUBJECT,
        "reference_sequence": "T1n",
        "reference_orientation": list(
            nib.aff2axcodes(reference.affine)
        ),
        "reference_shape": list(reference.shape),
        "reference_affine": reference.affine.tolist(),
        "mask_interpolation": "nearest-neighbour",
        "mri_interpolation_if_resampled": "linear",
        "display_orientation": {
            "axial": "Anterior at top; radiological left-right",
            "coronal": "Superior at top; radiological left-right",
            "sagittal": "Superior at top; anterior on displayed left",
        },
        "tumour_bbox_min_voxel": bbox_min.tolist(),
        "tumour_bbox_max_voxel": bbox_max.tolist(),
        "tumour_center_voxel_xyz": center,
        "tumour_center_ras_mm": get_ras_coordinate(
            reference.affine,
            center
        ),
        "selected_slice_indices": selected,
        "multiplanar_slice_indices": multiplanar,
        "sequence_channels": SEQUENCES,
        "component_definitions": component_metadata,
        "overlay_colours": COLORS,
        "overlay_alpha": OVERLAY_ALPHA,
        "preferred_display_sequence": PREFERRED_SEQUENCE,
        "notes": [
            "All MRI sequences and tumour labels are aligned to the T1n reference grid.",
            "MRI intensity images use linear interpolation if resampling is needed.",
            "Segmentation labels use nearest-neighbour interpolation.",
            "The same slice indices and display transforms are used for MRI and masks.",
            "TC is the union of labels 1, 2 and 3.",
            "WT is the union of labels 1, 2, 3 and 4.",
            "TC and WT are derived masks, not independent tumour classes.",
            "Spatial overlap does not establish histological invasion.",
            "All overlays are research visualizations requiring expert review."
        ],
    }

    for output_dir in [ORIGINAL_DIR, SEGMENTATION_DIR]:
        metadata_path = (
            output_dir / f"{SUBJECT}_alignment_and_slice_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("ALL OUTPUTS GENERATED")
    print("=" * 75)

    print("\nOriginal MRI:")
    print(ORIGINAL_DIR)
    print("  4 sequences x 4 collages = 16 PNGs")
    print("  1 matched axial comparison PNG")

    print("\nTumour segmentation:")
    print(SEGMENTATION_DIR)
    print("  6 components x 4 sequences x 4 views = 96 PNGs")

    print("\nComponent status:")
    for component, details in component_metadata.items():
        print(
            f"  {component}: "
            f"{'PRESENT' if details['present'] else 'NOT PRESENT'} "
            f"({details['voxel_count']:,} voxels)"
        )

    print("\nMetadata saved in both output folders.")
    print("Please visually inspect the overlays before Gemini ingestion.")


if __name__ == "__main__":
    main()
