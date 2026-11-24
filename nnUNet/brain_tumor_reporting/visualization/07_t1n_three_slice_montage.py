from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from matplotlib.colors import ListedColormap, BoundaryNorm


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "nnUNet_raw" / "Dataset202_BraTS" / "imagesTr_all"
LABEL_DIR = ROOT / "nnUNet_raw" / "Dataset202_BraTS" / "labelsTr_all"
OUTPUT_DIR = ROOT / "brain_tumor_reporting" / "outputs" / "visualization"

COLORS = {
    1: "red",
    2: "orange",
    3: "cyan",
    4: "lime",
}

LABEL_NAMES = {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}

cmap = ListedColormap([
    "black", "red", "orange", "cyan", "lime"
])

norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    cmap.N
)


def get_tumor_center(labels):
    """Find the center of mass of all non-background tumor labels."""
    coords = np.argwhere(np.isin(labels, [1, 2, 3, 4]))

    if len(coords) == 0:
        raise ValueError(
            "No tumor labels (1, 2, 3, 4) found in the segmentation."
        )

    center = np.round(coords.mean(axis=0)).astype(int)

    print("Tumor center (voxel coordinates):", center.tolist())
    return center


def get_three_slices(volume, axis, center):
    """Select three consecutive slices around the tumor center."""
    axis_center = int(center[axis])

    indices = [
        max(0, min(volume.shape[axis] - 1, axis_center + offset))
        for offset in [-1, 0, 1]
    ]

    slices = []

    for index in indices:
        if axis == 2:       # Axial
            image = volume[:, :, index].T
        elif axis == 1:     # Coronal
            image = volume[:, index, :].T
        else:               # Sagittal
            image = volume[index, :, :].T

        slices.append((index, image))

    return slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="BraTSPED_00002")
    args = parser.parse_args()

    subject = args.subject

    image_path = IMAGE_DIR / f"{subject}_0000.nii.gz"
    label_path = LABEL_DIR / f"{subject}.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load T1n MRI in canonical orientation
    image_nii = nib.as_closest_canonical(nib.load(str(image_path)))
    image = image_nii.get_fdata(dtype=np.float32)

    # Load tumor segmentation
    label_nii = nib.as_closest_canonical(nib.load(str(label_path)))

    # Align segmentation to the T1n reference grid
    if (
        label_nii.shape != image_nii.shape
        or not np.allclose(label_nii.affine, image_nii.affine, atol=1e-4)
    ):
        print("Resampling tumor segmentation to T1n grid...")
        label_nii = resample_from_to(
            label_nii,
            image_nii,
            order=0
        )

    labels = np.rint(
        label_nii.get_fdata(dtype=np.float32)
    ).astype(np.uint8)

    if image.shape != labels.shape:
        raise ValueError("MRI and segmentation dimensions do not match.")

    print("MRI shape:", image.shape)
    print("Labels present:", np.unique(labels))

    # Find tumor center in the aligned segmentation
    tumor_center = get_tumor_center(labels)

    # Robust MRI intensity window
    valid = image[np.isfinite(image)]
    low, high = np.percentile(valid, [1, 99])

    orientations = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    fig, axes = plt.subplots(
        6, 3,
        figsize=(13, 21)
    )

    for orientation_index, (orientation, axis) in enumerate(orientations):

        image_slices = get_three_slices(image, axis, tumor_center)
        label_slices = get_three_slices(labels, axis, tumor_center)

        for col in range(3):
            slice_index, image_slice = image_slices[col]
            _, label_slice = label_slices[col]

            # MRI without segmentation
            ax = axes[orientation_index * 2, col]
            ax.imshow(
                image_slice,
                cmap="gray",
                origin="lower",
                vmin=low,
                vmax=high,
                interpolation="nearest"
            )
            ax.set_title(
                f"{orientation} | Slice {slice_index} | MRI only",
                fontsize=9,
                pad=2
            )
            ax.axis("off")

            # MRI with segmentation overlay
            ax = axes[orientation_index * 2 + 1, col]
            ax.imshow(
                image_slice,
                cmap="gray",
                origin="lower",
                vmin=low,
                vmax=high,
                interpolation="nearest"
            )

            overlay = np.ma.masked_where(
                (label_slice == 0) | (label_slice > 4),
                label_slice
            )

            ax.imshow(
                overlay,
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="nearest",
                alpha=0.45
            )

            ax.set_title(
                f"{orientation} | Slice {slice_index} | With tumor mask",
                fontsize=9,
                pad=2
            )
            ax.axis("off")

    # Compact legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[k])
        for k in COLORS
    ]

    fig.legend(
        handles,
        [f"{LABEL_NAMES[k]}" for k in COLORS],
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002)
    )

    fig.suptitle(
        f"{subject} | T1n MRI and Tumor Segmentation",
        fontsize=13,
        y=0.998
    )

    # Remove horizontal and vertical gaps between panels
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0.025,
        top=0.985,
        wspace=0,
        hspace=0
    )

    output_path = (
        OUTPUT_DIR /
        f"{subject}_T1n_tumor_segmentation_6rows.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
