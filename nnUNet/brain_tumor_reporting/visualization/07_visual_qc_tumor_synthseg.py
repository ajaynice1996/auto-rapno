from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "nnUNet_raw/Dataset202_BraTS/imagesTr_all"
LABEL_DIR = ROOT / "nnUNet_raw/Dataset202_BraTS/labelsTr_all"
SYNTHSEG_DIR = ROOT / "brain_tumor_reporting/outputs/synthseg"
OUTPUT_DIR = ROOT / "brain_tumor_reporting/outputs/visual_qc"


# BraTS tumor labels
TUMOR_LABELS = {
    1: ("ET", (1.0, 0.0, 0.0)),
    2: ("NET", (0.0, 0.8, 0.2)),
    3: ("CC", (0.0, 0.5, 1.0)),
    4: ("ED", (1.0, 0.8, 0.0)),
}

# Selected SynthSeg anatomical labels for boundary inspection
ANATOMY_LABELS = {
    10: "Left thalamus",
    49: "Right thalamus",
    14: "Third ventricle",
    16: "Brainstem",
}


def load_on_t1_grid(path, reference, is_label=True):
    img = nib.as_closest_canonical(nib.load(str(path)))
    aligned = resample_from_to(
        img,
        (reference.shape, reference.affine),
        order=0 if is_label else 1,
    )
    data = aligned.get_fdata()
    return np.rint(data).astype(np.int32) if is_label else data


def get_slice_indices(tumor):
    """Choose the slice with the largest WT area in each orientation."""
    wt = tumor > 0

    if not np.any(wt):
        center = np.array(tumor.shape) // 2
        return tuple(center)

    axial = np.argmax(np.sum(wt, axis=(0, 1)))
    coronal = np.argmax(np.sum(wt, axis=(0, 2)))
    sagittal = np.argmax(np.sum(wt, axis=(1, 2)))

    return int(sagittal), int(coronal), int(axial)


def extract_plane(data, orientation, index):
    if orientation == "Axial":
        return data[:, :, index].T
    if orientation == "Coronal":
        return data[:, index, :].T
    return data[index, :, :].T  # Sagittal


def make_rgba_overlay(tumor_plane):
    rgba = np.zeros((*tumor_plane.shape, 4), dtype=float)

    for label, (_, color) in TUMOR_LABELS.items():
        mask = tumor_plane == label
        rgba[mask, :3] = color
        rgba[mask, 3] = 0.65

    return rgba


def main(subject):
    t1_path = IMAGE_DIR / f"{subject}_0000.nii.gz"
    tumor_path = LABEL_DIR / f"{subject}.nii.gz"
    synthseg_path = SYNTHSEG_DIR / f"{subject}_synthseg.nii.gz"

    for path in (t1_path, tumor_path, synthseg_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    t1_img = nib.as_closest_canonical(nib.load(str(t1_path)))
    t1 = t1_img.get_fdata()

    tumor = load_on_t1_grid(tumor_path, t1_img)
    synthseg = load_on_t1_grid(synthseg_path, t1_img)

    if not (t1.shape == tumor.shape == synthseg.shape):
        raise ValueError("Image arrays are not aligned to the same grid.")

    indices = get_slice_indices(tumor)

    # Robust display window, excluding background
    nonzero = t1[t1 > 0]
    vmin, vmax = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)

    orientations = [
        ("Axial", indices[2]),
        ("Coronal", indices[1]),
        ("Sagittal", indices[0]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))

    for ax, (orientation, index) in zip(axes, orientations):
        t1_plane = extract_plane(t1, orientation, index)
        tumor_plane = extract_plane(tumor, orientation, index)
        anatomy_plane = extract_plane(synthseg, orientation, index)

        ax.imshow(t1_plane, cmap="gray", origin="lower",
                  vmin=vmin, vmax=vmax)

        # Tumor subregion overlay
        ax.imshow(make_rgba_overlay(tumor_plane), origin="lower")

        # Outline selected anatomical structures
        for label, name in ANATOMY_LABELS.items():
            mask = (anatomy_plane == label).astype(float)
            if np.any(mask):
                ax.contour(
                    mask,
                    levels=[0.5],
                    colors=["white"],
                    linewidths=0.8,
                    origin="lower",
                )

        ax.set_title(f"{orientation} — slice {index}")
        ax.axis("off")

    tumor_legend = [
        Patch(facecolor=color, edgecolor="black", label=name)
        for name, color in TUMOR_LABELS.values()
    ]

    anatomy_legend = [
        Patch(facecolor="none", edgecolor="white", label=name)
        for name in ANATOMY_LABELS.values()
    ]

    fig.legend(
        handles=tumor_legend + anatomy_legend,
        loc="lower center",
        ncol=8,
        fontsize=9,
    )

    fig.suptitle(
        f"{subject}: T1 + tumor subregions + SynthSeg anatomical boundaries",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.94])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{subject}_visual_qc.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Subject: {subject}")
    print(f"T1 shape: {t1.shape}")
    print(f"Tumor labels present: {np.unique(tumor).tolist()}")
    print(f"SynthSeg labels present: {len(np.unique(synthseg))}")
    print(f"Selected slices: {orientations}")
    print(f"QC figure saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    main(args.subject)
