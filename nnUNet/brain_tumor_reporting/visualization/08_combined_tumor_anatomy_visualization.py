from pathlib import Path
import argparse
import csv

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from nibabel.processing import resample_from_to
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "nnUNet_raw/Dataset202_BraTS/imagesTr_all"
LABEL_DIR = ROOT / "nnUNet_raw/Dataset202_BraTS/labelsTr_all"
SYNTHSEG_DIR = ROOT / "brain_tumor_reporting/outputs/synthseg"

OVERLAP_DIR = ROOT / "brain_tumor_reporting/outputs/tumor_anatomical_overlap"
OUTPUT_DIR = ROOT / "brain_tumor_reporting/outputs/visualization"


# BraTS tumor labels
TUMOR_LABELS = {
    1: ("ET", (1.0, 0.0, 0.0)),
    2: ("NET", (0.0, 0.8, 0.2)),
    3: ("CC", (0.0, 0.5, 1.0)),
    4: ("ED", (1.0, 0.8, 0.0)),
}


# SynthSeg anatomical labels
ANATOMY_LABELS = {
    10: ("Left thalamus", "cyan"),
    49: ("Right thalamus", "magenta"),
    14: ("Third ventricle", "yellow"),
    16: ("Brainstem", "white"),
    28: ("Left ventral DC", "orange"),
}


def load_on_t1_grid(path, reference, is_label=True):
    """Resample image onto the canonical T1 reference grid."""

    img = nib.as_closest_canonical(nib.load(str(path)))

    aligned = resample_from_to(
        img,
        (reference.shape, reference.affine),
        order=0 if is_label else 1,
    )

    data = aligned.get_fdata()

    if is_label:
        return np.rint(data).astype(np.int32)

    return data


def get_slice_indices(tumor):
    """Select the slice containing the largest WT area per orientation."""

    wt = tumor > 0

    if not np.any(wt):
        center = np.array(tumor.shape) // 2
        return tuple(int(x) for x in center)

    sagittal = int(np.argmax(np.sum(wt, axis=(1, 2))))
    coronal = int(np.argmax(np.sum(wt, axis=(0, 2))))
    axial = int(np.argmax(np.sum(wt, axis=(0, 1))))

    return sagittal, coronal, axial


def calculate_tumor_volumes(tumor, affine):
    """Calculate ET, NET, CC, ED, TC and WT volumes in mL."""

    voxel_volume_mm3 = abs(np.linalg.det(affine[:3, :3]))

    component_masks = {
        "ET": tumor == 1,
        "NET": tumor == 2,
        "CC": tumor == 3,
        "ED": tumor == 4,
        "TC": np.isin(tumor, [1, 2, 3]),
        "WT": np.isin(tumor, [1, 2, 3, 4]),
    }

    results = {}

    for name, mask in component_masks.items():

        voxel_count = int(np.count_nonzero(mask))

        volume_ml = (
            voxel_count * voxel_volume_mm3 / 1000.0
        )

        results[name] = {
            "voxels": voxel_count,
            "volume_ml": volume_ml,
            "status": "Present" if voxel_count > 0 else "Absent",
        }

    return results


def extract_plane(data, orientation, index):

    if orientation == "Axial":
        return data[:, :, index].T

    if orientation == "Coronal":
        return data[:, index, :].T

    return data[index, :, :].T


def make_tumor_overlay(tumor_plane):

    rgba = np.zeros((*tumor_plane.shape, 4), dtype=float)

    for label, (_, color) in TUMOR_LABELS.items():

        mask = tumor_plane == label

        rgba[mask, :3] = color
        rgba[mask, 3] = 0.65

    return rgba


def load_wt_overlap(csv_path):
    """Load WT anatomical overlap records from the CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing overlap CSV: {csv_path}"
        )

    records = []

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        required = {
            "tumor_component",
            "anatomy_label_id",
            "anatomy_label_name",
            "overlap_volume_ml",
            "percent_anatomy_occupied",
            "percent_tumor_in_anatomy",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row in reader:

            if row["tumor_component"] == "WT":
                records.append(row)

    if not records:
        raise ValueError("No WT rows found in overlap CSV.")

    return {
        int(row["anatomy_label_id"]): row
        for row in records
    }


def add_volume_summary(fig, tumor_volumes):
    """Display tumor volumes and presence/absence at the top."""

    names = ["ET", "NET", "CC", "ED", "TC", "WT"]

    positions = np.linspace(0.09, 0.91, len(names))

    for name, x in zip(names, positions):

        values = tumor_volumes[name]

        status = values["status"]
        volume = values["volume_ml"]

        fig.text(
            x,
            0.945,
            name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        fig.text(
            x,
            0.915,
            f"{volume:.3f} mL",
            ha="center",
            va="center",
            fontsize=11,
        )

        fig.text(
            x,
            0.887,
            status,
            ha="center",
            va="center",
            fontsize=10,
            color="green" if status == "Present" else "red",
            fontweight="bold",
        )


def add_quantitative_table(ax, overlap_data):

    ax.axis("off")

    columns = [
        "Anatomical region",
        "Overlap (mL)",
        "Anatomy occupied (%)",
        "Tumor in region (%)",
    ]

    rows = []

    for label_id, (name, _) in ANATOMY_LABELS.items():

        row = overlap_data.get(label_id)

        if row is None:

            rows.append([
                name,
                "N/A",
                "N/A",
                "N/A",
            ])

            continue

        rows.append([
            name,
            f"{float(row['overlap_volume_ml']):.3f}",
            f"{float(row['percent_anatomy_occupied']):.2f}",
            f"{float(row['percent_tumor_in_anatomy']):.2f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.27, 0.17, 0.28, 0.28],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    ax.set_title(
        "Whole tumor (WT) – anatomical overlap",
        fontsize=12,
        pad=12,
    )


def main(subject):

    t1_path = IMAGE_DIR / f"{subject}_0000.nii.gz"
    tumor_path = LABEL_DIR / f"{subject}.nii.gz"

    synthseg_path = (
        SYNTHSEG_DIR / f"{subject}_synthseg.nii.gz"
    )

    overlap_csv = (
        OVERLAP_DIR
        / subject
        / f"{subject}_tumor_anatomical_overlap.csv"
    )

    # Check all input files
    for path in (
        t1_path,
        tumor_path,
        synthseg_path,
        overlap_csv,
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    # Load canonical T1 reference
    t1_img = nib.as_closest_canonical(
        nib.load(str(t1_path))
    )

    t1 = t1_img.get_fdata()

    # Resample tumor and SynthSeg to T1 reference grid
    tumor = load_on_t1_grid(
        tumor_path,
        t1_img,
    )

    synthseg = load_on_t1_grid(
        synthseg_path,
        t1_img,
    )

    if not (
        t1.shape == tumor.shape == synthseg.shape
    ):
        raise ValueError(
            "T1, tumor, and SynthSeg arrays are not aligned."
        )

    # Calculate tumor component volumes
    tumor_volumes = calculate_tumor_volumes(
        tumor,
        t1_img.affine,
    )

    # Load quantitative WT overlap results
    overlap_data = load_wt_overlap(overlap_csv)

    # Select tumor-centered slices
    indices = get_slice_indices(tumor)

    orientations = [
        ("Axial", indices[2]),
        ("Coronal", indices[1]),
        ("Sagittal", indices[0]),
    ]

    # Robust MRI intensity window
    nonzero = t1[t1 > 0]

    if nonzero.size:
        vmin, vmax = np.percentile(
            nonzero,
            [1, 99],
        )
    else:
        vmin, vmax = 0, 1

    # Create figure
    # Create a larger figure with two MRI comparison rows and a table
    fig = plt.figure(figsize=(19, 12.5))

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.55],
        hspace=0.20,
        wspace=0.06,
        top=0.79,
        bottom=0.075,
        left=0.045,
        right=0.97,
    )

    # Row 1: Original T1n without segmentation
    axes_original = [
        fig.add_subplot(gs[0, i])
        for i in range(3)
    ]

    # Row 2: T1n with tumor and anatomical overlap
    axes_overlay = [
        fig.add_subplot(gs[1, i])
        for i in range(3)
    ]

    # Draw axial, coronal, sagittal views
    # ============================================================
    # ROW 1: ORIGINAL T1 MRI WITHOUT TUMOR OR ANATOMY OVERLAY
    # ROW 2: SAME T1 MRI WITH TUMOR + ANATOMICAL OVERLAP
    # ============================================================

    for ax_original, ax_overlay, (orientation, index) in zip(
        axes_original,
        axes_overlay,
        orientations,
    ):

        # Extract identical slices for both rows
        t1_plane = extract_plane(
            t1,
            orientation,
            index,
        )

        tumor_plane = extract_plane(
            tumor,
            orientation,
            index,
        )

        anatomy_plane = extract_plane(
            synthseg,
            orientation,
            index,
        )

        # --------------------------------------------------------
        # ROW 1: Original T1n MRI
        # --------------------------------------------------------

        ax_original.imshow(
            t1_plane,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        ax_original.set_title(
            f"Original T1n — {orientation} slice {index}",
            fontsize=12,
            fontweight="bold",
        )

        ax_original.axis("off")

        # --------------------------------------------------------
        # ROW 2: Tumor segmentation + anatomical boundaries
        # --------------------------------------------------------

        ax_overlay.imshow(
            t1_plane,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        # Tumor segmentation overlay
        ax_overlay.imshow(
            make_tumor_overlay(tumor_plane),
            origin="lower",
        )

        # SynthSeg anatomical boundaries
        for label_id, (name, color) in ANATOMY_LABELS.items():

            mask = anatomy_plane == label_id

            if np.any(mask):

                ax_overlay.contour(
                    mask.astype(float),
                    levels=[0.5],
                    colors=[color],
                    linewidths=1.1,
                    origin="lower",
                )

        ax_overlay.set_title(
            f"Tumor + anatomical overlap — {orientation} slice {index}",
            fontsize=12,
            fontweight="bold",
        )

        ax_overlay.axis("off")

    # Tumor and anatomy legends
    tumor_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=name,
        )
        for name, color in TUMOR_LABELS.values()
    ]

    anatomy_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2,
            label=name,
        )
        for name, color in ANATOMY_LABELS.values()
    ]

    fig.legend(
        handles=tumor_handles + anatomy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=9,
        fontsize=9,
        frameon=True,
    )

    # Tumor volume summary at the top
    add_volume_summary(
        fig,
        tumor_volumes,
    )

    # Quantitative anatomical overlap table
    table_ax = fig.add_subplot(gs[2, :])

    add_quantitative_table(
        table_ax,
        overlap_data,
    )

    fig.suptitle(
        f"{subject}: Tumor burden and anatomical overlap",
        fontsize=16,
        y=0.99,
    )

    fig.text(
        0.5,
        0.025,
        "Spatial overlap does not establish histological tumor invasion.",
        ha="center",
        fontsize=10,
        style="italic",
    )

    # Save figure
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{subject}_combined_tumor_anatomy.png"
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Terminal report
    print("\nSubject:", subject)
    print("T1 shape:", t1.shape)
    print("Tumor labels:", np.unique(tumor).tolist())
    print("SynthSeg labels:", len(np.unique(synthseg)))
    print("Selected slices:", orientations)

    print("\n========== TUMOR VOLUME SUMMARY ==========")

    for name, values in tumor_volumes.items():

        print(
            f"{name}: "
            f"{values['volume_ml']:.3f} mL | "
            f"{values['status']} | "
            f"{values['voxels']} voxels"
        )

    print("\nWT anatomical regions included:", len(overlap_data))

    print("\nCombined visualization saved:")
    print(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Combined tumor burden and anatomical overlap visualization"
        )
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID, e.g. BraTSPED_00002",
    )

    args = parser.parse_args()

    main(args.subject)

