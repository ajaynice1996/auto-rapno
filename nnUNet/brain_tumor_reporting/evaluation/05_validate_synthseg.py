

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT_ID = "BraTSPED_00002"

T1_PATH = (
    PROJECT_ROOT
    / "nnUNet_raw/Dataset202_BraTS/imagesTr_all"
    / f"{SUBJECT_ID}_0000.nii.gz"
)

TUMOR_PATH = (
    PROJECT_ROOT
    / "nnUNet_raw/Dataset202_BraTS/labelsTr_all"
    / f"{SUBJECT_ID}.nii.gz"
)

SYNTHSEG_PATH = (
    PROJECT_ROOT
    / "brain_tumor_reporting/outputs/synthseg"
    / f"{SUBJECT_ID}_synthseg.nii.gz"
)

QC_PATH = (
    PROJECT_ROOT
    / "brain_tumor_reporting/outputs/synthseg"
    / f"{SUBJECT_ID}_qc.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "brain_tumor_reporting/outputs/synthseg/validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def print_image_info(name, img):
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print("Shape:", img.shape)
    print("Voxel spacing:", img.header.get_zooms()[:3])
    print("Orientation:", nib.aff2axcodes(img.affine))
    print("Affine matrix:\n", img.affine)


def save_image_info(name, img, report):
    report.append(f"\n{name}")
    report.append(f"Shape: {img.shape}")
    report.append(f"Voxel spacing: {img.header.get_zooms()[:3]}")
    report.append(f"Orientation: {nib.aff2axcodes(img.affine)}")
    report.append(f"Affine:\n{img.affine}")


# ---------------------------------------------------------
# 1. Check files
# ---------------------------------------------------------
files = {
    "Original T1": T1_PATH,
    "Tumor mask": TUMOR_PATH,
    "SynthSeg output": SYNTHSEG_PATH,
}

print("\nSYNTHSEG VALIDATION")
print("Subject:", SUBJECT_ID)

for name, path in files.items():
    print(f"{name}: {'FOUND' if path.exists() else 'MISSING'}")
    print(" ", path)

missing = [name for name, path in files.items() if not path.exists()]

if missing:
    raise FileNotFoundError(
        "Missing required files: " + ", ".join(missing)
    )


# ---------------------------------------------------------
# 2. Load NIfTI images
# ---------------------------------------------------------
t1_img = nib.load(str(T1_PATH))
tumor_img = nib.load(str(TUMOR_PATH))
synth_img = nib.load(str(SYNTHSEG_PATH))

report = [
    f"SynthSeg validation report",
    f"Subject: {SUBJECT_ID}",
]

for name, img in [
    ("Original T1", t1_img),
    ("Tumor mask", tumor_img),
    ("SynthSeg output", synth_img),
]:
    print_image_info(name, img)
    save_image_info(name, img, report)


# ---------------------------------------------------------
# 3. Inspect anatomical labels
# ---------------------------------------------------------
synth_data = np.asarray(synth_img.dataobj)

labels, counts = np.unique(
    synth_data.astype(np.int32),
    return_counts=True
)

print("\nSynthSeg unique label count:", len(labels))
print("\nLabel voxel counts:")

for label, count in zip(labels, counts):
    print(f"Label {label:4d}: {count:10d} voxels")

report.append(f"\nNumber of unique SynthSeg labels: {len(labels)}")

# Save label statistics
label_csv = OUTPUT_DIR / f"{SUBJECT_ID}_label_counts.csv"

with open(label_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "voxel_count"])
    writer.writerows(zip(labels.tolist(), counts.tolist()))


# ---------------------------------------------------------
# 4. Check physical-space alignment
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SPATIAL ALIGNMENT")
print("=" * 60)

print("Same shape:", t1_img.shape == synth_img.shape)
print(
    "Same affine:",
    np.allclose(t1_img.affine, synth_img.affine, atol=1e-4)
)

# Resample SynthSeg to original T1 grid using nearest-neighbor
# interpolation, preserving discrete anatomical labels.
synth_on_t1 = resample_from_to(
    synth_img,
    t1_img,
    order=0
)

# Resample tumor mask to original T1 grid as well.
tumor_on_t1 = resample_from_to(
    tumor_img,
    t1_img,
    order=0
)

synth_data_aligned = np.rint(
    synth_on_t1.get_fdata()
).astype(np.int32)

tumor_data_aligned = np.rint(
    tumor_on_t1.get_fdata()
).astype(np.int32)

t1_data = t1_img.get_fdata()

print("Resampled SynthSeg shape:", synth_data_aligned.shape)
print("Original T1 shape:", t1_data.shape)
print(
    "SynthSeg now shares T1 grid:",
    synth_on_t1.shape == t1_img.shape
    and np.allclose(synth_on_t1.affine, t1_img.affine)
)

print(
    "Tumor mask now shares T1 grid:",
    tumor_on_t1.shape == t1_img.shape
    and np.allclose(tumor_on_t1.affine, t1_img.affine)
)

report.append("\nSpatial alignment")
report.append(f"Original shape matches SynthSeg: {t1_img.shape == synth_img.shape}")
report.append(
    "Original affine matches SynthSeg: "
    f"{np.allclose(t1_img.affine, synth_img.affine, atol=1e-4)}"
)
report.append("SynthSeg resampled to original T1 grid using nearest-neighbor.")
report.append("Tumor mask resampled to original T1 grid using nearest-neighbor.")


# ---------------------------------------------------------
# 5. Select representative axial slice
# ---------------------------------------------------------
tumor_voxels = np.argwhere(tumor_data_aligned > 0)

if len(tumor_voxels) > 0:
    z = int(np.median(tumor_voxels[:, 2]))
    print("\nTumor-containing axial slice:", z)
else:
    z = t1_data.shape[2] // 2
    print("\nNo nonzero tumor labels found. Using center slice:", z)

report.append(f"Displayed axial slice index: {z}")


# ---------------------------------------------------------
# 6. Create visualization
# ---------------------------------------------------------
t1_slice = t1_data[:, :, z].T
synth_slice = synth_data_aligned[:, :, z].T
tumor_slice = tumor_data_aligned[:, :, z].T

# Robust display window for MRI
low, high = np.percentile(t1_data, [1, 99])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].imshow(
    t1_slice,
    cmap="gray",
    origin="lower",
    vmin=low,
    vmax=high
)
axes[0, 0].set_title("Original T1 MRI")

axes[0, 1].imshow(
    t1_slice,
    cmap="gray",
    origin="lower",
    vmin=low,
    vmax=high
)
axes[0, 1].imshow(
    synth_slice,
    cmap="nipy_spectral",
    alpha=0.40,
    origin="lower",
    interpolation="nearest"
)
axes[0, 1].set_title("T1 + SynthSeg anatomical labels")

axes[1, 0].imshow(
    t1_slice,
    cmap="gray",
    origin="lower",
    vmin=low,
    vmax=high
)
axes[1, 0].imshow(
    tumor_slice > 0,
    cmap="autumn",
    alpha=0.50,
    origin="lower",
    interpolation="nearest"
)
axes[1, 0].set_title("T1 + tumor mask")

axes[1, 1].imshow(
    t1_slice,
    cmap="gray",
    origin="lower",
    vmin=low,
    vmax=high
)
axes[1, 1].imshow(
    synth_slice,
    cmap="nipy_spectral",
    alpha=0.35,
    origin="lower",
    interpolation="nearest"
)
axes[1, 1].imshow(
    tumor_slice > 0,
    cmap="autumn",
    alpha=0.65,
    origin="lower",
    interpolation="nearest"
)
axes[1, 1].set_title("T1 + SynthSeg + tumor mask")

for ax in axes.flat:
    ax.axis("off")

fig.suptitle(
    f"{SUBJECT_ID} | Axial slice {z}",
    fontsize=15
)

plt.tight_layout()

figure_path = OUTPUT_DIR / f"{SUBJECT_ID}_synthseg_validation.png"

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)


# ---------------------------------------------------------
# 7. Save validation report
# ---------------------------------------------------------
report_path = OUTPUT_DIR / f"{SUBJECT_ID}_validation_report.txt"

with open(report_path, "w") as f:
    f.write("\n".join(report))

print("\n" + "=" * 60)
print("VALIDATION FILES SAVED")
print("=" * 60)
print("Visualization:", figure_path)
print("Label counts:", label_csv)
print("Report:", report_path)

if QC_PATH.exists():
    print("\nSynthSeg QC CSV:")
    with open(QC_PATH) as f:
        print(f.read())
else:
    print("\nSynthSeg QC CSV not found:", QC_PATH)

print("\nValidation script completed.")
