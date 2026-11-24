import nibabel as nib
import numpy as np
from nibabel.viewers import OrthoSlicer3D

# --------------------------
# User inputs
# --------------------------
image_path = "nnUNet_raw/Dataset202_BraTS/imagesTr/BraTSPED_00010_0002.nii.gz"
mask_path = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation/BraTSPED_00010.nii.gz"

# --------------------------
# BraTS Label Definitions
# --------------------------
LABELS = {
    "ET": [1],            # Enhancing Tumor
    "TC": [1, 2, 3],      # Tumor Core
    "WT": [1, 2, 3, 4],   # Whole Tumor
}

# --------------------------
# Load image and mask
# --------------------------
img = nib.load(image_path)
mask = nib.load(mask_path)
img_data = img.get_fdata()
mask_data = mask.get_fdata()

print(f"✅ Loaded MRI: {img_data.shape}, Segmentation: {mask_data.shape}")

# --------------------------
# Normalize MRI
# --------------------------
img_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

# --------------------------
# Choose mode: 'single' or 'all'
# --------------------------
mode = "all"  # Options: "ET", "TC", "WT", or "all"

# --------------------------
# Generate overlay
# --------------------------
if mode in LABELS:
    # --- Single label visualization ---
    label_vals = LABELS[mode]
    mask_binary = np.isin(mask_data, label_vals).astype(np.float32)
    overlay = np.clip(img_norm + 0.5 * mask_binary, 0, 1)
    print(f"🔍 Viewing {mode} region")

elif mode == "all":
    # --- Multi-label color-coded overlay ---
    et_mask = np.isin(mask_data, LABELS["ET"]).astype(np.float32)
    tc_mask = np.isin(mask_data, LABELS["TC"]).astype(np.float32)
    wt_mask = np.isin(mask_data, LABELS["WT"]).astype(np.float32)

    # Create RGB overlay
    overlay = np.zeros((*img_data.shape, 3), dtype=np.float32)
    overlay[..., 0] = np.clip(img_norm + 0.5 * et_mask, 0, 1)  # Red: ET
    overlay[..., 1] = np.clip(img_norm + 0.5 * tc_mask, 0, 1)  # Green: TC
    overlay[..., 2] = np.clip(img_norm + 0.5 * wt_mask, 0, 1)  # Blue: WT

    print("🌈 Viewing all labels (ET=Red, TC=Green, WT=Blue)")

else:
    raise ValueError(f"Unknown mode '{mode}'. Use 'ET', 'TC', 'WT', or 'all'.")

# --------------------------
# Display using OrthoSlicer3D
# --------------------------
viewer = OrthoSlicer3D(overlay)
viewer.show()
