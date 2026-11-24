import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
# Load and normalize MRI
# --------------------------
img = nib.load(image_path)
mask = nib.load(mask_path)
img_data = img.get_fdata()
mask_data = mask.get_fdata()

img_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

# --------------------------
# Overlay preparation
# --------------------------
mode = "all"  # Options: "ET", "TC", "WT", or "all"

if mode in LABELS:
    label_vals = LABELS[mode]
    mask_binary = np.isin(mask_data, label_vals).astype(np.float32)
    overlay_rgb = np.stack([img_norm]*3, axis=-1)
    overlay_rgb[..., 1] = np.clip(img_norm + 0.5 * mask_binary, 0, 1)
elif mode == "all":
    et_mask = np.isin(mask_data, LABELS["ET"]).astype(np.float32)
    tc_mask = np.isin(mask_data, LABELS["TC"]).astype(np.float32)
    wt_mask = np.isin(mask_data, LABELS["WT"]).astype(np.float32)
    overlay_rgb = np.stack([img_norm]*3, axis=-1)
    overlay_rgb[..., 0] = np.clip(img_norm + 0.5 * et_mask, 0, 1)  # Red
    overlay_rgb[..., 1] = np.clip(img_norm + 0.5 * tc_mask, 0, 1)  # Green
    overlay_rgb[..., 2] = np.clip(img_norm + 0.5 * wt_mask, 0, 1)  # Blue
else:
    raise ValueError(f"Unknown mode '{mode}'")

# --------------------------
# Setup figure
# --------------------------
n_slices_axial = img_data.shape[2]
init_slice = n_slices_axial // 2

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
plt.subplots_adjust(bottom=0.15, hspace=0.2)

# --------------------------
# Initial display
# --------------------------
# Top row: original
axial_orig = axes[0, 0].imshow(img_norm[:, :, init_slice], cmap="gray")
axes[0, 0].set_title("Axial - Original")
sagittal_orig = axes[0, 1].imshow(img_norm[init_slice, :, :], cmap="gray")
axes[0, 1].set_title("Sagittal - Original")
coronal_orig = axes[0, 2].imshow(img_norm[:, init_slice, :], cmap="gray")
axes[0, 2].set_title("Coronal - Original")

# Bottom row: overlay
axial_overlay = axes[1, 0].imshow(overlay_rgb[:, :, init_slice])
axes[1, 0].set_title("Axial - Overlay")
sagittal_overlay = axes[1, 1].imshow(overlay_rgb[init_slice, :, :])
axes[1, 1].set_title("Sagittal - Overlay")
coronal_overlay = axes[1, 2].imshow(overlay_rgb[:, init_slice, :])
axes[1, 2].set_title("Coronal - Overlay")

for ax in axes.flatten():
    ax.axis("off")

# --------------------------
# Slider setup
# --------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, "Slice", 0, n_slices_axial - 1, valinit=init_slice, valstep=1)

# Rotate slices 90 degrees for correct orientation
def rotate90(slice_2d):
    return np.rot90(slice_2d)

# In the update function and initial display, wrap each slice with rotate90:
# Top row: original
axial_orig = axes[0, 0].imshow(rotate90(img_norm[:, :, init_slice]), cmap="gray")
sagittal_orig = axes[0, 1].imshow(rotate90(img_norm[init_slice, :, :]), cmap="gray")
coronal_orig = axes[0, 2].imshow(rotate90(img_norm[:, init_slice, :]), cmap="gray")

# Bottom row: overlay
axial_overlay = axes[1, 0].imshow(rotate90(overlay_rgb[:, :, init_slice]))
sagittal_overlay = axes[1, 1].imshow(rotate90(overlay_rgb[init_slice, :, :]))
coronal_overlay = axes[1, 2].imshow(rotate90(overlay_rgb[:, init_slice, :]))

def update(val):
    slice_idx = int(slider.val)
    # Top row: original
    axial_orig.set_data(rotate90(img_norm[:, :, slice_idx]))
    sagittal_orig.set_data(rotate90(img_norm[slice_idx, :, :]))
    coronal_orig.set_data(rotate90(img_norm[:, slice_idx, :]))
    # Bottom row: overlay
    axial_overlay.set_data(rotate90(overlay_rgb[:, :, slice_idx]))
    sagittal_overlay.set_data(rotate90(overlay_rgb[slice_idx, :, :]))
    coronal_overlay.set_data(rotate90(overlay_rgb[:, slice_idx, :]))
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
