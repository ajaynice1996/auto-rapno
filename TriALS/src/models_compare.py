# import os
# import nibabel as nib

# # Folder containing GT segmentations
# gt_path = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"

# # List all nii.gz files
# files = [f for f in os.listdir(gt_path) if f.endswith(".nii.gz")]

# print(f"📂 Found {len(files)} ground truth files.\n")

# for f in sorted(files):
#     case_id = f.replace(".nii.gz", "")
#     nii = nib.load(os.path.join(gt_path, f))
#     data = nii.get_fdata()
#     print(f"{case_id}: shape = {data.shape}")

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
# Change to your prediction folder and filename
pred_path = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"
filename = "BraTSPED_00003.nii.gz"

# -----------------------------
# Load NIfTI file
# -----------------------------
file_path = os.path.join(pred_path, filename)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found!")

nii = nib.load(file_path)
data = nii.get_fdata()

print(f"✅ Loaded {filename}")
print(f"Shape: {data.shape}")
print(f"Voxel size: {nii.header.get_zooms()} mm")

# Normalize for visualization
data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

# -----------------------------
# Orthogonal slice visualization
# -----------------------------
def show_orthoslices(volume, title="Orthogonal Slices"):
    """Display axial, coronal, sagittal slices from 3D MRI volume."""
    z, y, x = np.array(volume.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(volume[z, :, :], cmap='gray')
    axes[0].set_title(f'Axial (z={z})')
    axes[0].axis('off')

    axes[1].imshow(volume[:, y, :], cmap='gray')
    axes[1].set_title(f'Coronal (y={y})')
    axes[1].axis('off')

    axes[2].imshow(volume[:, :, x], cmap='gray')
    axes[2].set_title(f'Sagittal (x={x})')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Show the orthogonal views
# -----------------------------
show_orthoslices(data, title=f"Orthogonal slices - {filename}")
