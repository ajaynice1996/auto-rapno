import os
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

# -----------------------------
# Load your NIfTI volume
# -----------------------------
pred_path = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"
filename = "BraTSPED_00010.nii.gz"  # replace with your file
file_path = os.path.join(pred_path, filename)

nii = nib.load(file_path)
data = nii.get_fdata()

print(f"✅ Loaded {filename}")
print(f"Shape: {data.shape}")
print(f"Voxel size: {nii.header.get_zooms()} mm")

# -----------------------------
# Launch interactive orthoslicer
# -----------------------------
viewer = OrthoSlicer3D(data)
viewer.show()
