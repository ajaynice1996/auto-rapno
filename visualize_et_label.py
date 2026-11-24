"""Confirm which label value is ET (expected: 1) by overlaying each label on T1ce (_0001)."""
import sys
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "/home/ashar/Documents/auto-rapno/nnUNet/nnUNet_raw/Dataset202_BraTS"
case = sys.argv[1] if len(sys.argv) > 1 else "BraTSPED_00001"
out = sys.argv[2] if len(sys.argv) > 2 else f"{RAW}/ET_check_{case}.png"

img = nib.load(f"{RAW}/imagesTr/{case}_0001.nii.gz").get_fdata()
seg = nib.load(f"{RAW}/labelsTr/{case}.nii.gz").get_fdata().astype(int)
print("labels present:", {int(l): int((seg == l).sum()) for l in np.unique(seg)})

# slice with most label-1 voxels (axial)
z = int(np.argmax((seg == 1).sum(axis=(0, 1))))
sl, sg = img[:, :, z].T, seg[:, :, z].T
names = {1: "label 1 (ET?)", 2: "label 2 (NET)", 3: "label 3 (CC)", 4: "label 4 (ED)"}
colors = {1: "red", 2: "yellow", 3: "cyan", 4: "lime"}

fig, ax = plt.subplots(1, 6, figsize=(24, 4.5))
ax[0].imshow(sl, cmap="gray"); ax[0].set_title(f"T1ce (_0001) z={z}")
for i, l in enumerate([1, 2, 3, 4], start=1):
    ax[i].imshow(sl, cmap="gray")
    m = np.ma.masked_where(sg != l, sg)
    ax[i].imshow(m, cmap=matplotlib.colors.ListedColormap([colors[l]]), alpha=0.7)
    ax[i].set_title(names[l])
ax[5].imshow(sl, cmap="gray")
ax[5].imshow(np.ma.masked_where(sg != 1, sg), cmap=matplotlib.colors.ListedColormap(["red"]), alpha=0.8)
ax[5].contour(sg > 0, colors="white", linewidths=0.6)
ax[5].set_title("ET=1 (red) in whole tumor")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved", out)
