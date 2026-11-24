"""Figure for the ET variants README: one example case, ET-slice view of the 4 channels of imagesTr_ET."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

RAW = "/home/ashar/Documents/auto-rapno/nnUNet/nnUNet_raw/Dataset202_BraTS"
case = sys.argv[1] if len(sys.argv) > 1 else "BraTSPED_00010"
seg = nib.load(f"{RAW}/labelsTr_ET/{case}.nii.gz").get_fdata().astype(int)
et = seg == 1
z = int(np.argmax(et.sum(axis=(0, 1))))
t1n = nib.load(f"{RAW}/imagesTr/{case}_0000.nii.gz").get_fdata(dtype=np.float32)
chs = [("t1n (reference, not in ET set)", t1n)] + [
    (n, nib.load(f"{RAW}/imagesTr_ET/{case}_{k}.nii.gz").get_fdata(dtype=np.float32))
    for k, n in [("0000", "_0000 t1c_z_minus_t1n_z"), ("0001", "_0001 t1c (original)"),
                 ("0002", "_0002 t1c_rel_enh"), ("0003", "_0003 t1c_local_z")]
]
ys, xs = np.where(et[:, :, z].T)
cy, cx = int(ys.mean()), int(xs.mean())
h = 45
fig, ax = plt.subplots(2, 5, figsize=(20, 8.4))
for j, (n, im) in enumerate(chs):
    s = im[:, :, z].T
    lo, hi = np.percentile(im[:, :, z], [1, 99.7])
    for i in range(2):
        a = ax[i, j]
        a.imshow(s, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        a.contour(et[:, :, z].T, levels=[0.5], colors="red", linewidths=0.9)
        a.axis("off")
        if i == 0:
            a.set_title(n, fontsize=10)
        else:
            a.set_xlim(cx - h, cx + h); a.set_ylim(cy - h, cy + h)
fig.suptitle(f"{case}, axial z={z}. Top: full slice, bottom: zoom on ET. Red = ET label (1)", fontsize=13)
plt.tight_layout()
out = f"{RAW}/imagesTr_ET_example.png"
plt.savefig(out, dpi=100)
print(out)
