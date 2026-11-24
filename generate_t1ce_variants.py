"""Generate T1ce-derived variant channels to help ET localization, plus PNG previews.

Usage:
  python generate_t1ce_variants.py                 # all cases -> NIfTI + previews for first N
  python generate_t1ce_variants.py CASE [CASE ...] # specific cases

Outputs (OUT):
  nifti/<case>_<variant>.nii.gz   (float32, same grid as input; can be added as extra nnUNet channels)
  png/<case>_variants.png         (montage on the slice with most ET, ET contour in red)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

RAW = Path("/home/ashar/Documents/auto-rapno/nnUNet/nnUNet_raw/Dataset202_BraTS")
OUT = Path("/home/ashar/Documents/auto-rapno/outputs/t1ce_variants")
N_PREVIEW = 10
EPS = 1e-3


def zscore(x, m):
    v = x[m]
    return np.where(m, (x - v.mean()) / (v.std() + 1e-8), 0).astype(np.float32)


def robust01(x, m, lo=1, hi=99.5):
    a, b = np.percentile(x[m], [lo, hi])
    return np.where(m, np.clip((x - a) / (b - a + 1e-8), 0, 1), 0).astype(np.float32)


def brain_mask(*mods):
    # Average of per-modality normalised, smoothed volumes; threshold; keep largest component
    m = sum(ndi.gaussian_filter(x / (np.percentile(x[x > 0], 99) + 1e-8), 1.5) for x in mods) / len(mods)
    m = m > 0.12
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    return ndi.binary_fill_holes(m)


def make_variants(t1n, t1c, t2f, t2w):
    mask = brain_mask(t1n, t1c, t2f, t2w)
    # Match T1n intensity scale to T1c (robust linear fit inside brain), so subtraction isolates enhancement
    n, c = t1n[mask], t1c[mask]
    slope = np.median(c) / (np.median(n) + 1e-8)
    t1n_s = t1n * slope

    zc, zn = zscore(t1c, mask), zscore(t1n, mask)
    zf, zw = zscore(t2f, mask), zscore(t2w, mask)
    v = {}
    v["t1c_norm"] = robust01(t1c, mask)
    v["t1c_minus_t1n"] = np.where(mask, t1c - t1n_s, 0).astype(np.float32)
    v["t1c_minus_t1n_pos"] = np.maximum(v["t1c_minus_t1n"], 0)
    v["t1c_z_minus_t1n_z"] = (zc - zn) * mask
    v["t1c_rel_enh"] = np.where(mask, np.clip((t1c - t1n_s) / (t1n_s + EPS * t1n_s[mask].mean()), -1, 5), 0).astype(np.float32)
    v["t1c_over_t1n_log"] = np.where(mask, np.log((t1c + EPS * c.mean()) / (t1n_s + EPS * c.mean())), 0).astype(np.float32)
    v["t1c_inverse"] = ((1 - v["t1c_norm"]) * mask).astype(np.float32)
    v["t1c_gamma2"] = (v["t1c_norm"] ** 2).astype(np.float32)  # suppresses mid-range, keeps bright enhancement
    v["t1c_hi_clip"] = np.where(mask, np.clip(zc, 1.0, 4.0) - 1.0, 0).astype(np.float32)  # only bright voxels
    v["t1c_minus_t2f"] = (zc - zf) * mask  # enhancing but not just edema/FLAIR-bright
    v["t1c_minus_t2w"] = (zc - zw) * mask
    v["t1c_x_enh"] = (v["t1c_norm"] * robust01(v["t1c_minus_t1n_pos"], mask)).astype(np.float32)
    s = ndi.gaussian_filter(t1c, 3)
    v["t1c_dog"] = np.where(mask, ndi.gaussian_filter(t1c, 1) - s, 0).astype(np.float32)  # local contrast
    sm = ndi.gaussian_filter(t1c, 1)
    v["t1c_gradmag"] = ndi.gaussian_gradient_magnitude(t1c, 1).astype(np.float32) * mask
    v["t1c_local_z"] = np.where(
        mask, (t1c - ndi.uniform_filter(t1c, 15)) / (np.sqrt(np.maximum(ndi.uniform_filter(t1c**2, 15) - ndi.uniform_filter(t1c, 15) ** 2, 0)) + 1e-3), 0
    ).astype(np.float32)
    v["t1c_tophat"] = ndi.white_tophat(sm, size=(7, 7, 3)).astype(np.float32) * mask  # small bright structures
    return v


def load(case, suffix):
    return nib.load(RAW / "imagesTr" / f"{case}_{suffix}.nii.gz")


def process(case, preview):
    img = load(case, "0001")
    t1n, t1c, t2f, t2w = (load(case, s).get_fdata(dtype=np.float32) for s in ("0000", "0001", "0002", "0003"))
    v = make_variants(t1n, t1c, t2f, t2w)
    (OUT / "nifti").mkdir(parents=True, exist_ok=True)
    for k, a in v.items():
        nib.save(nib.Nifti1Image(a, img.affine, img.header), OUT / "nifti" / f"{case}_{k}.nii.gz")
    if not preview:
        return
    seg = nib.load(RAW / "labelsTr" / f"{case}.nii.gz").get_fdata().astype(int)
    et = seg == 1
    z = int(np.argmax(et.sum(axis=(0, 1)))) if et.any() else t1c.shape[2] // 2
    names = ["t1n", "t1c", "t2f"] + list(v)
    imgs = [t1n, t1c, t2f] + list(v.values())
    cols = 6
    rows = int(np.ceil(len(names) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    for a in ax.ravel():
        a.axis("off")
    for a, nm, im in zip(ax.ravel(), names, imgs):
        s = im[:, :, z].T
        lo, hi = np.percentile(im[:, :, z], [1, 99.7])
        a.imshow(s, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        if et[:, :, z].any():
            a.contour(et[:, :, z].T, levels=[0.5], colors="red", linewidths=0.8)
        a.set_title(nm, fontsize=9)
    fig.suptitle(f"{case}  axial z={z}  (red = ET label)")
    plt.tight_layout()
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / "png" / f"{case}_variants.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    cases = sys.argv[1:]
    if cases:
        for c in cases:
            process(c, True)
            print(c)
    else:
        allc = sorted(p.name[:-len("_0001.nii.gz")] for p in (RAW / "imagesTr").glob("*_0001.nii.gz"))
        # previews for cases that actually have ET
        for i, c in enumerate(allc):
            process(c, i < N_PREVIEW)
            print(c, flush=True)
    print("done ->", OUT)
