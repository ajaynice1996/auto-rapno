#!/usr/bin/env python
"""Publication figure of one case: rows = modalities (0000..0003), columns =
Input (no overlay), WT, TC, ET, the combined prediction overlay and the combined ground-truth overlay. Each region
column uses the axial slice with the largest ground-truth area of that region
(combined column: the WT slice). The yellow DSC is the volumetric Dice of the
whole case. No spacing between panels.

Colours: WT = green, TC = blue, ET = red. Combined: ED green, core (TC minus
ET) blue, ET red.

Usage:
    python visualize_case.py --case BraTSPED_00003 [--pred-dir DIR] [--out results/fig.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

RAW = Path("nnUNet/nnUNet_raw/Dataset202_BraTS/imagesTr")
GT_DIR = Path("nnUNet/nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations")
DEFAULT_PRED = Path("nnUNet/nnUNet_results/Dataset202_BraTS/"
                    "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation")
MODALITIES = {0: "T1n", 1: "T1c", 2: "T2-FLAIR", 3: "T2w"}  # from dataset.json
REGIONS = {"WT": [1, 2, 3, 4], "TC": [1, 2, 3], "ET": [1]}
COLORS = {"WT": (0, 1, 0), "TC": (0, 0.4, 1), "ET": (1, 0, 0)}


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def norm(sl):
    nz = sl[sl > 0]
    if nz.size == 0:
        return sl
    lo, hi = np.percentile(nz, [1, 99.5])
    return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)


def orient(x):
    return np.flipud(np.rot90(x))


def overlay(gray, layers, alpha=0.6):
    rgb = np.stack([gray] * 3, -1)
    for mask, col in layers:
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(col)
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs="+", default=["BraTSPED_00003"], help="one or more case ids")
    ap.add_argument("--pred-dir", default=str(DEFAULT_PRED))
    ap.add_argument("--out", default=None, help="output file (single case only)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    for case in args.case:
        render(case, args)


def render(case, args):
    pred = np.asarray(nib.load(str(Path(args.pred_dir) / f"{case}.nii.gz")).dataobj).astype(np.int16)
    gt = np.asarray(nib.load(str(GT_DIR / f"{case}.nii.gz")).dataobj).astype(np.int16)
    imgs = {m: np.asarray(nib.load(str(RAW / f"{case}_{m:04d}.nii.gz")).dataobj).astype(np.float32)
            for m in MODALITIES}

    pm = {r: np.isin(pred, ids) for r, ids in REGIONS.items()}
    gm = {r: np.isin(gt, ids) for r, ids in REGIONS.items()}
    dsc = {r: dice(pm[r], gm[r]) for r in REGIONS}
    zsl = {r: int(np.argmax(gm[r].sum((0, 1)))) if gm[r].any() else int(np.argmax(pm[r].sum((0, 1))))
           for r in REGIONS}
    zsl["ALL"] = zsl["GT"] = zsl["WT"]

    # common in-plane crop around the brain
    zs = sorted(set(zsl.values()))
    brain = np.any([im[:, :, z] > 0.1 * np.percentile(im[im > 0], 99) for im in imgs.values() for z in zs], axis=0)
    xs, ys = np.where(brain)
    crop = (slice(xs.min(), xs.max() + 1), slice(ys.min(), ys.max() + 1))

    cols = ["IN", "WT", "TC", "ET", "ALL", "GT"]
    h, w = orient(imgs[0][crop + (0,)]).shape
    margin = 0.18  # label strip width, in units of one panel
    fig, axes = plt.subplots(len(MODALITIES), len(cols) + 1, figsize=((len(cols) + margin) * 2.2, len(MODALITIES) * 2.2 * h / w),
                             gridspec_kw={"width_ratios": [margin] + [1] * len(cols)})
    fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)

    for i, m in enumerate(MODALITIES):
        for j, c in enumerate(cols):
            z = zsl["WT" if c == "IN" else c]
            gray = orient(norm(imgs[m][crop + (z,)]))
            if c == "IN":
                layers, txt = [], "Input"
            elif c in ("ALL", "GT"):
                p = (pred if c == "ALL" else gt)[crop + (z,)]
                layers = [(p == 4, COLORS["WT"]), (np.isin(p, [2, 3]), COLORS["TC"]), (p == 1, COLORS["ET"])]
                layers = [(orient(mk), col) for mk, col in layers]
                txt = "  ".join(f"{r} {dsc[r]:.2f}" for r in REGIONS) if c == "ALL" else "Ground truth"
            else:
                layers = [(orient(pm[c][crop + (z,)]), COLORS[c])]
                txt = f"{c}  DSC {dsc[c]:.2f}"
            ax = axes[i, j + 1]
            ax.imshow(overlay(gray, layers), interpolation="nearest")
            ax.set_axis_off()
            if i == 0:
                ax.text(0.5, 0.985, txt, color="white" if c in ("IN", "GT") else "yellow", ha="center", va="top",
                        transform=ax.transAxes, fontsize=7 if c == "ALL" else 9, fontweight="bold")
        lab = axes[i, 0]
        lab.set_axis_off()
        lab.text(0.5, 0.5, MODALITIES[m], color="white", ha="center", va="center", rotation=90,
                 transform=lab.transAxes, fontsize=18, fontweight="bold")

    out = Path(args.out or f"results/{case}_figure.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="black", pad_inches=0)
    print(f"Saved {out}  DSC: " + ", ".join(f"{r}={v:.4f}" for r, v in dsc.items()) + f"  slices: {zsl}")


if __name__ == "__main__":
    main()
