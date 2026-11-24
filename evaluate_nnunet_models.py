#!/usr/bin/env python
"""Compute Dice, IoU and Hausdorff distance (HD, HD95) on the fold validation
predictions of every nnUNet model found under a results base folder.

Usage:
    python evaluate_nnunet_models.py \
        --base nnUNet/nnUNet_results/Dataset202_BraTS \
        --gt nnUNet/nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations \
        [--fold 2] [--val-dirs validation] [--models "*"] [--out metrics.csv]

Layout expected: <base>/<model>/fold_<N>/validation/<case>.nii.gz
Metrics are computed per case, per label (1..4) plus Tumor Core (TC: 1+2+3) and Whole Tumor (WT: 1+2+3+4),
then averaged over the whole validation set. HD is in mm (uses voxel spacing).
Cases where both masks are empty score 1 (Dice/IoU) and 0 (HD); cases where
only one is empty score 0 (Dice/IoU) and are excluded from HD averages.
"""
import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

LABELS = {"ET": [1], "NET": [2], "CC": [3], "ED": [4], "TC": [1, 2, 3], "WT": [1, 2, 3, 4]}


def surface_distances(a, b, spacing):
    """Distances from surface voxels of a to the surface of b."""
    a_surf = a & ~ndimage.binary_erosion(a)
    b_surf = b & ~ndimage.binary_erosion(b)
    dt_b = ndimage.distance_transform_edt(~b_surf, sampling=spacing)
    return dt_b[a_surf]


def case_metrics(pred, gt, spacing):
    p_sum, g_sum = pred.sum(), gt.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0, 0.0, 0.0
    if p_sum == 0 or g_sum == 0:
        return 0.0, 0.0, np.nan, np.nan
    inter = np.logical_and(pred, gt).sum()
    dice = 2.0 * inter / (p_sum + g_sum)
    iou = inter / (p_sum + g_sum - inter)
    d = np.concatenate([surface_distances(pred, gt, spacing), surface_distances(gt, pred, spacing)])
    return dice, iou, d.max(), np.percentile(d, 95)


def evaluate_model(val_dir, gt_dir):
    rows = []
    for pf in sorted(val_dir.glob("*.nii.gz")):
        gf = gt_dir / pf.name
        if not gf.exists():
            print(f"  [warn] no GT for {pf.name}, skipped", file=sys.stderr)
            continue
        pimg, gimg = nib.load(str(pf)), nib.load(str(gf))
        pred = np.asarray(pimg.dataobj).astype(np.int16)
        gt = np.asarray(gimg.dataobj).astype(np.int16)
        if pred.shape != gt.shape:
            print(f"  [warn] shape mismatch for {pf.name}, skipped", file=sys.stderr)
            continue
        spacing = gimg.header.get_zooms()[:3]
        for name, ids in LABELS.items():
            m = case_metrics(np.isin(pred, ids), np.isin(gt, ids), spacing)
            rows.append((pf.name, name, *m))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Folder containing the trained model folders")
    ap.add_argument("--gt", default=None, help="Ground-truth folder (default: sibling nnUNet_preprocessed/<dataset>/gt_segmentations)")
    ap.add_argument("--fold", default="2")
    ap.add_argument("--val-dirs", nargs="+", default=["validation"],
                    help="Validation subfolder names inside fold_N (e.g. validation_raw validation_raw_postprocessed)")
    ap.add_argument("--models", default="*", help="Glob to select model folders, e.g. '*MedNeXt*'")
    ap.add_argument("--out", default="nnunet_metrics.csv", help="CSV with the summary per model/label")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    gt_dir = Path(args.gt) if args.gt else base.parents[1] / "nnUNet_preprocessed" / base.name / "gt_segmentations"
    if not gt_dir.is_dir():
        sys.exit(f"GT folder not found: {gt_dir}")

    summary = []
    for model in sorted(p for p in base.glob(args.models) if p.is_dir()):
        for vname in args.val_dirs:
            val_dir = model / f"fold_{args.fold}" / vname
            if not val_dir.is_dir():
                continue
            print(f"Evaluating {model.name}/{vname} ...")
            rows = evaluate_model(val_dir, gt_dir)
            for label in LABELS:
                sel = [r for r in rows if r[1] == label]
                if not sel:
                    continue
                arr = np.array([r[2:] for r in sel], dtype=float)
                mean = np.nanmean(arr, axis=0)
                summary.append((model.name, vname, label, len(sel), *mean))

    hdr = ("model", "validation_dir", "label", "n_cases", "Dice", "IoU", "HD", "HD95")
    print("\n" + "\t".join(hdr))
    for r in summary:
        print("\t".join([r[0], r[1], r[2], str(r[3])] + [f"{v:.4f}" for v in r[4:]]))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows([(*r[:4], *[f"{v:.4f}" for v in r[4:]]) for r in summary])
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
