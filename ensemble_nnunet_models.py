#!/usr/bin/env python
"""Search for the best majority-vote ensemble of 3 or 4 models.

Only hard label maps are available (no softmax .npz), so ensembling is a
voxel-wise majority vote; ties are broken by the order of the models given
(earlier model wins). Every 3- and 4-model combination is scored on all
validation cases for ET, NET, CC, ED, TC and WT (Dice, IoU, HD, HD95).

Outputs (in --out-dir):
  ensemble_all_combinations.csv   every combination x region
  ensemble_best_per_region.csv    best combination per region (by --rank-by)
  ensemble_best_overall.csv       best combination by mean Dice over regions
"""
import argparse
import csv
import itertools
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_nnunet_models import LABELS, case_metrics

BASE = Path("nnUNet/nnUNet_results/Dataset202_BraTS")
GT_DIR = Path("nnUNet/nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations")
# (name, folder, validation subfolder)
MODELS = [
    ("nnUNet", "nnUNetTrainer__nnUNetPlans__3d_fullres", "validation"),
    ("ResEncL", "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres", "validation"),
    ("PrimusV2S", "nnUNet_PrimusV2S_Trainer__nnUNetPlans__3d_fullres", "validation"),
    ("PrimusV3S", "nnUNet_PrimusV3S_Trainer__nnUNetPlans__3d_fullres", "validation"),
    ("MedNeXt_k3", "nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1", "validation_raw_postprocessed"),
    ("MedNeXt_k5", "nnUNetTrainerV2_MedNeXt_S_kernel5__nnUNetPlansv2.1_trgSp_1x1x1", "validation_raw_postprocessed"),
]
NAMES = [m[0] for m in MODELS]
COMBOS = [c for k in (3, 4) for c in itertools.combinations(range(len(MODELS)), k)]
NUM_CLASSES = 5


def vote(preds, combo):
    scores = np.zeros((NUM_CLASSES,) + preds[0].shape, dtype=np.float32)
    for rank, i in enumerate(combo):
        w = 1.0 + 1e-3 * (len(combo) - rank)
        for c in range(NUM_CLASSES):
            scores[c] += w * (preds[i] == c)
    return scores.argmax(0).astype(np.int16)


def crop(pred, gt, margin=2):
    idx = np.argwhere(pred | gt)
    if len(idx) == 0:
        return pred, gt
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = idx.max(0) + margin + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return pred[sl], gt[sl]


def process_case(case):
    gimg = nib.load(str(GT_DIR / case))
    gt = np.asarray(gimg.dataobj).astype(np.int16)
    spacing = gimg.header.get_zooms()[:3]
    preds = [np.asarray(nib.load(str(BASE / f / "fold_2" / v / case)).dataobj).astype(np.int16) for _, f, v in MODELS]
    out = {}
    for combo in COMBOS:
        ens = vote(preds, combo)
        for name, ids in LABELS.items():
            p, g = crop(np.isin(ens, ids), np.isin(gt, ids))
            out[(combo, name)] = case_metrics(p, g, spacing)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--rank-by", default="Dice", choices=["Dice", "IoU"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cases = sorted(p.name for p in (BASE / MODELS[0][1] / "fold_2" / MODELS[0][2]).glob("*.nii.gz"))
    cases = [c for c in cases if all((BASE / f / "fold_2" / v / c).exists() for _, f, v in MODELS)]
    print(f"{len(cases)} cases, {len(COMBOS)} combinations", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(process_case, cases)

    rows = []
    for combo in COMBOS:
        for name in LABELS:
            arr = np.array([r[(combo, name)] for r in results], dtype=float)
            rows.append(("+".join(NAMES[i] for i in combo), len(combo), name, len(cases), *np.nanmean(arr, axis=0)))

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    hdr = ("ensemble", "n_models", "label", "n_cases", "Dice", "IoU", "HD", "HD95")
    fmt = lambda r: (*r[:4], *[f"{v:.4f}" for v in r[4:]])

    def write(fname, data):
        with open(out / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(fmt(r) for r in data)

    metric_idx = 4 if args.rank_by == "Dice" else 5
    write("ensemble_all_combinations.csv", rows)
    best = [max((r for r in rows if r[2] == name), key=lambda r: r[metric_idx]) for name in LABELS]
    write("ensemble_best_per_region.csv", best)

    by_ens = {}
    for r in rows:
        by_ens.setdefault(r[0], []).append(r)
    top = max(by_ens, key=lambda e: np.mean([r[metric_idx] for r in by_ens[e]]))
    write("ensemble_best_overall.csv", by_ens[top])

    print(f"\nBest ensemble per region (by {args.rank_by}):")
    for r in best:
        print("\t".join(map(str, fmt(r))))
    print(f"\nBest overall (mean {args.rank_by} over regions): {top}")
    for r in by_ens[top]:
        print("\t".join(map(str, fmt(r))))


if __name__ == "__main__":
    main()
