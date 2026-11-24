"""Predict tumor segmentations with the trained nnU-Net for the cases of a split.

    python predict_segmentations.py --split_file data/splits/nnunet_fold2.json --split test --out_dir data/nnunet_pred_fold2_test

The output folder can then be passed to evaluation_segmentation.py --pred_seg_dir.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from i2r.nnunet_utils import DEFAULT_NNUNET_DIR, predict_segmentations  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--split_file", default=str(ROOT / "data/splits/nnunet_fold2.json"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nnunet_dir", default=str(DEFAULT_NNUNET_DIR))
    ap.add_argument("--fold", default="2")
    args = ap.parse_args()
    pairs = json.load(open(Path(args.data_dir) / "pairs.json"))
    ids = json.load(open(args.split_file))[args.split]
    predict_segmentations({c: pairs[c]["modalities"] for c in ids}, args.out_dir, args.nnunet_dir, fold=args.fold)
