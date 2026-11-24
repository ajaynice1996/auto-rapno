"""Train image -> report model from the 4 MRI modalities only (no segmentation input).

    python train.py                       # 70/15/15 split, scratch cross-attention decoder
    python train.py --decoder facebook/bart-base --epochs 40
    python train.py --split_file data/splits/nnunet_fold2.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.cli import run_training, train_parser  # noqa: E402

if __name__ == "__main__":
    args = train_parser(__doc__, default_use_seg=False).parse_args()
    run_training(args, "mri_only")
