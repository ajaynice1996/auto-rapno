"""Train image -> report model with the tumor segmentation included.

The ground-truth segmentation (labelsTr; ET/NET/CC/ED) is added as 4 one-hot input channels next to the
4 MRI modalities, and volume/extent/position measurements derived from it are fed to the decoder as an
extra visual token. Otherwise identical to train.py so the two runs can be compared directly.

    python Train_segmentation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.cli import run_training, train_parser  # noqa: E402

if __name__ == "__main__":
    args = train_parser(__doc__, default_use_seg=True).parse_args()
    run_training(args, "mri_plus_seg")
