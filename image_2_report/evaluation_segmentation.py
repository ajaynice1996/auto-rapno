"""Evaluate a model trained WITH segmentation (Train_segmentation.py).

    python evaluation_segmentation.py --run_dir runs/mri_plus_seg_scratch --split test
    python evaluation_segmentation.py --run_dir runs/... --pred_seg_dir path/to/nnunet_predictions

By default the ground-truth segmentation (labelsTr) is fed to the model. With --pred_seg_dir the
segmentation comes from nnU-Net predictions (<case_id>.nii.gz), which is the realistic deployment
setting. For an unbiased result use predictions from the nnU-Net fold in which the case was NOT trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.evaluation_core import evaluation_parser, run_evaluation  # noqa: E402

if __name__ == "__main__":
    ap = evaluation_parser(__doc__)
    ap.add_argument("--pred_seg_dir", default=None, help="folder with predicted segmentations named <case_id>.nii.gz")
    args = ap.parse_args()
    run_evaluation(args, seg_mode=True, pred_seg_dir=args.pred_seg_dir)
