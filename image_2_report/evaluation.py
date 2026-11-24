"""Evaluate a model trained WITHOUT segmentation (train.py).

    python evaluation.py --run_dir runs/mri_only_scratch --split test

Writes predictions.json (generated JSON + extracted attributes), metrics.json, summary.txt and a
radiologist_review_sheet.csv into <run_dir>/eval_<split>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.evaluation_core import evaluation_parser, run_evaluation  # noqa: E402

if __name__ == "__main__":
    run_evaluation(evaluation_parser(__doc__).parse_args(), seg_mode=False)
