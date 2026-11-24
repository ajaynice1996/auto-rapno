"""Command-line arguments shared by train.py and Train_segmentation.py."""
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def train_parser(description, default_use_seg):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--split_file", default=str(ROOT / "data/splits/ratio_70_15_15.json"),
                    help="ratio_70_15_15.json (default) or data/splits/nnunet_fold2.json")
    ap.add_argument("--run_dir", default=None, help="output folder (default runs/<name>_<timestamp>)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--decoder", default="scratch",
                    help="'scratch' (transformer trained from scratch) or a HF seq2seq name, e.g. facebook/bart-base, GanjinZero/biobart-v2-base")
    ap.add_argument("--size", type=int, default=128, help="cached volume size (must match prepare_data.py)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lm_lr", type=float, default=5e-5, help="lr of the pretrained LM (HF decoders only)")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--aux_weight", type=float, default=0.5, help="weight of the attribute-prediction auxiliary loss (0 disables)")
    ap.add_argument("--modality_dropout", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--max_len", type=int, default=448)
    ap.add_argument("--val_gen_every", type=int, default=5, help="generate on the val split every N epochs for checkpoint selection")
    ap.add_argument("--patience", type=int, default=0, help="stop after N generation checks without improvement (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.set_defaults(use_seg=default_use_seg)
    return ap


def run_training(args, default_name):
    from .engine import train
    cfg = {k: v for k, v in vars(args).items() if k not in ("run_dir", "name")}
    name = args.name or default_name
    run_dir = args.run_dir or str(ROOT / "runs" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    print("run dir:", run_dir)
    train(cfg, run_dir)
