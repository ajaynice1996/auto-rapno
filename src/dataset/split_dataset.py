from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Support both:
# 1) python -m src.dataset.split_dataset
# 2) python src/dataset/split_dataset.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.common.io_utils import read_csv


def _write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test id files from split column")
    parser.add_argument("--manifest-csv", default="data/metadata/master_manifest.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    rows = read_csv((root / args.manifest_csv).resolve())

    train_ids = [r["case_id"] for r in rows if r.get("split", "") == "train"]
    val_ids = [r["case_id"] for r in rows if r.get("split", "") == "val"]
    test_ids = [r["case_id"] for r in rows if r.get("split", "") == "test"]

    _write_ids(root / "data/metadata/train_ids.txt", train_ids)
    _write_ids(root / "data/metadata/val_ids.txt", val_ids)
    _write_ids(root / "data/metadata/test_ids.txt", test_ids)
    print("Split id files written in data/metadata")


if __name__ == "__main__":
    main()
