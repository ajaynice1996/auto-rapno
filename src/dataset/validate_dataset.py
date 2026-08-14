from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Support both:
# 1) python -m src.dataset.validate_dataset
# 2) python src/dataset/validate_dataset.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.common.io_utils import read_csv, write_csv


def validate_manifest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        case_id = row.get("case_id", "")
        missing_files: List[str] = []
        for key in ["t1_path", "t1ce_path", "t2_path", "flair_path", "seg_path", "report_txt_path"]:
            p = row.get(key, "")
            if p and not Path(p).exists():
                missing_files.append(key)

        out.append(
            {
                "case_id": case_id,
                "missing_file_fields": "|".join(missing_files),
                "dataset_qc_status": "PASS" if not missing_files else "WARN",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest paths")
    parser.add_argument("--manifest-csv", default="data/metadata/master_manifest.csv")
    parser.add_argument("--output-csv", default="data/metadata/data_qc_report.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    rows = read_csv((root / args.manifest_csv).resolve())
    qc_rows = validate_manifest_rows(rows)
    out_csv = (root / args.output_csv).resolve()
    write_csv(out_csv, qc_rows)
    print(f"Data QC report written: {out_csv}")


if __name__ == "__main__":
    main()
