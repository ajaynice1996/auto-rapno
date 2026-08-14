from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Support both:
# 1) python -m src.dataset.create_manifest
# 2) python src/dataset/create_manifest.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.common.io_utils import canonical_modality, read_json, write_csv


def _extract_case_rows(index_payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case_id, raw_case in index_payload.items():
        if not isinstance(raw_case, dict):
            continue
        modalities = raw_case.get("modalities", {})
        if not isinstance(modalities, dict):
            modalities = {}

        canon: Dict[str, str] = {}
        for raw_mod, raw_path in modalities.items():
            if isinstance(raw_path, str):
                canon[canonical_modality(raw_mod)] = raw_path

        rows.append(
            {
                "case_id": str(case_id),
                "subject_id": str(raw_case.get("subject_id", case_id)),
                "study_uid": str(case_id),
                "split": "unassigned",
                "t1_path": canon.get("T1", ""),
                "t1ce_path": canon.get("T1CE", ""),
                "t2_path": canon.get("T2", ""),
                "flair_path": canon.get("FLAIR", ""),
                "dwi_path": canon.get("DWI", ""),
                "adc_path": canon.get("ADC", ""),
                "seg_path": canon.get("SEG", ""),
                "report_txt_path": "",
            }
        )
    return rows


def _build_demo_manifest_rows(root: Path) -> List[Dict[str, object]]:
    case_dir = root / "data" / "raw" / "demo_case_0001"
    return [
        {
            "case_id": "demo_case_0001",
            "subject_id": "demo_case_0001",
            "study_uid": "demo_case_0001",
            "split": "test",
            "t1_path": str(case_dir / "demo_case_0001-t1n.nii.gz"),
            "t1ce_path": str(case_dir / "demo_case_0001-t1c.nii.gz"),
            "t2_path": str(case_dir / "demo_case_0001-t2w.nii.gz"),
            "flair_path": str(case_dir / "demo_case_0001-t2f.nii.gz"),
            "dwi_path": "",
            "adc_path": "",
            "seg_path": str(case_dir / "demo_case_0001-seg.nii.gz"),
            "report_txt_path": str(case_dir / "report.txt"),
        }
    ]


def build_manifest(index_path: Path, output_csv: Path, use_demo: bool = False) -> Path:
    root = output_csv.parents[2]
    if use_demo:
        rows = _build_demo_manifest_rows(root)
    else:
        payload = read_json(index_path)
        rows = _extract_case_rows(payload)
    write_csv(output_csv, rows)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Create master manifest CSV from index")
    parser.add_argument("--index-json", default="data_index.json")
    parser.add_argument("--output-csv", default="data/metadata/master_manifest.csv")
    parser.add_argument("--use-demo", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    index_path = (root / args.index_json).resolve()
    output_csv = (root / args.output_csv).resolve()
    build_manifest(index_path, output_csv, args.use_demo)
    print(f"Manifest written: {output_csv}")


if __name__ == "__main__":
    main()
