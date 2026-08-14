from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.common.io_utils import CANONICAL_MODALITIES, write_csv


def run_image_intake(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in manifest_rows:
        modalities = {
            "T1": row.get("t1_path", ""),
            "T1CE": row.get("t1ce_path", ""),
            "T2": row.get("t2_path", ""),
            "FLAIR": row.get("flair_path", ""),
            "DWI": row.get("dwi_path", ""),
            "ADC": row.get("adc_path", ""),
            "SEG": row.get("seg_path", ""),
        }

        missing = [m for m in CANONICAL_MODALITIES if not modalities.get(m, "")]
        detected = sorted([m for m, p in modalities.items() if p])

        results.append(
            {
                "case_id": row.get("case_id", ""),
                "subject_id": row.get("subject_id", row.get("case_id", "")),
                "modalities_detected": "|".join(detected),
                "missing_required_modalities": "|".join(missing),
                "missing_required_modality_flag": bool(missing),
                **{k.lower() + "_path": v for k, v in modalities.items()},
                "report_txt_path": row.get("report_txt_path", ""),
            }
        )
    return results


def main() -> None:
    from src.common.io_utils import read_csv

    parser = argparse.ArgumentParser(description="Build modality inventory from manifest")
    parser.add_argument("--manifest-csv", default="data/metadata/master_manifest.csv")
    parser.add_argument("--output-csv", default="outputs/image_intake/modality_inventory.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    rows = read_csv((root / args.manifest_csv).resolve())
    out = run_image_intake(rows)
    out_csv = (root / args.output_csv).resolve()
    write_csv(out_csv, out)
    print(f"Modality inventory written: {out_csv}")


if __name__ == "__main__":
    main()
