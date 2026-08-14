from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.common.io_utils import l2_affine_delta, load_header_info, read_csv, write_csv


def run_registration_triage(intake_rows: List[Dict[str, object]], threshold: float = 1e-3) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in intake_rows:
        case_id = str(row["case_id"])
        ref_path = str(row.get("flair_path", "") or row.get("t2_path", "") or row.get("t1_path", ""))
        if not ref_path:
            out.append({"case_id": case_id, "registration_required_flag": True, "reason": "no_reference_modality", "max_affine_l2_delta": ""})
            continue

        ref = load_header_info(ref_path)
        if (not ref.exists) or (ref.affine is None):
            out.append({"case_id": case_id, "registration_required_flag": True, "reason": "reference_missing_or_invalid", "max_affine_l2_delta": ""})
            continue

        deltas: List[float] = []
        for field_name in ["t1_path", "t1ce_path", "t2_path", "flair_path"]:
            p = str(row.get(field_name, ""))
            if not p:
                continue
            h = load_header_info(p)
            if h.exists and h.affine is not None:
                deltas.append(l2_affine_delta(ref.affine, h.affine))

        max_delta = max(deltas) if deltas else 0.0
        out.append(
            {
                "case_id": case_id,
                "registration_required_flag": bool(max_delta > threshold),
                "reason": "affine_delta_exceeds_threshold" if max_delta > threshold else "aligned",
                "max_affine_l2_delta": f"{max_delta:.8f}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Registration triage from modality inventory CSV")
    parser.add_argument("--input-csv", default="outputs/image_intake/modality_inventory.csv")
    parser.add_argument("--output-csv", default="outputs/registration/registration_triage.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    rows = read_csv((root / args.input_csv).resolve())
    out = run_registration_triage(rows)
    out_csv = (root / args.output_csv).resolve()
    write_csv(out_csv, out)
    print(f"Registration triage written: {out_csv}")


if __name__ == "__main__":
    main()
