from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def run_report_draft(quant_rows: List[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, object]] = []
    for row in quant_rows:
        case_id = str(row["case_id"])
        if str(row.get("status", "")).startswith("FAIL"):
            out.append({"case_id": case_id, "report_path": "", "report_status": "FAIL:no_quantification"})
            continue

        total = float(row.get("total_tumor_ml", 0.0) or 0.0)
        enh = float(row.get("enhancing_ml", 0.0) or 0.0)
        edema = float(row.get("edema_ml", 0.0) or 0.0)
        max_d = float(row.get("max_diameter_vox", 0.0) or 0.0)
        text = (
            "Findings:\n"
            f"- Intracranial tumor burden detected with total volume {total:.2f} mL.\n"
            f"- Enhancing component volume {enh:.2f} mL.\n"
            f"- Edema component volume {edema:.2f} mL.\n"
            f"- Maximum lesion span (voxel-space) {max_d:.2f}.\n\n"
            "Impression:\n"
            "- Auto-generated draft for radiologist review.\n"
            "- Clinical finalization required before diagnostic use.\n"
        )
        path = output_dir / f"{case_id}_report_draft.txt"
        path.write_text(text, encoding="utf-8")

        structured = {
            "case_id": case_id,
            "findings": {
                "total_tumor_ml": total,
                "enhancing_ml": enh,
                "edema_ml": edema,
                "max_diameter_vox": max_d,
            },
            "impression": "Auto-generated draft for radiologist review.",
            "guardrail": "Clinical finalization required before diagnostic use.",
        }
        structured_path = output_dir / f"{case_id}_report_structured.json"
        structured_path.write_text(json.dumps(structured, indent=2), encoding="utf-8")

        out.append(
            {
                "case_id": case_id,
                "report_path": str(path),
                "report_structured_path": str(structured_path),
                "report_status": "PASS",
            }
        )
    return out
