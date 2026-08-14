from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def run_report_validation(report_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in report_rows:
        case_id = str(row["case_id"])
        p = str(row.get("report_path", ""))
        if not p or not Path(p).exists():
            out.append({"case_id": case_id, "validation_status": "FAIL", "reason": "missing_report"})
            continue

        text = Path(p).read_text(encoding="utf-8")
        ok = ("Findings:" in text) and ("Impression:" in text) and ("radiologist review" in text.lower())
        out.append({"case_id": case_id, "validation_status": "PASS" if ok else "FAIL", "reason": "" if ok else "template_mismatch"})
    return out
