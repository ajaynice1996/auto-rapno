from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.common.io_utils import write_csv


def parse_reports_from_manifest(rows: List[Dict[str, object]], output_csv: Path) -> Path:
    labels: List[Dict[str, object]] = []
    for row in rows:
        case_id = str(row["case_id"])
        report_path = str(row.get("report_txt_path", ""))
        text = Path(report_path).read_text(encoding="utf-8") if report_path and Path(report_path).exists() else ""
        labels.append(
            {
                "case_id": case_id,
                "report_exists": bool(text),
                "label_mass_effect": "mass effect" in text.lower(),
                "label_edema": "edema" in text.lower(),
                "label_enhancement": ("enhancement" in text.lower()) or ("enhancing" in text.lower()),
            }
        )
    write_csv(output_csv, labels)
    return output_csv
