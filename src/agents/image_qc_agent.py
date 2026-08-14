from __future__ import annotations

from typing import Dict, List

from src.image_qc.geometry_qc import check_geometry
from src.image_qc.intensity_qc import check_intensity
from src.image_qc.orientation_qc import check_orientation


def run_image_qc(intake_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in intake_rows:
        errors, warnings, details = check_geometry(row)
        warnings.extend(check_orientation(row))
        warnings.extend(check_intensity(row))
        status = "FAIL" if errors else ("WARN" if warnings else "PASS")
        out.append({"case_id": row["case_id"], "qc_status": status, "errors": "|".join(sorted(set(errors))), "warnings": "|".join(sorted(set(warnings))), **details})
    return out
