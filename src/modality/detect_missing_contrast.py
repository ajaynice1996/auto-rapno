from __future__ import annotations

from typing import Dict, List

from src.common.io_utils import CANONICAL_MODALITIES


def detect_missing_required_modalities(intake_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in intake_rows:
        case_id = str(row["case_id"])
        missing: List[str] = []
        for m in CANONICAL_MODALITIES:
            p = str(row.get(f"{m.lower()}_path", ""))
            if not p:
                missing.append(m)
        out.append(
            {
                "case_id": case_id,
                "missing_required_modality_flag": bool(missing),
                "missing_modality_list": "|".join(sorted(missing)),
            }
        )
    return out
