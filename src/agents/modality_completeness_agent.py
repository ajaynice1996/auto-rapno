from __future__ import annotations

from typing import Dict, List

from src.modality.build_synthesis_queue import build_synthesis_queue
from src.modality.detect_missing_contrast import detect_missing_required_modalities


def run_modality_completeness(intake_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    missing_rows = detect_missing_required_modalities(intake_rows)
    queue = build_synthesis_queue(missing_rows)
    q_map = {str(q["case_id"]): q for q in queue}

    out: List[Dict[str, object]] = []
    for row in missing_rows:
        case_id = str(row["case_id"])
        q = q_map.get(case_id, {})
        out.append(
            {
                **row,
                "synthesis_recommended_flag": bool(q.get("synthesis_recommended_flag", False)),
                "synthesis_targets": str(q.get("synthesis_targets", "")),
            }
        )
    return out
