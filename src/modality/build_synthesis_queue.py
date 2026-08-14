from __future__ import annotations

from typing import Dict, List


def build_synthesis_queue(missing_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    queue: List[Dict[str, object]] = []
    for row in missing_rows:
        if str(row.get("missing_required_modality_flag", "")).lower() in {"true", "1"}:
            queue.append(
                {
                    "case_id": row["case_id"],
                    "synthesis_recommended_flag": True,
                    "synthesis_targets": row.get("missing_modality_list", ""),
                }
            )
    return queue
