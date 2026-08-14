from __future__ import annotations

from typing import Dict, List

from src.common.io_utils import load_header_info


def check_orientation(case_row: Dict[str, object]) -> List[str]:
    orientations: List[str] = []
    warnings: List[str] = []
    for field_name in ["t1_path", "t1ce_path", "t2_path", "flair_path"]:
        p = str(case_row.get(field_name, ""))
        if not p:
            continue
        hdr = load_header_info(p)
        if hdr.exists and hdr.orientation:
            orientations.append(hdr.orientation)

    if orientations and len(set(orientations)) > 1:
        warnings.append("orientation_mismatch_across_modalities")
    return warnings
