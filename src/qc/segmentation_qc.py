from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nibabel as nib
from scipy import ndimage


def run_segmentation_qc(seg_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in seg_rows:
        case_id = str(row["case_id"])
        p = str(row.get("segmentation_path", ""))
        if not p or not Path(p).exists():
            out.append({"case_id": case_id, "seg_qc_status": "FAIL", "reason": "missing_segmentation", "voxel_count": 0, "connected_components": 0})
            continue

        seg = nib.load(p).get_fdata()
        mask = seg > 0
        voxel_count = int(mask.sum())
        _, comp = ndimage.label(mask)

        status = "PASS"
        reason: List[str] = []
        if voxel_count < 32:
            status = "FAIL"
            reason.append("too_small")
        if comp > 20 and status == "PASS":
            status = "WARN"
            reason.append("many_components")

        out.append({"case_id": case_id, "seg_qc_status": status, "reason": "|".join(reason), "voxel_count": voxel_count, "connected_components": int(comp)})
    return out
