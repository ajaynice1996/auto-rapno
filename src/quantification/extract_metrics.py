from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


def run_quantification(seg_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in seg_rows:
        case_id = str(row["case_id"])
        p = str(row.get("segmentation_path", ""))
        if not p or not Path(p).exists():
            out.append({"case_id": case_id, "total_tumor_ml": "", "enhancing_ml": "", "edema_ml": "", "necrotic_ml": "", "max_diameter_vox": "", "status": "FAIL:no_seg"})
            continue

        img = nib.load(p)
        seg = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

        mask = seg > 0
        coords = np.argwhere(mask)
        if coords.size == 0:
            out.append({"case_id": case_id, "total_tumor_ml": 0.0, "enhancing_ml": 0.0, "edema_ml": 0.0, "necrotic_ml": 0.0, "max_diameter_vox": 0.0, "status": "WARN:empty_seg"})
            continue

        bbox_diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        total_ml = float(mask.sum() * voxel_ml)
        out.append(
            {
                "case_id": case_id,
                "total_tumor_ml": round(total_ml, 4),
                "enhancing_ml": round(float((seg == 4).sum() * voxel_ml), 4),
                "edema_ml": round(float((seg == 2).sum() * voxel_ml), 4),
                "necrotic_ml": round(float((seg == 1).sum() * voxel_ml), 4),
                "max_diameter_vox": round(bbox_diag, 3),
                "status": "PASS",
            }
        )
    return out
