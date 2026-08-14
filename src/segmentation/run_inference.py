from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


def run_segmentation(intake_rows: List[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for row in intake_rows:
        case_id = str(row["case_id"])
        existing = str(row.get("seg_path", ""))
        if existing and Path(existing).exists():
            rows.append({"case_id": case_id, "segmentation_path": existing, "segmentation_source": "provided", "segmentation_status": "PASS"})
            continue

        source = str(row.get("flair_path", "") or row.get("t2_path", "") or row.get("t1ce_path", "") or row.get("t1_path", ""))
        if not source or not Path(source).exists():
            rows.append({"case_id": case_id, "segmentation_path": "", "segmentation_source": "none", "segmentation_status": "FAIL:no_input_scan"})
            continue

        img = nib.load(source)
        data = img.get_fdata(dtype=np.float32)
        if data.size == 0:
            rows.append({"case_id": case_id, "segmentation_path": "", "segmentation_source": "none", "segmentation_status": "FAIL:empty_scan"})
            continue

        thr = np.percentile(data, 95.0)
        seg = (data > thr).astype(np.uint8)
        pred = output_dir / f"{case_id}_seg_pred.nii.gz"
        nib.save(nib.Nifti1Image(seg, img.affine, img.header), str(pred))
        rows.append({"case_id": case_id, "segmentation_path": str(pred), "segmentation_source": "stub_threshold", "segmentation_status": "PASS"})
    return rows
