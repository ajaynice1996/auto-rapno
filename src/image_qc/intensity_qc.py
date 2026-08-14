from __future__ import annotations

from typing import Dict, List

import nibabel as nib
import numpy as np


def check_intensity(case_row: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    for modality, field_name in [("T1", "t1_path"), ("T1CE", "t1ce_path"), ("T2", "t2_path"), ("FLAIR", "flair_path")]:
        p = str(case_row.get(field_name, ""))
        if not p:
            continue
        try:
            img = nib.load(p)
            data = img.get_fdata(dtype=np.float32)
            if data.size == 0:
                warnings.append(f"empty_scan:{modality}")
                continue
            if np.std(data) < 1e-6:
                warnings.append(f"near_constant_intensity:{modality}")
        except Exception:
            warnings.append(f"intensity_read_failed:{modality}")
    return warnings
