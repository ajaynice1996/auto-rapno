from __future__ import annotations

from typing import Dict, List, Tuple

from src.common.io_utils import load_header_info, spacing_anisotropy_ratio


def check_geometry(case_row: Dict[str, object]) -> Tuple[List[str], List[str], Dict[str, object]]:
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, object] = {}

    for modality_key, field_name in [
        ("T1", "t1_path"),
        ("T1CE", "t1ce_path"),
        ("T2", "t2_path"),
        ("FLAIR", "flair_path"),
    ]:
        p = str(case_row.get(field_name, ""))
        if not p:
            errors.append(f"missing_file:{modality_key}")
            continue
        hdr = load_header_info(p)
        if not hdr.exists:
            errors.append(f"missing_file:{modality_key}")
            continue

        details[f"{modality_key.lower()}_shape"] = "x".join(str(x) for x in hdr.shape or ())
        details[f"{modality_key.lower()}_spacing"] = "" if hdr.voxel_spacing is None else ",".join(f"{v:.3f}" for v in hdr.voxel_spacing)

        if hdr.voxel_spacing is not None:
            aniso = spacing_anisotropy_ratio(hdr.voxel_spacing)
            details[f"{modality_key.lower()}_anisotropy"] = round(aniso, 4)
            if aniso > 4.0:
                warnings.append(f"high_anisotropy:{modality_key}")
            if any(v < 0.3 or v > 5.0 for v in hdr.voxel_spacing):
                warnings.append(f"unusual_spacing:{modality_key}")

        if hdr.shape is not None and any(s < 32 for s in hdr.shape):
            warnings.append(f"small_matrix:{modality_key}")

    return errors, warnings, details
