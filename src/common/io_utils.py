from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

CANONICAL_MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]
OPTIONAL_MODALITIES = ["DWI", "ADC", "SEG"]

MODALITY_ALIASES = {
    "T1": {"T1", "T1N", "T1W", "T1_NATIVE"},
    "T1CE": {"T1CE", "T1C", "T1PC", "T1GD", "T1_POST"},
    "T2": {"T2", "T2W"},
    "FLAIR": {"FLAIR", "T2F", "T2FLAIR"},
    "DWI": {"DWI"},
    "ADC": {"ADC"},
    "SEG": {"SEG", "LABEL", "MASK"},
}


@dataclass
class ImageHeaderInfo:
    path: str
    exists: bool
    shape: Optional[Tuple[int, int, int]]
    voxel_spacing: Optional[Tuple[float, float, float]]
    orientation: Optional[str]
    affine: Optional[np.ndarray]


def canonical_modality(raw_modality: str) -> str:
    upper = raw_modality.upper()
    for canonical, aliases in MODALITY_ALIASES.items():
        if upper in aliases:
            return canonical
    return upper


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    ensure_dir(path.parent)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_header_info(image_path: str) -> ImageHeaderInfo:
    p = Path(image_path)
    if not p.exists():
        return ImageHeaderInfo(str(p), False, None, None, None, None)

    img = nib.load(str(p))
    zooms = img.header.get_zooms()[:3]
    shape = tuple(int(x) for x in img.shape[:3])
    orientation = "".join(nib.orientations.aff2axcodes(img.affine))
    return ImageHeaderInfo(
        path=str(p),
        exists=True,
        shape=shape,
        voxel_spacing=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
        orientation=orientation,
        affine=img.affine,
    )


def spacing_anisotropy_ratio(spacing: Tuple[float, float, float]) -> float:
    vals = np.array(spacing, dtype=float)
    if np.any(vals <= 0):
        return float("inf")
    return float(vals.max() / vals.min())


def l2_affine_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
