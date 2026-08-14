from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def _write_nifti(path: Path, data: np.ndarray) -> None:
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def create_demo_case(root: Path) -> Path:
    case_dir = root / "data" / "raw" / "demo_case_0001"
    case_dir.mkdir(parents=True, exist_ok=True)

    shape = (64, 64, 48)
    zz, yy, xx = np.indices(shape)
    center = np.array([32, 30, 24])
    radius = 10
    tumor = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2) < radius**2

    base = np.random.default_rng(42).normal(loc=200.0, scale=25.0, size=shape).astype(np.float32)
    t1 = base.copy()
    t1ce = base.copy()
    t2 = base.copy()
    flair = base.copy()

    t1[tumor] += 30
    t1ce[tumor] += 70
    t2[tumor] += 40
    flair[tumor] += 80

    seg = np.zeros(shape, dtype=np.uint8)
    seg[tumor] = 4

    _write_nifti(case_dir / "demo_case_0001-t1n.nii.gz", t1)
    _write_nifti(case_dir / "demo_case_0001-t1c.nii.gz", t1ce)
    _write_nifti(case_dir / "demo_case_0001-t2w.nii.gz", t2)
    _write_nifti(case_dir / "demo_case_0001-t2f.nii.gz", flair)
    _write_nifti(case_dir / "demo_case_0001-seg.nii.gz", seg)

    (case_dir / "report.txt").write_text(
        "Demo radiology report placeholder. Left hemispheric mass with edema and enhancement.",
        encoding="utf-8",
    )

    return case_dir
