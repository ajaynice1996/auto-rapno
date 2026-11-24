"""Build Dataset202_BraTS/imagesTr_ET: 4 channels focused on enhancing tumor (ET).

  _0000 = t1c_z_minus_t1n_z   (z-scored T1ce minus z-scored T1n)
  _0001 = t1c                 (original, copied untouched)
  _0002 = t1c_rel_enh         ((T1c - T1n) / T1n)
  _0003 = t1c_local_z         (local 15^3 z-score of T1ce)

labelsTr_ET is a copy of labelsTr (same labels).
"""
import shutil
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

from generate_t1ce_variants import RAW, load, make_variants

DST = RAW / "imagesTr_ET"
LBL = RAW / "labelsTr_ET"
MAP = {"0000": "t1c_z_minus_t1n_z", "0002": "t1c_rel_enh", "0003": "t1c_local_z"}


def work(case):
    if all((DST / f"{case}_{k}.nii.gz").exists() for k in ("0000", "0001", "0002", "0003")):
        return case
    img = load(case, "0001")
    t1n, t1c, t2f, t2w = (load(case, s).get_fdata(dtype=np.float32) for s in ("0000", "0001", "0002", "0003"))
    v = make_variants(t1n, t1c, t2f, t2w)
    for k, name in MAP.items():
        out = DST / f"{case}_{k}.nii.gz"
        nib.save(nib.Nifti1Image(v[name], img.affine, img.header), str(out) + ".tmp.nii.gz")
        Path(str(out) + ".tmp.nii.gz").rename(out)
    shutil.copyfile(RAW / "imagesTr" / f"{case}_0001.nii.gz", DST / f"{case}_0001.nii.gz")
    return case


if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    LBL.mkdir(exist_ok=True)
    for f in (RAW / "labelsTr").glob("*.nii.gz"):
        shutil.copyfile(f, LBL / f.name)
    cases = sorted(p.name[: -len("_0001.nii.gz")] for p in (RAW / "imagesTr").glob("*_0001.nii.gz"))
    with Pool(12) as p:
        for i, c in enumerate(p.imap_unordered(work, cases), 1):
            print(i, len(cases), c, flush=True)
