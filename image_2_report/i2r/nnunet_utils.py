"""Run the project's trained nnU-Net (Dataset202_BraTS) to obtain tumor segmentations."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .volume import MODALITIES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NNUNET_DIR = ROOT.parent / "nnUNet"


def predict_segmentations(case_modalities, out_dir, nnunet_dir=DEFAULT_NNUNET_DIR, dataset="202", config="3d_fullres",
                          trainer="nnUNetTrainer", plans="nnUNetPlans", fold="2", checkpoint="checkpoint_best.pth"):
    """case_modalities: {case_id: {modality: path}} (all four modalities required by nnU-Net).

    Writes <out_dir>/<case_id>.nii.gz label maps. Note: nnU-Net was only trained for fold 2 here; cases that
    belong to the fold-2 training set were seen by the segmentation model (optimistic), fold-2 validation
    cases are out-of-fold.
    """
    nnunet_dir = Path(nnunet_dir)
    exe = nnunet_dir / ".venv/bin/nnUNetv2_predict"
    if not exe.exists():
        raise FileNotFoundError(f"nnUNetv2_predict not found at {exe}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, nnUNet_raw=str(nnunet_dir / "nnUNet_raw"), nnUNet_preprocessed=str(nnunet_dir / "nnUNet_preprocessed"),
               nnUNet_results=str(nnunet_dir / "nnUNet_results"))
    with tempfile.TemporaryDirectory() as tmp:
        for cid, mods in case_modalities.items():
            missing = [m for m in MODALITIES if not mods.get(m)]
            if missing:
                raise ValueError(f"{cid}: nnU-Net needs all 4 modalities, missing {missing}")
            for i, m in enumerate(MODALITIES):
                os.symlink(os.path.abspath(mods[m]), os.path.join(tmp, f"{cid}_{i:04d}.nii.gz"))
        cmd = [str(exe), "-i", tmp, "-o", str(out_dir), "-d", dataset, "-c", config, "-tr", trainer, "-p", plans,
               "-f", str(fold), "-chk", checkpoint]
        subprocess.run(cmd, check=True, env=env)
    return {cid: out_dir / f"{cid}.nii.gz" for cid in case_modalities}
