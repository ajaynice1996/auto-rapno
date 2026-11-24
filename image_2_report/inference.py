"""Generate a structured report (JSON) for a new case with a trained model.

Inputs: any subset of the four MRI modalities (missing ones are zero-filled and flagged to the model) and,
for models trained with Train_segmentation.py, a tumor segmentation (nnU-Net label map: 1=ET 2=NET 3=CC 4=ED).

Examples
    python inference.py --run_dir runs/mri_only_scratch --case_dir ../nnUNet/nnUNet_raw/Dataset202_BraTS/imagesTr --case_id BraTSPED_00003
    python inference.py --run_dir runs/mri_only_scratch --t1n a_t1n.nii.gz --t1c a_t1c.nii.gz --t2f a_t2f.nii.gz --t2w a_t2w.nii.gz
    python inference.py --run_dir runs/mri_plus_seg_scratch --case_dir ... --case_id BraTSPED_00003 --seg labelsTr/BraTSPED_00003.nii.gz
    python inference.py --run_dir runs/mri_plus_seg_scratch --case_dir ... --case_id BraTSPED_00003 --run_nnunet   # segment first

The output JSON keeps the raw generated text, the schema-validated report (or null if it could not be
parsed), keyword-extracted attributes, image-only attribute predictions and (if a segmentation is used)
tumor measurements. Output is a research prototype and must be reviewed by a radiologist.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.attributes import extract_attributes  # noqa: E402
from i2r.data import collate, scale_tumor_feats  # noqa: E402
from i2r.engine import load_model, to_device  # noqa: E402
from i2r.nnunet_utils import DEFAULT_NNUNET_DIR, predict_segmentations  # noqa: E402
from i2r.output_parser import parse_generated  # noqa: E402
from i2r.volume import MODALITIES, TUMOR_FEATURE_NAMES, preprocess_case  # noqa: E402


def find_modalities(args):
    mods = {m: getattr(args, m) for m in MODALITIES if getattr(args, m)}
    if args.case_dir and args.case_id:
        d = Path(args.case_dir)
        for i, m in enumerate(MODALITIES):
            for cand in (d / f"{args.case_id}_{i:04d}.nii.gz", d / f"{args.case_id}-{m}.nii.gz", d / f"{args.case_id}_{m}.nii.gz"):
                if cand.exists():
                    mods.setdefault(m, str(cand))
    if not mods:
        sys.exit("no MRI modality found: pass --t1n/--t1c/--t2f/--t2w or --case_dir with --case_id")
    return mods


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--case_dir")
    ap.add_argument("--case_id")
    for m in MODALITIES:
        ap.add_argument(f"--{m}", help=f"path to {m} NIfTI")
    ap.add_argument("--seg", help="segmentation NIfTI (required for models trained with Train_segmentation.py unless --run_nnunet)")
    ap.add_argument("--run_nnunet", action="store_true", help="predict the segmentation with the trained nnU-Net first")
    ap.add_argument("--nnunet_dir", default=str(DEFAULT_NNUNET_DIR))
    ap.add_argument("--nnunet_fold", default="2")
    ap.add_argument("--output", default=None, help="output JSON path (default: print to stdout)")
    args = ap.parse_args()

    model, tok, cfg, device = load_model(args.run_dir, args.ckpt)
    mods = find_modalities(args)
    patient_id = args.case_id or "unknown"
    seg_path, seg_source = args.seg, "provided" if args.seg else None
    with tempfile.TemporaryDirectory() as tmp:
        if cfg["use_seg"] and not seg_path:
            if not args.run_nnunet:
                sys.exit("this model uses segmentation: pass --seg or --run_nnunet")
            pred = predict_segmentations({patient_id: mods}, tmp, args.nnunet_dir, fold=args.nnunet_fold)
            seg_path, seg_source = str(pred[patient_id]), f"nnunet_fold{args.nnunet_fold}"
        arr = preprocess_case(mods, seg_path if cfg["use_seg"] else None, cfg["size"])

    item = {"case_id": patient_id, "image": torch.from_numpy(arr["image"]), "seg": torch.from_numpy(arr["seg"]),
            "modality_mask": torch.from_numpy(arr["modality_mask"]), "tumor_feats": scale_tumor_feats(arr["tumor_feats"])}
    batch = to_device(collate([item], tok.pad_id), device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        ids, aux = model.generate(batch, max_len=cfg["max_len"])
    parsed = parse_generated(tok.decode(ids[0]))
    sections = parsed["sections"]
    result = {
        "patient_id": patient_id,
        "model": {"run_dir": str(args.run_dir), "checkpoint": args.ckpt, "uses_segmentation": cfg["use_seg"], "segmentation_source": seg_source},
        "available_modalities": [m for m, v in zip(MODALITIES, arr["modality_mask"]) if v],
        "report": sections,
        "report_status": parsed["status"],
        "report_errors": parsed["errors"],
        "raw_generated_text": parsed["raw_text"],
        "attributes_from_report_text": extract_attributes(sections or {"findings": parsed["raw_text"]}),
        "attributes_predicted_from_image": {k: v[0] for k, v in aux.items()},
        "disclaimer": "Automatically generated research output; not a clinical report. Requires radiologist review.",
    }
    if cfg["use_seg"] and arr["has_seg"]:
        result["tumor_measurements"] = {n: round(float(v), 3) for n, v in zip(TUMOR_FEATURE_NAMES, arr["tumor_feats"])}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output} (report_status={parsed['status']})")
    else:
        print(text)


if __name__ == "__main__":
    main()
