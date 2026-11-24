#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def resolve_predict_python() -> str:
    preferred = Path("/home/ashar/.pyenv/versions/3.10.13/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, tp, fp, fn

    dice = safe_div(2 * tp, (2 * tp + fp + fn))
    iou = safe_div(tp, (tp + fp + fn))
    return float(dice), float(iou), tp, fp, fn


def find_label_dir(dataset_raw: Path) -> Path:
    candidates = [dataset_raw / "labelsTr", dataset_raw / "labelsTr_all"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find labels folder in {dataset_raw}")


def find_images_dir(dataset_raw: Path, sample_case: str) -> Path:
    candidates = [dataset_raw / "imagesTr", dataset_raw / "imagesTr_all"]
    for c in candidates:
        if (c / f"{sample_case}_0000.nii.gz").exists():
            return c
    raise FileNotFoundError(f"Could not find images folder for case {sample_case}")


def prepare_prediction_input(images_dir: Path, cases: list[str], temp_input_dir: Path) -> None:
    if temp_input_dir.exists():
        shutil.rmtree(temp_input_dir)
    temp_input_dir.mkdir(parents=True, exist_ok=True)

    for case_id in cases:
        for ch in range(4):
            src = images_dir / f"{case_id}_{ch:04d}.nii.gz"
            if not src.exists():
                raise FileNotFoundError(f"Missing image channel file: {src}")
            dst = temp_input_dir / src.name
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)


def parse_model_folder_name(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected model folder name format: {name}")
    trainer = parts[0]
    plans = parts[1]
    config = "__".join(parts[2:])
    return trainer, plans, config


def run_prediction_for_model(
    predict_python: str,
    model_dir_name: str,
    dataset_id: str,
    fold: int,
    input_dir: Path,
    output_dir: Path,
    env: dict,
    checkpoint_name: str,
    repo_root: Path,
    device: str,
    disable_tta: bool,
) -> None:
    trainer, plans, config = parse_model_folder_name(model_dir_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        predict_python,
        "-c",
        "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
        "-d", dataset_id,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-f", str(fold),
        "-tr", trainer,
        "-p", plans,
        "-c", config,
        "-chk", checkpoint_name,
        "-device", device,
        "-npp", "1",
        "-nps", "1",
    ]

    if disable_tta:
        cmd.append("--disable_tta")

    print(f"Running prediction for {model_dir_name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))


def load_nifti_int(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.int16)


def evaluate_predictions(
    model_name: str,
    prediction_dir: Path,
    labels_dir: Path,
    cases: list[str],
    et_label: int,
) -> list[dict]:
    rows: list[dict] = []
    for case_id in cases:
        pred_path = prediction_dir / f"{case_id}.nii.gz"
        gt_path = labels_dir / f"{case_id}.nii.gz"

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT label file: {gt_path}")

        pred = load_nifti_int(pred_path)
        gt = load_nifti_int(gt_path)

        pred_et = pred == et_label
        gt_et = gt == et_label

        dice, iou, tp, fp, fn = dice_iou(pred_et, gt_et)

        rows.append(
            {
                "model": model_name,
                "case_id": case_id,
                "et_label": et_label,
                "dice": dice,
                "iou": iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred_voxels": int(pred_et.sum()),
                "gt_voxels": int(gt_et.sum()),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out: list[dict] = []
    for model, items in sorted(by_model.items()):
        dice_vals = np.array([x["dice"] for x in items], dtype=np.float64)
        iou_vals = np.array([x["iou"] for x in items], dtype=np.float64)
        out.append(
            {
                "model": model,
                "num_cases": len(items),
                "mean_dice": float(dice_vals.mean()),
                "std_dice": float(dice_vals.std()),
                "mean_iou": float(iou_vals.mean()),
                "std_iou": float(iou_vals.std()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with saved nnUNet models and evaluate ET performance.")
    parser.add_argument("--dataset-id", default="202")
    parser.add_argument("--dataset-name", default="Dataset202_BraTS")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--et-label", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--model-contains", default=None, help="Only process models whose folder name contains this substring")
    parser.add_argument("--model-exact", default=None, help="Only process the exact model folder name")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction and only evaluate existing prediction files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    dataset_raw = repo_root / "nnUNet_raw" / args.dataset_name
    dataset_preprocessed = repo_root / "nnUNet_preprocessed" / args.dataset_name
    dataset_results = repo_root / "nnUNet_results" / args.dataset_name

    splits_file = dataset_preprocessed / "splits_final.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"Missing split file: {splits_file}")

    splits = json.loads(splits_file.read_text())
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"Fold {args.fold} is out of range for {len(splits)} splits")

    val_cases = splits[args.fold]["val"]
    if not val_cases:
        raise RuntimeError("No validation cases found")

    labels_dir = find_label_dir(dataset_raw)
    images_dir = find_images_dir(dataset_raw, val_cases[0])

    model_dirs = sorted(
        [
            d
            for d in dataset_results.iterdir()
            if d.is_dir() and (d / f"fold_{args.fold}" / "checkpoint_best.pth").exists()
        ]
    )

    if args.model_contains:
        model_dirs = [d for d in model_dirs if args.model_contains in d.name]

    if args.model_exact:
        model_dirs = [d for d in model_dirs if d.name == args.model_exact]

    if not model_dirs:
        raise RuntimeError(f"No model folders with fold_{args.fold}/checkpoint_best.pth found in {dataset_results}")

    eval_root = dataset_results / "et_eval"
    pred_root = eval_root / "predictions"
    tmp_input = eval_root / "_tmp_input_fold"

    prepare_prediction_input(images_dir, val_cases, tmp_input)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(repo_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(repo_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(repo_root / "nnUNet_results")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)

    predict_python = resolve_predict_python()

    all_rows: list[dict] = []
    failed_models: list[dict] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_pred_dir = pred_root / model_name

        try:
            if not args.skip_predict:
                run_prediction_for_model(
                    predict_python=predict_python,
                    model_dir_name=model_name,
                    dataset_id=args.dataset_id,
                    fold=args.fold,
                    input_dir=tmp_input,
                    output_dir=model_pred_dir,
                    env=env,
                    checkpoint_name=args.checkpoint,
                    repo_root=repo_root,
                    device=args.device,
                    disable_tta=args.disable_tta,
                )

            rows = evaluate_predictions(
                model_name=model_name,
                prediction_dir=model_pred_dir,
                labels_dir=labels_dir,
                cases=val_cases,
                et_label=args.et_label,
            )
            all_rows.extend(rows)
        except Exception as e:
            failed_models.append({"model": model_name, "error": str(e)})
            print(f"Skipping model due to error: {model_name} -> {e}")

    per_case_csv = eval_root / f"et_metrics_per_case_fold{args.fold}.csv"
    summary_csv = eval_root / f"et_metrics_summary_fold{args.fold}.csv"

    write_csv(
        per_case_csv,
        all_rows,
        ["model", "case_id", "et_label", "dice", "iou", "tp", "fp", "fn", "pred_voxels", "gt_voxels"],
    )

    summary_rows = summarize(all_rows)
    write_csv(
        summary_csv,
        summary_rows,
        ["model", "num_cases", "mean_dice", "std_dice", "mean_iou", "std_iou"],
    )

    if failed_models:
        failed_csv = eval_root / f"et_eval_failures_fold{args.fold}.csv"
        write_csv(failed_csv, failed_models, ["model", "error"])
        print(f"Saved failures to:         {failed_csv}")

    print(f"Saved per-case metrics to: {per_case_csv}")
    print(f"Saved summary metrics to:  {summary_csv}")


if __name__ == "__main__":
    main()
