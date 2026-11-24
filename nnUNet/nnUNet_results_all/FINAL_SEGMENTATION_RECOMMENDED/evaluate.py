#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


# Final recommended class-wise strategy from no-GT analysis.
FINAL_CLASS_CONFIG = {
	"ET": {
		"label_value": 1,
		"models": ["ResEncL", "PrimusV2S", "ET500_ResEncL"],
		"rule": "adaptive_nogt_variation",
		"variation_threshold": 0.18,
	},
	"NET": {
		"label_value": 2,
		"models": ["ResEncL", "MedNeXt_k3_pp", "MedNeXt_k5_pp"],
		"rule": "union",
		"variation_threshold": None,
	},
	"CC": {
		"label_value": 3,
		"models": ["ResEncL", "MedNeXt_k3_raw", "MedNeXt_k3_pp"],
		"rule": "intersection",
		"variation_threshold": None,
	},
	"ED": {
		"label_value": 4,
		"models": ["PrimusV3S", "MedNeXt_k5_raw", "MedNeXt_k5_pp"],
		"rule": "union",
		"variation_threshold": None,
	},
}

# Tie-break priority for overlapping class masks during final label fusion.
# Higher validation Dice classes are preferred first.
TIEBREAK_CLASS_ORDER = ["NET", "ED", "ET", "CC"]


def build_model_paths(repo_root: Path) -> Dict[str, Path]:
	base = repo_root / "nnUNet_results_all" / "Dataset202_BraTS"
	return {
		"nnUNet": base / "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation",
		"ResEncL": base / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation",
		"PrimusV2S": base / "nnUNet_PrimusV2S_Trainer__nnUNetPlans__3d_fullres/fold_2/validation",
		"PrimusV3S": base / "nnUNet_PrimusV3S_Trainer__nnUNetPlans__3d_fullres/fold_2/validation",
		"MedNeXt_k3_raw": base / "nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/validation_raw",
		"MedNeXt_k3_pp": base / "nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/validation_raw_postprocessed",
		"MedNeXt_k5_raw": base / "nnUNetTrainerV2_MedNeXt_S_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/validation_raw",
		"MedNeXt_k5_pp": base / "nnUNetTrainerV2_MedNeXt_S_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/validation_raw_postprocessed",
		"ET500_ResEncL": repo_root / "nnUNet_results/Dataset202_BraTS/nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation",
		"ETvar_ResEncL_A": repo_root / "nnUNet_results_ET_variant/Dataset202_BraTS/et_eval/predictions/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
		"ETvar_ResEncL_B": repo_root / "nnUNet_results_ET_variant/Dataset202_BraTS/et_eval/predictions/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres1",
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build final fused segmentation predictions with recommended class-wise ensemble rules and optional evaluation."
	)
	parser.add_argument(
		"--labels-dir",
		type=Path,
		default=Path("nnUNet_raw/Dataset202_BraTS/labelsTr_all"),
		help="Ground-truth labels directory for evaluation. If absent, metrics are skipped unless --evaluate is set with valid path.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED"),
		help="Output directory for predictions and reports.",
	)
	parser.add_argument(
		"--cases",
		type=str,
		default="all",
		help="Comma-separated case IDs (without spaces), or 'all'.",
	)
	parser.add_argument(
		"--evaluate",
		action="store_true",
		help="Compute ET/NET/CC/ED/TC/WT metrics if labels are available.",
	)
	return parser.parse_args()


def metric(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, int, int]:
	p = int(pred_mask.sum())
	g = int(gt_mask.sum())
	if p == 0 and g == 0:
		return 1.0, 1.0, 1, 0
	if p == 0 or g == 0:
		return 0.0, 0.0, 0, 1
	inter = int((pred_mask & gt_mask).sum())
	dice = (2.0 * inter) / (p + g)
	iou = inter / (p + g - inter)
	return float(dice), float(iou), 0, 0


def mask_for_region(arr: np.ndarray, label: str) -> np.ndarray:
	if label == "ET":
		return arr == 1
	if label == "NET":
		return arr == 2
	if label == "CC":
		return arr == 3
	if label == "ED":
		return arr == 4
	if label == "TC":
		return (arr == 1) | (arr == 2) | (arr == 3)
	if label == "WT":
		return arr > 0
	raise ValueError(f"Unknown label: {label}")


def get_common_cases(model_paths: Dict[str, Path], labels_dir: Path | None) -> List[str]:
	available_models = {
		k: v
		for k, v in model_paths.items()
		if v.exists() and any(v.glob("*.nii.gz"))
	}
	if not available_models:
		raise RuntimeError("No model prediction folders found.")

	common = None
	for path in available_models.values():
		names = {p.name for p in path.glob("*.nii.gz")}
		common = names if common is None else (common & names)

	if labels_dir is not None and labels_dir.exists():
		label_names = {p.name for p in labels_dir.glob("*.nii.gz")}
		common = common & label_names

	cases = sorted(common) if common else []
	if not cases:
		raise RuntimeError("No common case files across selected model folders.")
	return cases


def select_cases(all_cases: List[str], requested: str) -> List[str]:
	if requested.lower() == "all":
		return all_cases
	picked = [c.strip() for c in requested.split(",") if c.strip()]
	missing = [c for c in picked if c not in all_cases]
	if missing:
		raise RuntimeError(f"Requested cases not found in common set: {missing}")
	return picked


def load_case_model_arrays(case_name: str, model_paths: Dict[str, Path], needed_models: Iterable[str]) -> Dict[str, np.ndarray]:
	arrs = {}
	for model in needed_models:
		path = model_paths[model] / case_name
		if not path.exists():
			raise RuntimeError(f"Missing prediction for {model}: {path}")
		arrs[model] = np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)
	return arrs


def build_class_mask_and_support(
	case_arrays: Dict[str, np.ndarray],
	class_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
	cfg = FINAL_CLASS_CONFIG[class_name]
	label_value = cfg["label_value"]
	model_list = cfg["models"]

	votes = np.zeros_like(next(iter(case_arrays.values())), dtype=np.uint8)
	for model in model_list:
		votes += (case_arrays[model] == label_value).astype(np.uint8)

	n_models = len(model_list)
	union_mask = votes > 0
	intersection_mask = votes == n_models
	support_ratio = votes.astype(np.float32) / float(n_models)

	chosen_rule = cfg["rule"]
	if chosen_rule == "union":
		final_mask = union_mask
		variation = float("nan")
		threshold = float("nan")
	elif chosen_rule == "intersection":
		final_mask = intersection_mask
		variation = float("nan")
		threshold = float("nan")
	elif chosen_rule == "adaptive_nogt_variation":
		u_size = int(union_mask.sum())
		i_size = int(intersection_mask.sum())
		threshold = float(cfg["variation_threshold"])
		variation = 0.0 if u_size == 0 else 1.0 - (i_size / u_size)
		use_union = variation > threshold or u_size == 0
		final_mask = union_mask if use_union else intersection_mask
		chosen_rule = "union" if use_union else "intersection"
	else:
		raise RuntimeError(f"Unsupported rule: {cfg['rule']}")

	details = {
		"configured_rule": cfg["rule"],
		"chosen_effective_rule": chosen_rule,
		"variation": variation,
		"variation_threshold": threshold,
		"union_size": int(union_mask.sum()),
		"intersection_size": int(intersection_mask.sum()),
	}
	return final_mask, support_ratio, details


def fuse_final_labels(class_masks: Dict[str, np.ndarray], class_supports: Dict[str, np.ndarray]) -> np.ndarray:
	shape = next(iter(class_masks.values())).shape
	final = np.zeros(shape, dtype=np.uint8)

	score_stack = []
	label_value_stack = []
	for cls in TIEBREAK_CLASS_ORDER:
		mask = class_masks[cls]
		support = class_supports[cls]
		score = np.where(mask, support, -1.0).astype(np.float32)
		score_stack.append(score)
		label_value_stack.append(FINAL_CLASS_CONFIG[cls]["label_value"])

	scores = np.stack(score_stack, axis=0)
	best_idx = np.argmax(scores, axis=0)
	best_score = np.max(scores, axis=0)
	has_class = best_score >= 0

	out_vals = np.array(label_value_stack, dtype=np.uint8)
	final[has_class] = out_vals[best_idx[has_class]]
	return final


def ensure_output_dirs(out_dir: Path) -> Tuple[Path, Path]:
	pred_dir = out_dir / "predictions"
	report_dir = out_dir / "reports"
	pred_dir.mkdir(parents=True, exist_ok=True)
	report_dir.mkdir(parents=True, exist_ok=True)
	return pred_dir, report_dir


def main() -> None:
	args = parse_args()

	script_path = Path(__file__).resolve()
	repo_root = script_path.parents[2]

	model_paths = build_model_paths(repo_root)

	labels_dir = args.labels_dir
	if not labels_dir.is_absolute():
		labels_dir = repo_root / labels_dir
	if not labels_dir.exists():
		fallback = repo_root / "nnUNet_raw/Dataset202_BraTS/labelsTr"
		labels_dir = fallback if fallback.exists() else labels_dir

	out_dir = args.output_dir
	if not out_dir.is_absolute():
		out_dir = repo_root / out_dir
	pred_dir, report_dir = ensure_output_dirs(out_dir)

	needed_models = sorted({m for c in FINAL_CLASS_CONFIG.values() for m in c["models"]})
	for model in needed_models:
		if model not in model_paths or not model_paths[model].exists():
			raise RuntimeError(f"Missing required model folder for {model}: {model_paths.get(model)}")

	all_cases = get_common_cases(model_paths, labels_dir if labels_dir.exists() else None)
	target_cases = select_cases(all_cases, args.cases)

	metric_rows = []
	selection_rows = []

	for i, case_name in enumerate(target_cases, start=1):
		arrays = load_case_model_arrays(case_name, model_paths, needed_models)

		class_masks = {}
		class_supports = {}
		for cls in ("ET", "NET", "CC", "ED"):
			msk, sup, details = build_class_mask_and_support(arrays, cls)
			class_masks[cls] = msk
			class_supports[cls] = sup
			selection_rows.append(
				{
					"case": case_name,
					"label": cls,
					"configured_rule": details["configured_rule"],
					"chosen_effective_rule": details["chosen_effective_rule"],
					"variation": details["variation"],
					"variation_threshold": details["variation_threshold"],
					"union_size": details["union_size"],
					"intersection_size": details["intersection_size"],
				}
			)

		final_pred = fuse_final_labels(class_masks, class_supports)

		ref_img = nib.load(str(model_paths[FINAL_CLASS_CONFIG["NET"]["models"][0]] / case_name))
		nib.save(nib.Nifti1Image(final_pred.astype(np.uint8), ref_img.affine, ref_img.header), str(pred_dir / case_name))

		if args.evaluate and labels_dir.exists():
			gt = np.asanyarray(nib.load(str(labels_dir / case_name)).dataobj).astype(np.uint8)
			for label in ("ET", "NET", "CC", "ED", "TC", "WT"):
				pm = mask_for_region(final_pred, label)
				gm = mask_for_region(gt, label)
				dice, iou, be, oe = metric(pm, gm)
				metric_rows.append(
					{
						"case": case_name,
						"label": label,
						"Dice": dice,
						"IoU": iou,
						"both_empty": be,
						"one_empty": oe,
					}
				)

		if i % 10 == 0 or i == len(target_cases):
			print(f"processed {i}/{len(target_cases)}")

	pd.DataFrame(selection_rows).to_csv(report_dir / "selection_log_per_case.csv", index=False)

	if metric_rows:
		per_case = pd.DataFrame(metric_rows)
		per_case.to_csv(report_dir / "metrics_per_case.csv", index=False)

		summary = (
			per_case.groupby("label", as_index=False)
			.agg(
				n_cases=("Dice", "count"),
				Dice=("Dice", "mean"),
				IoU=("IoU", "mean"),
				both_empty_cases=("both_empty", "sum"),
				one_empty_cases=("one_empty", "sum"),
			)
			.sort_values("label")
			.reset_index(drop=True)
		)
		summary.to_csv(report_dir / "metrics_summary.csv", index=False)

		# Publication helper text.
		lines = []
		lines.append("Final recommended segmentation ensemble (deployment-safe no-GT class-wise selection)")
		lines.append(f"cases_evaluated: {len(target_cases)}")
		lines.append(f"labels_dir: {labels_dir}")
		lines.append("empty_case_policy: both_empty=>1 ; one_empty=>0")
		lines.append("")
		lines.append("Class-wise final prediction strategy:")
		for cls in ("ET", "NET", "CC", "ED"):
			cfg = FINAL_CLASS_CONFIG[cls]
			lines.append(
				f"- {cls}: models={'+'.join(cfg['models'])} | rule={cfg['rule']} | threshold={cfg['variation_threshold']}"
			)
		lines.append("")
		lines.append("Validation summary (from fused final predictions):")
		for _, row in summary.iterrows():
			lines.append(
				f"- {row['label']}: Dice={row['Dice']:.4f}, IoU={row['IoU']:.4f}, both_empty={int(row['both_empty_cases'])}, one_empty={int(row['one_empty_cases'])}"
			)

		(report_dir / "publication_final_summary.txt").write_text("\n".join(lines))
		print(f"wrote: {report_dir / 'metrics_summary.csv'}")
		print(f"wrote: {report_dir / 'publication_final_summary.txt'}")

	print(f"done. predictions: {pred_dir}")
	print(f"done. reports: {report_dir}")


if __name__ == "__main__":
	main()
