from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Support both:
# 1) python -m src.agents.orchestrator
# 2) python src/agents/orchestrator.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.agents.image_intake_agent import run_image_intake
from src.agents.image_qc_agent import run_image_qc
from src.agents.modality_completeness_agent import run_modality_completeness
from src.agents.qc_agent import run_segmentation_qc
from src.agents.quantification_agent import run_quantification
from src.agents.registration_triage_agent import run_registration_triage
from src.agents.report_critic_agent import run_report_validation
from src.agents.report_draft_agent import run_report_draft
from src.agents.segmentation_agent import run_segmentation
from src.common.io_utils import read_csv, write_csv, write_json
from src.dataset.create_manifest import build_manifest
from src.dataset.demo_data import create_demo_case
from src.evaluation.compare_systems import write_phase2_placeholder
from src.evaluation.report_metrics import summarize_validation
from src.reports.parse_reports import parse_reports_from_manifest


def _merge_case_rows(*row_sets: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for rows in row_sets:
        for row in rows:
            case_id = str(row["case_id"])
            if case_id not in merged:
                merged[case_id] = {"case_id": case_id}
            merged[case_id].update(row)
    return [merged[k] for k in sorted(merged.keys())]


def run_pipeline(root: Path, manifest_csv: Path) -> None:
    manifest_rows = read_csv(manifest_csv)
    if not manifest_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_csv}")

    intake_rows = run_image_intake(manifest_rows)
    write_csv(root / "outputs/image_intake/modality_inventory.csv", intake_rows)
    modality_summary = {
        "num_cases": len(intake_rows),
        "cases_with_missing_required": sum(
            1 for r in intake_rows if str(r.get("missing_required_modality_flag", "")).lower() in {"true", "1"}
        ),
    }
    write_json(root / "outputs/image_intake/case_modality_summary.json", modality_summary)

    qc_rows = run_image_qc(intake_rows)
    write_csv(root / "outputs/image_qc/image_qc_scores.csv", qc_rows)
    failure_rows = [r for r in qc_rows if str(r.get("qc_status", "")) in {"WARN", "FAIL"}]
    write_csv(root / "outputs/image_qc/image_qc_failures.csv", failure_rows)

    reg_rows = run_registration_triage(intake_rows)
    write_csv(root / "outputs/registration/registration_triage.csv", reg_rows)

    comp_rows = run_modality_completeness(intake_rows)
    write_csv(root / "outputs/modality_qc/missing_contrast_flags.csv", comp_rows)
    synth_queue = [r for r in comp_rows if str(r.get("synthesis_recommended_flag", "")).lower() in {"true", "1"}]
    write_csv(root / "outputs/modality_qc/synthesis_queue.csv", synth_queue)

    seg_rows = run_segmentation(intake_rows, root / "data/derived/segmentations_pred")
    write_csv(root / "data/derived/segmentations_pred/segmentation_outputs.csv", seg_rows)

    seg_qc_rows = run_segmentation_qc(seg_rows)
    write_csv(root / "outputs/segmentation_qc/seg_qc_scores.csv", seg_qc_rows)

    quant_rows = run_quantification(seg_rows)
    write_csv(root / "outputs/quantification/tumor_metrics.csv", quant_rows)

    report_rows = run_report_draft(quant_rows, root / "outputs/report_generation")
    write_csv(root / "outputs/report_generation/report_draft_index.csv", report_rows)

    validation_rows = run_report_validation(report_rows)
    write_csv(root / "outputs/report_validation/report_validation_results.csv", validation_rows)

    parse_reports_from_manifest(intake_rows, root / "data/derived/report_structured_labels.csv")

    summary = summarize_validation(validation_rows)
    summary["num_missing_required_modality"] = sum(
        1 for r in comp_rows if str(r.get("missing_required_modality_flag", "")).lower() in {"true", "1"}
    )
    summary["num_registration_required"] = sum(
        1 for r in reg_rows if str(r.get("registration_required_flag", "")).lower() in {"true", "1"}
    )
    write_json(root / "outputs/evaluation/phase1_summary.json", summary)

    merged = _merge_case_rows(intake_rows, qc_rows, reg_rows, comp_rows, seg_rows, seg_qc_rows, quant_rows, validation_rows)
    write_csv(root / "outputs/evaluation/phase1_case_summary.csv", merged)

    write_phase2_placeholder(root / "outputs/evaluation/phase2_agentic_vs_model_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run organized agentic pipeline (Phase 1 baseline)")
    parser.add_argument("--index-json", default="data_index.json")
    parser.add_argument("--manifest-csv", default="data/metadata/master_manifest.csv")
    parser.add_argument("--use-demo", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest_csv = (root / args.manifest_csv).resolve()
    index_json = (root / args.index_json).resolve()

    if args.use_demo:
        create_demo_case(root)
        build_manifest(index_json, manifest_csv, use_demo=True)
    elif not manifest_csv.exists():
        build_manifest(index_json, manifest_csv, use_demo=False)

    run_pipeline(root, manifest_csv)
    print("Pipeline completed.")
    print(f"Summary: {root / 'outputs/evaluation/phase1_summary.json'}")


if __name__ == "__main__":
    main()