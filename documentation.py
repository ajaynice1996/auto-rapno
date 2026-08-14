"""Execution documentation for Phase 1 pipeline.

Run this file to print full execution instructions:
    python documentation.py
"""

from __future__ import annotations

from pathlib import Path


DOC = """
AUTO-RAPNO ORGANIZED PIPELINE DOCUMENTATION
===========================================

Goal
----
Build and run the complete baseline pipeline for pediatric brain tumor agentic reporting:
1) Image intake and modality inventory
2) Image QC (voxel, orientation, size)
3) Registration triage
4) Missing-contrast triage and synthetic queue flagging
5) Segmentation (provided mask or fallback stub)
6) Segmentation QC
7) Quantification
8) Draft report generation
9) Report validation
10) Phase summary outputs

Code Entry Points (organized layout)
------------------------------------
Main orchestrator:
- src/agents/orchestrator.py

Dataset prep:
- src/dataset/create_manifest.py
- src/dataset/validate_dataset.py
- src/dataset/split_dataset.py

Core modules:
- src/intake/*
- src/image_qc/*
- src/registration/*
- src/modality/*
- src/segmentation/*
- src/qc/*
- src/quantification/*
- src/reports/*
- src/evaluation/*


PHASE 1 INPUTS
--------------
Input A: data_index.json (real dataset mode)
- Format expected per case:
  - case_id key -> payload with "modalities"
  - modality keys like T1N/T1C/T2W/T2F/SEG are supported

Input B: demo data mode (no external data needed)
- Auto-generated NIfTI files under data/raw/demo_case_0001/

Input C: optional existing manifest
- data/metadata/master_manifest.csv


HOW TO RUN
----------
0) Create and use local virtual environment (recommended on macOS)
   /opt/homebrew/bin/python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install nibabel numpy scipy

   Optional full install:
   .venv/bin/python -m pip install -r requirements.txt

1) Demo run (recommended first)
   .venv/bin/python -m src.agents.orchestrator --use-demo

2) Real dataset run from data_index.json
   .venv/bin/python -m src.agents.orchestrator --index-json data_index.json

3) Build manifest only (real index mode)
   .venv/bin/python -m src.dataset.create_manifest \
       --index-json data_index.json \
       --output-csv data/metadata/master_manifest.csv

4) Validate manifest paths
   .venv/bin/python -m src.dataset.validate_dataset \
       --manifest-csv data/metadata/master_manifest.csv \
       --output-csv data/metadata/data_qc_report.csv

5) Write train/val/test ID files from split column
   .venv/bin/python -m src.dataset.split_dataset \
       --manifest-csv data/metadata/master_manifest.csv


STEP-BY-STEP OUTPUTS
--------------------
Step 1: Image intake + modality inventory
- outputs/image_intake/modality_inventory.csv

Step 2: Image QC
- outputs/image_qc/image_qc_scores.csv

Step 3: Registration triage
- outputs/registration/registration_triage.csv

Step 4: Missing contrast and synthesis queue
- outputs/modality_qc/missing_contrast_flags.csv
- outputs/modality_qc/synthesis_queue.csv

Step 5: Segmentation outputs
- data/derived/segmentations_pred/segmentation_outputs.csv
- data/derived/segmentations_pred/<case_id>_seg_pred.nii.gz (if prediction fallback used)

Step 6: Segmentation QC
- outputs/segmentation_qc/seg_qc_scores.csv

Step 7: Quantification
- outputs/quantification/tumor_metrics.csv

Step 8: Draft report generation
- outputs/report_generation/<case_id>_report_draft.txt
- outputs/report_generation/report_draft_index.csv

Step 9: Report validation
- outputs/report_validation/report_validation_results.csv

Step 10: Global case summary
- outputs/evaluation/phase1_case_summary.csv
- outputs/evaluation/phase1_summary.json

Phase 2 placeholder output:
- outputs/evaluation/phase2_agentic_vs_model_comparison.csv


ONE EXAMPLE TO PROCEED
----------------------
Use exactly this for first execution:

.venv/bin/python -m src.agents.orchestrator --use-demo

Expected outcome:
- One demo case created
- Full pipeline completed
- All output files listed above generated


NOTES
-----
- If your real data paths in data_index.json are not available on this machine,
  rows will be flagged as missing and QC/triage outputs will indicate failures.
- Segmentation behavior:
  - If SEG exists and file is present: it is used directly.
  - If SEG is missing: fallback stub segmentation is generated from available MRI volume.
- Synthetic contrast generation is not executed in Phase 1 baseline; only triaged via
  synthesis queue flags for later integration.

- Previous folders src/phase1 and src/step1 are legacy and not required for new runs.
"""


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")
    print(DOC)


if __name__ == "__main__":
    main()
