

#!/usr/bin/env python3

"""
Subject-specific brain segmentation and parcellation using SynthSeg.

Supports:
    --subject BraTSPED_00003  Process one subject
    --limit 5                 Process first five subjects
    --all                     Process entire dataset
    --force                   Reprocess existing outputs

Input:
    T1 MRI (_0000.nii.gz)

Output:
    Anatomical segmentation and parcellation NIfTI
    Regional volume CSV
    SynthSeg QC CSV
    Individual execution log
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "nnUNet_raw/Dataset202_BraTS/imagesTr_all"
)

SYNTHSEG_DIR = PROJECT_ROOT / "external/SynthSeg"

SYNTHSEG_SCRIPT = (
    SYNTHSEG_DIR
    / "scripts/commands/SynthSeg_predict.py"
)

SYNTHSEG_PYTHON = (
    SYNTHSEG_DIR
    / ".venv_synthseg/bin/python"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "brain_tumor_reporting/outputs/synthseg"
)

LOG_DIR = OUTPUT_DIR / "logs"


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run subject-specific SynthSeg parcellation."
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--subject",
        type=str,
        help="Subject ID, e.g. BraTSPED_00003"
    )

    selection.add_argument(
        "--limit",
        type=int,
        help="Process the first N subjects."
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Process all available subjects."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess subjects with existing outputs."
    )

    parser.add_argument(
        "--no-parc",
        action="store_true",
        help="Disable cortical parcellation."
    )

    parser.add_argument(
        "--no-robust",
        action="store_true",
        help="Disable robust SynthSeg model."
    )

    return parser.parse_args()


# ---------------------------------------------------------
# DISCOVER SUBJECTS
# ---------------------------------------------------------

def discover_subjects():

    image_files = sorted(
        INPUT_DIR.glob("*_0000.nii.gz")
    )

    subject_ids = [
        file.name.replace("_0000.nii.gz", "")
        for file in image_files
    ]

    return subject_ids


# ---------------------------------------------------------
# RUN SYNTHSEG FOR ONE SUBJECT
# ---------------------------------------------------------

def run_subject(subject_id, args):

    input_path = (
        INPUT_DIR / f"{subject_id}_0000.nii.gz"
    )

    output_path = (
        OUTPUT_DIR / f"{subject_id}_synthseg.nii.gz"
    )

    volume_path = (
        OUTPUT_DIR / f"{subject_id}_volumes.csv"
    )

    qc_path = (
        OUTPUT_DIR / f"{subject_id}_qc.csv"
    )

    log_path = (
        LOG_DIR / f"{subject_id}.log"
    )

    required_outputs = [
        output_path,
        volume_path,
        qc_path
    ]

    if not input_path.exists():
        print(f"[ERROR] Missing T1 image: {input_path}")
        return False

    # Skip only when all expected files exist.
    if all(path.exists() for path in required_outputs):
        if not args.force:
            print(f"[SKIP] {subject_id}: outputs already exist.")
            return True

    command = [
        str(SYNTHSEG_PYTHON),
        str(SYNTHSEG_SCRIPT),
        "--i", str(input_path),
        "--o", str(output_path),
        "--vol", str(volume_path),
        "--qc", str(qc_path),
    ]

    if not args.no_parc:
        command.append("--parc")

    if not args.no_robust:
        command.append("--robust")

    print("\n" + "=" * 65)
    print(f"PROCESSING: {subject_id}")
    print("=" * 65)

    print("Input:", input_path)
    print("Output:", output_path)
    print("Parcellation:", not args.no_parc)
    print("Robust model:", not args.no_robust)

    environment = os.environ.copy()

    # Use CPU for compatibility with legacy TensorFlow 2.2.
    environment["CUDA_VISIBLE_DEVICES"] = "-1"

    start_time = datetime.now()

    try:

        with open(log_path, "w") as log_file:

            result = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False
            )

        elapsed = (
            datetime.now() - start_time
        ).total_seconds()

        if result.returncode == 0 and all(
            path.exists() for path in required_outputs
        ):

            print(f"[SUCCESS] {subject_id}")
            print(f"Elapsed time: {elapsed / 60:.2f} minutes")

            return True

        print(f"[FAILED] {subject_id}")
        print("Exit code:", result.returncode)
        print("Check log:", log_path)

        return False

    except Exception as error:

        print(f"[ERROR] {subject_id}: {error}")
        return False


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    args = parse_arguments()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    if not SYNTHSEG_SCRIPT.exists():
        raise FileNotFoundError(SYNTHSEG_SCRIPT)

    if not SYNTHSEG_PYTHON.exists():
        raise FileNotFoundError(SYNTHSEG_PYTHON)

    available_subjects = discover_subjects()

    print("\nSYNTHSEG SUBJECT-SPECIFIC PIPELINE")
    print("Project:", PROJECT_ROOT)
    print("Available subjects:", len(available_subjects))

    # Select subjects.
    if args.subject:

        if args.subject not in available_subjects:
            raise ValueError(
                f"Subject not found: {args.subject}"
            )

        selected_subjects = [args.subject]

    elif args.limit is not None:

        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")

        selected_subjects = available_subjects[:args.limit]

    elif args.all:

        selected_subjects = available_subjects

    else:

        # Safe default: one subject only.
        selected_subjects = ["BraTSPED_00003"]

        if selected_subjects[0] not in available_subjects:
            raise ValueError(
                "Default subject BraTSPED_00003 not found."
            )

    print("Selected subjects:", len(selected_subjects))
    print("Force reprocessing:", args.force)

    successful = []
    failed = []
    skipped_or_completed = []

    for subject_id in selected_subjects:

        success = run_subject(subject_id, args)

        if success:
            successful.append(subject_id)
        else:
            failed.append(subject_id)

    # Save run summary.
    summary_path = OUTPUT_DIR / "batch_run_summary.txt"

    with open(summary_path, "w") as f:

        f.write("SynthSeg pipeline summary\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Selected: {len(selected_subjects)}\n")
        f.write(f"Successful or already complete: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")

        f.write("Successful:\n")
        for subject in successful:
            f.write(f"{subject}\n")

        f.write("\nFailed:\n")
        for subject in failed:
            f.write(f"{subject}\n")

    print("\n" + "=" * 65)
    print("PIPELINE SUMMARY")
    print("=" * 65)

    print("Successful or already complete:", len(successful))
    print("Failed:", len(failed))
    print("Summary:", summary_path)

    if failed:
        print("\nFailed subjects:", ", ".join(failed))


if __name__ == "__main__":
    main()
