#!/usr/bin/env python3
"""
Next steps: Gemini-generated pediatric brain tumor MRI report.

Priority: Complete and validate BraTSPED_00002 before batch processing.
"""

from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent
SUBJECT = "BraTSPED_00002"

REPORT_SCRIPT = (
    ROOT / "brain_tumor_reporting/reporting/"
    "10_gemini_report_generator.py"
)

OUTPUT = ROOT / "gemini_generated_reports" / SUBJECT

INPUTS = {
    "T1n": ROOT / f"nnUNet_raw/Dataset202_BraTS/imagesTr_all/{SUBJECT}_0000.nii.gz",
    "T1c": ROOT / f"nnUNet_raw/Dataset202_BraTS/imagesTr_all/{SUBJECT}_0001.nii.gz",
    "T2f": ROOT / f"nnUNet_raw/Dataset202_BraTS/imagesTr_all/{SUBJECT}_0002.nii.gz",
    "T2w": ROOT / f"nnUNet_raw/Dataset202_BraTS/imagesTr_all/{SUBJECT}_0003.nii.gz",
    "Tumor mask": ROOT / f"nnUNet_raw/Dataset202_BraTS/labelsTr_all/{SUBJECT}.nii.gz",
    "SynthSeg": ROOT / f"brain_tumor_reporting/outputs/synthseg/{SUBJECT}_synthseg.nii.gz",
    "Tumor-anatomical overlap": (
        ROOT / f"brain_tumor_reporting/outputs/tumor_anatomical_overlap/"
        f"{SUBJECT}/{SUBJECT}_tumor_anatomical_overlap.csv"
    ),
    "Top affected regions": (
        ROOT / f"brain_tumor_reporting/outputs/tumor_anatomical_overlap/"
        f"{SUBJECT}/{SUBJECT}_top_affected_regions.csv"
    ),
}

EXPECTED_OUTPUTS = {
    "Report Markdown": OUTPUT / "report.md",
    "Report DOCX": OUTPUT / "report.docx",
    "Gemini response": OUTPUT / "gemini_response.json",
    "Input summary": OUTPUT / "input_summary.json",
    "Input manifest": OUTPUT / "input_manifest.json",
    "Validation": OUTPUT / "validation.json",
}


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_file(label, path):
    status = "FOUND" if path.exists() else "MISSING"
    print(f"[{status:7}] {label}: {path}")
    return path.exists()


def main():
    print(f"NEXT STEPS FOR GEMINI MRI REPORT")
    print(f"Date: {date.today()}")
    print(f"Subject: {SUBJECT}")
    print(f"Project: {ROOT}")

    section("1. Verify report-generation prerequisites")

    all_inputs = True
    for label, path in INPUTS.items():
        all_inputs &= check_file(label, path)

    script_exists = check_file("Gemini report generator", REPORT_SCRIPT)

    env_file = ROOT / "gemini_generated_reports" / ".env"
    env_exists = check_file("Gemini .env configuration", env_file)

    section("2. Run Gemini report generation")

    if all_inputs and script_exists and env_exists:
        print("READY:")
        print(
            f"python {REPORT_SCRIPT.relative_to(ROOT)} "
            f"--subject {SUBJECT}"
        )
    else:
        print("Resolve missing prerequisites before running Gemini.")

    section("3. Expected report outputs")

    for label, path in EXPECTED_OUTPUTS.items():
        check_file(label, path)

    section("4. Tomorrow's prioritized work")

    tasks = [
        "Confirm the Gemini API connection and configured model.",
        "Run the report generator for BraTSPED_00002 only.",
        "Inspect report.md for completeness and factual accuracy.",
        "Inspect report.docx for formatting, headings and tables.",
        "Compare all tumor volumes against the original quantitative CSV.",
        "Verify ET, NET, CC, ED, TC and WT definitions and values.",
        "Verify anatomical overlap values against the overlap CSV.",
        "Ensure spatial overlap is not described as histological invasion.",
        "Ensure tumor grade, histology and clinical history are not invented.",
        "Check that MRI images and visual overlays are correctly represented.",
        "Review the report's limitations and quality-control section.",
        "Improve DOCX tables, page layout, headers and figure captions.",
        "Record Gemini model name, prompt version and generation date.",
        "Save reviewer corrections and finalize the BraTSPED_00002 template.",
        "Only after approval, test the workflow on 2-3 additional subjects.",
    ]

    for i, task in enumerate(tasks, start=1):
        print(f"[ ] {i:02d}. {task}")

    section("5. Do not start yet")

    print("[ ] Do not generate reports for all 260 subjects.")
    print("[ ] Do not treat Gemini output as a clinical diagnosis.")
    print("[ ] Do not accept unverified measurements or anatomical claims.")

    section("6. Final target")

    print("One complete, reproducible and manually reviewed report:")
    print(f"  {OUTPUT / 'report.docx'}")
    print("\nNext milestone: validate the report-generation workflow on 3 subjects.")


if __name__ == "__main__":
    main()
