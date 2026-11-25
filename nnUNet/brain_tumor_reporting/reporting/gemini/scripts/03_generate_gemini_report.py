#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_ROOT = SCRIPT_DIR.parent
NNUNET_ROOT = SCRIPT_DIR.parents[3]

MAX_ATTEMPTS = 4

IMAGE_PROMPT = """
You are assisting with research-oriented brain MRI image interpretation.

Analyze the supplied images for the specified subject.

IMPORTANT EVIDENCE DISTINCTIONS:
1. Original unsegmented MRI images show the underlying image appearance.
2. Segmentation overlays show regions predicted by a computational model.
3. Anatomical visualizations show automated anatomical segmentation and
   spatial overlap, not histological confirmation.

For each image, consider:
- MRI sequence and viewing plane, when identifiable from metadata.
- Visible signal characteristics and anatomical location.
- Whether the finding is visible in the original MRI.
- Whether the corresponding segmentation overlay appears spatially
  concordant with the underlying MRI.
- Image quality, artifacts, limitations, and uncertainty.

Provide these sections:

A. IMAGE INVENTORY AND COVERAGE
B. OBSERVATIONS FROM ORIGINAL, UNSEGMENTED MRI
C. OBSERVATIONS FROM TUMOR SEGMENTATION OVERLAYS
D. MULTI-SEQUENCE AND MULTIPLANAR CONCORDANCE
E. ANATOMICAL VISUALIZATION OBSERVATIONS
F. IMAGE QUALITY AND LIMITATIONS
G. FINDINGS REQUIRING HUMAN REVIEW

STRICT RULES:
- Do not infer histology, molecular subtype, or tumor grade.
- Do not infer enhancement solely from an ET segmentation label.
- Do not describe anatomical overlap as tumor invasion or infiltration.
- Do not claim that a predicted segmentation is correct merely because
  it is displayed as an overlay.
- Do not invent clinical history, symptoms, age, sex, or pathology.
- Do not infer findings that cannot be confidently assessed from the
  supplied images.
- Distinguish observations from original MRI from model-derived outputs.
- If an image is ambiguous or low quality, explicitly state this.
- Do not fabricate measurements.
- This is a research interpretation, not a clinical diagnosis.

Use cautious language. If the evidence is insufficient, say so.
"""


DEFAULT_REPORT_PROMPT = """
You are assisting with research-oriented brain MRI report generation.

Generate a structured report using:
1. The supplied image interpretation.
2. The supplied quantitative case summary.
3. The source and quality-control metadata.

The quantitative measurements in the structured case summary are the
authoritative numerical source. Preserve them exactly. Do not invent
measurements or silently alter values.

Use the following report structure:

# Brain MRI Research Report

## 1. Examination and Available Sequences
Describe only the sequences and image information actually supplied.
Do not invent acquisition parameters.

## 2. Image-Based Findings
Summarize observations from original unsegmented MRI images.
Clearly distinguish visible signal characteristics from uncertain findings.

## 3. Model-Based Tumor Segmentation
Report the available segmentation components and their measured volumes.
Distinguish ET, NET, CC, and ED from the composite TC and WT measurements.
Do not add composite volumes to their component volumes.

## 4. Anatomical Distribution and Spatial Overlap
Summarize the leading anatomical overlap measurements.
Use the exact meanings of:
- percent_tumor_in_anatomy
- percent_anatomy_occupied

These are different quantities. Do not interchange them.
Spatial overlap does not establish histological invasion or infiltration.

## 5. Image and Segmentation Quality
Summarize relevant visual QC and SynthSeg QC information.
SynthSeg QC scores are not diagnostic confidence scores.

## 6. Integrated Research Summary
Provide a concise synthesis, distinguishing:
- Direct MRI observations.
- Model-derived segmentation observations.
- Quantitative spatial measurements.
- Uncertainties and limitations.

## 7. Limitations and Human Review
List findings requiring radiologist review and any limitations of the
available images, segmentation, or anatomical alignment.

## 8. Research Disclaimer
State that this is an AI-assisted research draft and is not a clinical
diagnosis or a substitute for review by a qualified radiologist.

MANDATORY SAFEGUARDS:
- Do not infer tumor histology, molecular subtype, or grade.
- Do not infer enhancement from the ET label alone.
- Do not call anatomical overlap invasion or infiltration.
- Do not invent patient age, sex, symptoms, history, or pathology.
- Do not invent sequence parameters.
- Do not interpret absent segmentation labels as proof of absent disease.
- Do not claim segmentation accuracy without independent validation.
- Do not describe a tiny component as clinically significant without
  appropriate context.
- If the MRI images and segmentation outputs disagree, explicitly report
  the discrepancy rather than choosing one without justification.
- Do not fabricate findings to make the report appear complete.

Use professional, concise radiology-style language, but label the output
as a research draft rather than a clinical report.
"""


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def configure_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                log_dir / "gemini_report_generation.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def retry_call(function, description):
    last_error = None
    attempt_limit = max(MAX_ATTEMPTS, 5)

    for attempt in range(1, attempt_limit + 1):
        try:
            logging.info(
                "%s: attempt %d/%d",
                description,
                attempt,
                attempt_limit,
            )
            return function()

        except Exception as exc:
            last_error = exc
            logging.warning("%s failed: %s", description, exc)

            if attempt < attempt_limit:
                wait = 15 * (2 ** (attempt - 1))
                logging.info(
                    "Retrying in %d seconds.",
                    wait,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"{description} failed after {attempt_limit} attempts."
    ) from last_error


def validate_inputs(case_dir):
    prepared = case_dir / "prepared"

    summary_path = prepared / "case_summary.json"
    selection_path = prepared / "gemini_selected_images.json"

    summary = read_json(summary_path)
    selection = read_json(selection_path)

    if summary.get("subject_id") != case_dir.name:
        raise ValueError("Subject ID mismatch in case_summary.json.")

    if selection.get("subject_id") != case_dir.name:
        raise ValueError("Subject ID mismatch in image selection.")

    checks = summary.get("component_volume_consistency_checks", {})

    if not checks.get("TC_equals_ET_plus_NET_plus_CC", False):
        raise ValueError("TC volume consistency check failed.")

    if not checks.get("WT_equals_TC_plus_ED", False):
        raise ValueError("WT volume consistency check failed.")

    images = selection.get("images", [])

    if not images:
        raise ValueError("No selected images found.")

    if selection.get("selected_image_count") != len(images):
        raise ValueError("Image count does not match the selection list.")

    resolved = []

    for item in images:
        image_path = (case_dir / item["path"]).resolve()

        try:
            image_path.relative_to(case_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Image path escapes the subject directory: {image_path}"
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Selected image is missing: {image_path}"
            )

        if image_path.suffix.lower() != ".png":
            raise ValueError(
                f"Unsupported image format: {image_path}"
            )

        resolved.append((item, image_path))

    return summary, selection, resolved


def get_model_and_client():
    env_path = GEMINI_ROOT / ".env"

    if not env_path.is_file():
        raise FileNotFoundError(
            f"Gemini environment file not found: {env_path}"
        )

    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the configured .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=180000),
    )

    return client, model


def upload_images(client, images, output_dir):
    """
    Reuse previously uploaded Gemini files when possible.

    Images are matched using their relative local_path.
    Only missing images are uploaded.
    """

    manifest_path = (
        output_dir / "google_uploaded_image_manifest.json"
    )

    existing_records = {}

    if manifest_path.is_file():
        try:
            saved_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            for record in saved_manifest.get("images", []):
                local_path = record.get("local_path")
                file_uri = record.get("google_file_uri")

                if local_path and file_uri:
                    existing_records[local_path] = record

            logging.info(
                "Loaded %d existing Gemini image references.",
                len(existing_records),
            )

        except Exception as exc:
            logging.warning(
                "Could not read existing upload manifest: %s",
                exc,
            )

    uploaded_parts = []
    updated_manifest = []

    for index, (item, path) in enumerate(images, start=1):
        local_path = item["path"]

        record = existing_records.get(local_path)

        if record and record.get("google_file_uri"):
            uri = record["google_file_uri"]
            mime_type = record.get("mime_type") or "image/png"

            logging.info(
                "Reusing uploaded image %d/%d: %s",
                index,
                len(images),
                path.name,
            )

        else:
            logging.info(
                "Uploading missing image %d/%d: %s",
                index,
                len(images),
                path.name,
            )

            uploaded_file = retry_call(
                lambda p=path: client.files.upload(file=str(p)),
                f"Upload {path.name}",
            )

            uri = getattr(uploaded_file, "uri", None)
            mime_type = (
                getattr(uploaded_file, "mime_type", None)
                or "image/png"
            )

            if not uri:
                raise RuntimeError(
                    f"No file URI returned for {path.name}"
                )

        image_part = types.Part.from_uri(
            file_uri=uri,
            mime_type=mime_type,
        )

        uploaded_parts.append((item, image_part))

        updated_manifest.append({
            "local_path": local_path,
            "filename": path.name,
            "google_file_uri": uri,
            "mime_type": mime_type,
            "category": item.get("category"),
            "component": item.get("component"),
            "sequence": item.get("sequence"),
            "view": item.get("view"),
        })

    write_json(
        manifest_path,
        {
            "image_count": len(updated_manifest),
            "images": updated_manifest,
        },
    )

    reused_count = sum(
        1 for item, _ in uploaded_parts
        if item["path"] in existing_records
    )

    logging.info(
        "Image preparation complete: %d reused, %d total.",
        reused_count,
        len(uploaded_parts),
    )

    return uploaded_parts


def generate_content(client, model, contents, description):
    response = retry_call(
        lambda: client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        ),
        description,
    )

    response_text = response.text

    if not response_text or not response_text.strip():
        raise RuntimeError(
            f"{description}: Gemini returned an empty response."
        )

    return response_text


def generate_image_interpretation(
    client,
    model,
    uploaded_parts,
    subject,
    output_dir,
):
    parts = [
        types.Part.from_text(
            text=(
                IMAGE_PROMPT
                + f"\n\nSUBJECT ID: {subject}\n"
                + "The following images are supplied with metadata."
            )
        )
    ]

    for item, image_part in uploaded_parts:
        metadata = {
            "filename": item.get("filename"),
            "path": item.get("path"),
            "category": item.get("category"),
            "component": item.get("component"),
            "sequence": item.get("sequence"),
            "view": item.get("view"),
            "reason": item.get("reason_for_selection"),
        }

        parts.append(
            types.Part.from_text(
                text="\nIMAGE METADATA:\n"
                + json.dumps(metadata, ensure_ascii=False)
            )
        )
        parts.append(image_part)

    response_text = generate_content(
        client,
        model,
        [
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        "Gemini image interpretation",
    )

    output_path = output_dir / "01_image_interpretation.md"
    output_path.write_text(response_text, encoding="utf-8")

    logging.info("Saved image interpretation: %s", output_path)

    return response_text


def generate_final_report(
    client,
    model,
    summary,
    image_interpretation,
    output_dir,
):
    prompt_path = GEMINI_ROOT / "prompts" / "01_brain_tumor_mri_report.md"

    report_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.is_file()
        else DEFAULT_REPORT_PROMPT
    )

    contents = f"""
{report_prompt}

# SUBJECT
{summary.get("subject_id")}

# STRUCTURED CASE SUMMARY

The following JSON contains the quantitative measurements and metadata.
Treat the numerical values as authoritative. Preserve the values and
their definitions. Do not invent measurements.

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

# IMAGE INTERPRETATION

The following text contains the observations from the first-stage
image interpretation. Treat it as model-generated interpretation,
not independent ground truth.

{image_interpretation}

# FINAL INSTRUCTION

Generate the complete structured research report now.
Use the source measurements exactly.
Distinguish image findings from model-derived findings.
Explicitly identify uncertainty and limitations.
"""

    response_text = generate_content(
        client,
        model,
        contents,
        "Gemini final report generation",
    )

    report_path = output_dir / "02_brain_mri_research_report.md"
    report_path.write_text(response_text, encoding="utf-8")

    logging.info("Saved final report: %s", report_path)

    return response_text


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Gemini brain MRI report generation."
    )

    parser.add_argument(
        "--subject",
        default="BraTSPED_00002",
        help="Subject identifier.",
    )

    parser.add_argument(
        "--stage",
        choices=["all", "images", "report"],
        default="all",
        help="Run both stages, image interpretation only, or report only.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs without uploading or calling Gemini.",
    )

    args = parser.parse_args()

    case_dir = GEMINI_ROOT / args.subject
    output_dir = case_dir / "generated"
    log_dir = GEMINI_ROOT / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)

    logging.info("=" * 65)
    logging.info("GEMINI BRAIN MRI REPORT GENERATION")
    logging.info("Subject: %s", args.subject)
    logging.info("Stage: %s", args.stage)
    logging.info("=" * 65)

    summary, selection, images = validate_inputs(case_dir)

    logging.info("Validated subject summary.")
    logging.info("Validated %d selected images.", len(images))

    if args.dry_run:
        logging.info("DRY RUN: no API calls will be made.")
        logging.info("Image count: %d", len(images))
        logging.info("Original unsegmented images: %d",
                     selection.get("original_unsegmented_image_count", 0))
        logging.info("Segmentation overlays: %d",
                     selection.get("segmentation_overlay_count", 0))
        logging.info("Output directory: %s", output_dir)
        print("\nDRY RUN PASSED. Inputs are ready.")
        return

    client, model = get_model_and_client()
    logging.info("Configured Gemini model: %s", model)

    image_interpretation = None

    if args.stage in ("all", "images"):
        uploaded_parts = upload_images(
            client,
            images,
            output_dir,
        )

        image_interpretation = generate_image_interpretation(
            client,
            model,
            uploaded_parts,
            args.subject,
            output_dir,
        )

        logging.info("Stage 1 completed.")

    if args.stage in ("all", "report"):
        if image_interpretation is None:
            interpretation_path = (
                output_dir / "01_image_interpretation.md"
            )

            if not interpretation_path.is_file():
                raise FileNotFoundError(
                    "Image interpretation is missing. "
                    "Run --stage images first, or use --stage all."
                )

            image_interpretation = interpretation_path.read_text(
                encoding="utf-8"
            )

        generate_final_report(
            client,
            model,
            summary,
            image_interpretation,
            output_dir,
        )

        logging.info("Stage 2 completed.")

    logging.info("=" * 65)
    logging.info("GENERATION FINISHED")
    logging.info("Output directory: %s", output_dir)
    logging.info("=" * 65)

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Report generation failed.")
        raise
