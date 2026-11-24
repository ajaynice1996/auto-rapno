import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "gemini_generated_reports" / "src")
)

from prepare_mri import prepare_mri_visualization


CACHE_DIR = PROJECT_ROOT / "data" / "cache_128"

BASE_DIR = PROJECT_ROOT / "gemini_generated_reports"

OUTPUT_DIR = BASE_DIR / "outputs"
VIS_DIR = BASE_DIR / "visualizations"

ENV_FILE = BASE_DIR / ".env"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def load_patient(patient_id):

    npz_path = CACHE_DIR / f"{patient_id}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Patient file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)

    return {
        "image": data["image"].astype(np.float32),
        "seg": data["seg"],
        "modality_mask": data["modality_mask"],
        "tumor_feats": data["tumor_feats"],
        "has_seg": int(data["has_seg"])
    }


def extract_patient_information(patient_id, patient):

    image = patient["image"]
    seg = patient["seg"]

    label_values, label_counts = np.unique(
        seg,
        return_counts=True
    )

    label_summary = {
        str(int(label)): int(count)
        for label, count in zip(label_values, label_counts)
    }

    channel_statistics = []

    for channel in range(image.shape[0]):

        volume = image[channel]
        finite = volume[np.isfinite(volume)]

        channel_statistics.append({
            "channel_index": channel,
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "standard_deviation": float(np.std(finite))
        })

    feature_summary = {
        f"feature_{i:02d}": float(value)
        for i, value in enumerate(patient["tumor_feats"])
    }

    return {
        "patient_id": patient_id,
        "image_shape": list(image.shape),
        "segmentation_shape": list(seg.shape),
        "modality_mask": patient["modality_mask"].tolist(),
        "has_segmentation": patient["has_seg"],
        "segmentation_label_counts": label_summary,
        "channel_statistics": channel_statistics,
        "unnamed_tumor_features": feature_summary
    }


def generate_report(patient_id, patient, visualization_path):

    if not API_KEY or API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API key is missing from .env")

    client = genai.Client(api_key=API_KEY)

    patient_info = extract_patient_information(
        patient_id,
        patient
    )

    prompt = f"""
You are assisting with a research prototype for pediatric brain
MRI report generation.

Analyze the supplied MRI montage and the accompanying metadata.

PATIENT METADATA:
{json.dumps(patient_info, indent=2)}

IMPORTANT DATA LIMITATIONS:

1. The four MRI channel names and their order are not verified.
2. Segmentation label meanings are not verified.
3. The 19 tumor features are not clinically defined.
4. Only one selected 2D slice is shown, not the full 3D volume.
5. Image orientation, voxel spacing, and anatomical registration
   have not been independently verified.
6. Do not assume that the segmentation is ground truth or prediction.

INSTRUCTIONS:

- Describe only visible imaging characteristics.
- Distinguish image observations from metadata.
- Do not invent tumor location, diagnosis, grade, enhancement,
  edema, necrosis, mass effect, or progression.
- Do not assign clinical meaning to unnamed features or labels.
- Clearly identify limitations.
- This is a research output, not a clinical diagnosis.
- Require expert review.

Return a structured report with:
patient_id
report_type
image_observations
segmentation_observations
quantitative_data_summary
limitations
requires_expert_review
"""

    image_bytes = Path(visualization_path).read_bytes()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=3000
        )
    )

    report = json.loads(response.text)

    report["patient_id"] = patient_id
    report["generated_at"] = datetime.now().isoformat()

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient",
        default="BraTSPED_00001"
    )

    args = parser.parse_args()

    patient_id = args.patient

    print(f"\nLoading patient: {patient_id}")

    patient = load_patient(patient_id)

    visualization_path = (
        VIS_DIR / f"{patient_id}_mri_overlay.png"
    )

    print("Preparing MRI visualization...")

    visualization_info = prepare_mri_visualization(
        CACHE_DIR / f"{patient_id}.npz",
        visualization_path
    )

    print("Visualization saved:", visualization_path)
    print("Visualization details:", visualization_info)

    print("\nSending MRI montage and metadata to Gemini...")

    report = generate_report(
        patient_id,
        patient,
        visualization_path
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        OUTPUT_DIR / f"{patient_id}_report_{timestamp}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\nReport generated successfully!")
    print("Saved to:", output_path)

    print("\nGenerated report:")
    print(json.dumps(report, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
