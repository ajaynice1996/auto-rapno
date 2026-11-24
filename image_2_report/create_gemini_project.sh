#!/bin/bash

set -e

PROJECT="gemini_generated_reports"

echo "Creating Gemini report-generation project..."

mkdir -p "$PROJECT"/{scripts,src,outputs,visualizations,logs,tests}

# --------------------------------------------------
# 1. Requirements
# --------------------------------------------------

cat > "$PROJECT/requirements.txt" <<'EOF'
google-genai
python-dotenv
numpy
matplotlib
Pillow
EOF

# --------------------------------------------------
# 2. Environment configuration
# --------------------------------------------------

cat > "$PROJECT/.env" <<'EOF'
GEMINI_API_KEY=PASTE_YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
EOF

# --------------------------------------------------
# 3. Git ignore
# --------------------------------------------------

cat > "$PROJECT/.gitignore" <<'EOF'
.env
__pycache__/
*.pyc
outputs/
visualizations/
logs/
.venv/
EOF

# --------------------------------------------------
# 4. MRI visualization script
# --------------------------------------------------

cat > "$PROJECT/src/prepare_mri.py" <<'PY'
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def prepare_mri_visualization(npz_path, output_path):

    data = np.load(npz_path, allow_pickle=False)

    image = data["image"].astype(np.float32)
    seg = data["seg"]

    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(
            f"Expected image shape (4, X, Y, Z), got {image.shape}"
        )

    if seg.shape != image.shape[1:]:
        raise ValueError("Image and segmentation shapes do not match.")

    # Select the slice with the largest foreground area.
    best_axis = None
    best_index = None
    best_area = -1

    for axis in range(3):

        areas = []

        for index in range(seg.shape[axis]):

            slice_mask = np.take(seg > 0, index, axis=axis)
            areas.append(np.count_nonzero(slice_mask))

        index = int(np.argmax(areas))
        area = int(areas[index])

        if area > best_area:
            best_area = area
            best_axis = axis
            best_index = index

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for channel in range(4):

        mri_slice = np.take(
            image[channel], best_index, axis=best_axis
        )

        mask_slice = np.take(
            seg, best_index, axis=best_axis
        )

        valid = mri_slice[np.isfinite(mri_slice)]

        if valid.size == 0:
            raise ValueError("MRI slice contains no finite values.")

        vmin, vmax = np.percentile(valid, [1, 99])

        axes[channel].imshow(
            mri_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        # Generic foreground overlay; label meanings are not assumed.
        overlay = np.ma.masked_where(
            mask_slice == 0,
            mask_slice > 0
        )

        axes[channel].imshow(
            overlay,
            cmap="autumn",
            alpha=0.45
        )

        axes[channel].set_title(f"Channel {channel}")
        axes[channel].axis("off")

    fig.suptitle(
        f"MRI channels with foreground mask | Axis {best_axis}, "
        f"slice {best_index}"
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "axis": best_axis,
        "slice_index": best_index,
        "foreground_voxels": int(np.count_nonzero(seg)),
        "visualization": str(output_path)
    }
PY

# --------------------------------------------------
# 5. Gemini API connection test
# --------------------------------------------------

cat > "$PROJECT/scripts/test_gemini_connection.py" <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "gemini_generated_reports" / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY or API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    raise ValueError(
        "Please add your Gemini API key to gemini_generated_reports/.env"
    )

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model=MODEL,
    contents="Reply with exactly: Gemini API connection successful."
)

print(response.text)
PY

# --------------------------------------------------
# 6. Generate report for one patient
# --------------------------------------------------

cat > "$PROJECT/scripts/generate_one_patient.py" <<'PY'
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
PY

# --------------------------------------------------
# 7. README
# --------------------------------------------------

cat > "$PROJECT/README.md" <<'EOF'
# Gemini MRI Report Generation

Research prototype for generating structured MRI observations
using the Gemini API.

## Setup

1. Activate your Python environment.
2. Install requirements:
   pip install -r requirements.txt
3. Add your Gemini API key to .env.
4. Test the API:
   python scripts/test_gemini_connection.py
5. Generate a report:
   python scripts/generate_one_patient.py --patient BraTSPED_00001

Input:
../data/cache_128/<patient_id>.npz

Outputs:
outputs/
visualizations/

This prototype is not for clinical use without institutional
approval, validation, and expert review.
EOF

echo ""
echo "Project created successfully!"
echo "Location: $PROJECT"
echo ""
echo "Next: install requirements and configure your API key."
