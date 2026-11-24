from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PROJECT_ROOT = nnUNet/brain_tumor_reporting
# REPOSITORY_ROOT = nnUNet
REPOSITORY_ROOT = PROJECT_ROOT.parent

DATASET_ROOT = REPOSITORY_ROOT / "nnUNet_raw" / "Dataset202_BraTS"

IMAGES_DIR = DATASET_ROOT / "imagesTr_all"
LABELS_DIR = DATASET_ROOT / "labelsTr_all"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
SUBJECT_OUTPUT_DIR = OUTPUT_DIR / "subjects"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
SUBJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = {
    "0000": "t1n",
    "0001": "t1c",
    "0002": "t2f",
    "0003": "t2w",
}

SEQUENCE_DESCRIPTIONS = {
    "t1n": "T1-weighted native",
    "t1c": "T1-weighted contrast-enhanced",
    "t2f": "T2-FLAIR",
    "t2w": "T2-weighted",
}

LABELS = {
    0: "background",
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}

COMPONENTS = {
    "ET": [1],
    "NET": [2],
    "CC": [3],
    "ED": [4],
    "TC": [1, 2, 3],
    "WT": [1, 2, 3, 4],
}

EXPECTED_NUM_TRAINING = 260
FILE_ENDING = ".nii.gz"