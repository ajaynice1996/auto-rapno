import os
import json
import pickle
import subprocess
from pathlib import Path

import torch


def stage0_integrity(task_dir: Path, plans_identifier: str):
    plans_file = task_dir / f"{plans_identifier}_plans_3D.pkl"
    if not plans_file.exists():
        return False, None, "plans file missing"

    with plans_file.open("rb") as f:
        plans_meta = pickle.load(f)

    data_identifier = plans_meta.get("data_identifier")
    if not data_identifier:
        return False, None, "data_identifier missing in plans"

    stage0_dir = task_dir / f"{data_identifier}_stage0"
    if not stage0_dir.exists():
        return False, None, f"stage0 folder missing: {stage0_dir.name}"

    npz_stems = {p.stem for p in stage0_dir.glob("*.npz")}
    pkl_stems = {p.stem for p in stage0_dir.glob("*.pkl")}
    missing_props = sorted(npz_stems - pkl_stems)

    if not npz_stems:
        return False, stage0_dir, "no .npz files found in stage0"
    if missing_props:
        return False, stage0_dir, f"missing .pkl for {len(missing_props)} cases (example: {missing_props[0]})"

    return True, stage0_dir, None


def ensure_v1_dataset_json(raw_task_dir: Path):
    dataset_json_path = raw_task_dir / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found in {raw_task_dir}")

    with dataset_json_path.open("r") as f:
        dataset = json.load(f)

    if "training" in dataset and "modality" in dataset:
        return

    images_tr = raw_task_dir / "imagesTr"
    labels_tr = raw_task_dir / "labelsTr"
    if not images_tr.exists() or not labels_tr.exists():
        raise FileNotFoundError(
            f"Expected imagesTr and labelsTr in {raw_task_dir} for v1 compatibility conversion"
        )

    channel_names = dataset.get("channel_names", {})
    sorted_channels = sorted(channel_names.items(), key=lambda kv: int(kv[0]))
    modality = {str(i): name for i, (_, name) in enumerate(sorted_channels)}

    if not modality:
        raise ValueError(
            f"channel_names missing or empty in {dataset_json_path}; cannot build v1 modality mapping"
        )

    training_cases = []
    label_files = sorted(labels_tr.glob("*.nii.gz"))
    for label_file in label_files:
        case_id = label_file.name[:-7]
        first_channel = images_tr / f"{case_id}_0000.nii.gz"
        if first_channel.exists():
            training_cases.append(
                {
                    "image": f"./imagesTr/{case_id}.nii.gz",
                    "label": f"./labelsTr/{case_id}.nii.gz",
                }
            )

    if not training_cases:
        raise ValueError(
            f"No valid training cases found in {raw_task_dir}. Ensure imagesTr/labelsTr follow nnUNet naming."
        )

    dataset["modality"] = modality
    dataset["training"] = training_cases
    dataset["numTraining"] = len(training_cases)
    dataset.setdefault("test", [])

    with dataset_json_path.open("w") as f:
        json.dump(dataset, f, indent=4)
        f.write("\n")

    print(f"Converted dataset.json to nnUNet v1-compatible format in {raw_task_dir.name}")


# ============================================================
# 1. Activate/check environment
# ============================================================

print("=" * 70)
print("MedNeXt Training")
print("=" * 70)

print(f"Python: {os.sys.executable}")
print(f"PyTorch: {torch.__version__}")


# ============================================================
# 2. GPU CHECK
# ============================================================

cuda_available = torch.cuda.is_available()

print(f"\nCUDA available: {cuda_available}")

if cuda_available:

    num_gpus = torch.cuda.device_count()

    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):

        props = torch.cuda.get_device_properties(i)

        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        free = total - reserved

        print(f"\nGPU {i}: {props.name}")
        print(f"Total memory:     {total:.2f} GB")
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved:  {reserved:.2f} GB")
        print(f"Memory free:     {free:.2f} GB")

    torch.cuda.empty_cache()

    print("\nCUDA cache cleared.")


def pick_task_name(raw_root: Path, preprocessed_root: Path, dataset_id: str, plans_identifier: str) -> str:
    prefix = f"Task{int(dataset_id):03d}_"
    pre_candidates = sorted(
        p.name
        for p in preprocessed_root.glob(f"{prefix}*")
        if p.is_dir() and (p / f"{plans_identifier}_plans_3D.pkl").exists()
    )
    complete_candidates = []
    for name in pre_candidates:
        ok, _, _ = stage0_integrity(preprocessed_root / name, plans_identifier)
        if ok:
            complete_candidates.append(name)

    if complete_candidates:
        return complete_candidates[0]
    if pre_candidates:
        return pre_candidates[0]

    raw_candidates = sorted(p.name for p in raw_root.glob(f"{prefix}*") if p.is_dir())
    if raw_candidates:
        return raw_candidates[0]

    raise FileNotFoundError(
        f"No task folder found for prefix '{prefix}' in {raw_root} or {preprocessed_root}."
    )


# ============================================================
# 3. MedNeXt configuration
# ============================================================
mednext_path = Path(__file__).resolve().parent
python_exe = os.sys.executable

# MedNeXt v1 path convention:
# nnUNet_raw_data_base/<nnUNet_raw_data|nnUNet_cropped_data>
os.environ["nnUNet_raw_data_base"] = str(mednext_path)
os.environ["nnUNet_preprocessed"] = str(mednext_path / "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = str(mednext_path / "nnUNet_results")


# ---------- Configuration ----------
dataset_id = "202"
fold = "2"
configuration = "3d_fullres"
planner = "ExperimentPlanner3D_v21_customTargetSpacing_1x1x1"
plans = "nnUNetPlansv2.1_trgSp_1x1x1"
trainer = "nnUNetTrainerV2_MedNeXt_S_kernel5"

raw_base = mednext_path / "nnUNet_raw_data"
preprocessed_base = Path(os.environ["nnUNet_preprocessed"])

task = pick_task_name(raw_base, preprocessed_base, dataset_id, plans)
raw_task_dir = raw_base / task

# ============================================================
# 4. CHECK DATA
# ============================================================

print("\nTask:", task)
print("Fold:", fold)
print("Configuration:", configuration)

plans_file = preprocessed_base / task / f"{plans}_plans_3D.pkl"
needs_preprocessing = not plans_file.exists()

if plans_file.exists():
    ok, _, reason = stage0_integrity(preprocessed_base / task, plans)
    needs_preprocessing = not ok
    if needs_preprocessing:
        print(f"\nDetected incomplete preprocessing for {task}: {reason}")

if needs_preprocessing:
    if not raw_task_dir.exists():
        raw_candidates = sorted(p.name for p in raw_base.glob(f"Task{int(dataset_id):03d}_*") if p.is_dir())
        if not raw_candidates:
            raise FileNotFoundError(
                f"Preprocessing needed but raw folder '{raw_task_dir}' was not found and no alternatives exist."
            )
        raw_task_dir = raw_base / raw_candidates[0]
        print(
            f"Using raw folder {raw_task_dir.name} for preprocessing "
            f"(selected preprocessed task was {task})."
        )

    ensure_v1_dataset_json(raw_task_dir)

    print("\nMissing or incomplete v1 preprocessing outputs. Running MedNeXt preprocessing...")
    preprocess_env = os.environ.copy()
    preprocess_results = mednext_path / "nnUNet_results_preprocess_tmp"
    preprocess_results.mkdir(parents=True, exist_ok=True)
    preprocess_env["RESULTS_FOLDER"] = str(preprocess_results)

    subprocess.run([
        python_exe,
        "-m",
        "nnunet_mednext.experiment_planning.nnUNet_plan_and_preprocess",
        "-t",
        dataset_id,
        "-pl3d",
        planner,
        "-pl2d",
        "None",
    ], check=True, env=preprocess_env)

    # Task folder may differ after preprocessing (for example renamed suffixes).
    task = pick_task_name(raw_base, preprocessed_base, dataset_id, plans)
    print("Detected task after preprocessing:", task)
    plans_file = preprocessed_base / task / f"{plans}_plans_3D.pkl"

if not plans_file.exists():
    raise FileNotFoundError(f"Expected plans file not found after preprocessing: {plans_file}")

ok, stage0_dir, reason = stage0_integrity(preprocessed_base / task, plans)
if not ok:
    raise FileNotFoundError(
        f"Preprocessed stage data is incomplete for task {task}: {reason}. "
        f"Stage folder: {stage0_dir}"
    )


# ============================================================
# 6. TRAIN
# ============================================================
print("\nStarting MedNeXt training...")
print("Trainer:", trainer)

subprocess.run([
    python_exe,
    "-m",
    "nnunet_mednext.run.run_training",

    configuration,

    trainer,

    task,

    fold,

    "-p",
    plans

], check=True)