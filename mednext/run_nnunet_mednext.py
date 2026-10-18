import os
import subprocess
from pathlib import Path
import nnunet_mednext
# from nnunet_mednext.paths import nnUNet_raw_data_base, nnUNet_preprocessed, RESULTS_FOLDER

# ---------- Configuration ----------
dataset_id = "202"
dataset_name = f"Dataset{dataset_id}"
config = "3d_fullres"
trainer = "nnUNetTrainerV2MedNeXt"
plans = "nnUNetResEncUNetPlans"
fold = "2"
max_epochs = "5"  # Short training
# input_test_folder = f"../nnUNet/nnUNet_raw/{dataset_name}/imagesTs"
output_folder = f"results/{dataset_name}/fold{fold}"

# ---------- Set nnUNet environment from external folder ----------
# Define absolute paths correctly
mednext_path = 'mednext'

# Set required nnUNet environment variables
os.environ["nnUNet_raw_data_base"] = os.path.join(mednext_path, "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"] = os.path.join(mednext_path, "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = os.path.join(mednext_path, "nnUNet_results")

print("✅ Environment configured:")
for key in ["nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER"]:
    print(f"{key} = {os.environ[key]}")

# ---------- Step 1: Plan & Preprocess ----------
print("🧠 Running MedNeXt planner...")
subprocess.run([
    "mednextv1_plan_and_preprocess",
    "-t", dataset_id,
    "-pl3d", "ExperimentPlanner3D_v21_customTargetSpacing_1x1x1" 
], check=True)

# ---------- Step 2: Train ----------
print("🚀 Starting training (fewer epochs)...")
subprocess.run([
    "mednextv1_train",
    dataset_id,
    config,
    fold,
    "-tr", trainer,
    "-p", plans,
    "--epochs", max_epochs
], check=True)