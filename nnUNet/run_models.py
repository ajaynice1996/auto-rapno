import os
import subprocess
from pathlib import Path
import torch

for var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
    os.environ.pop(var, None)  # removes variable if it exists

# Set environment variables for nnUNet
os.environ["nnUNet_raw"] = "nnUNet_raw"
os.environ["nnUNet_results"] = "nnUNet_results"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed"

# Define paths for the dataset

print(len(os.listdir('nnUNet_raw/Dataset202_BraTS/labelsTr')))

# Define dataset ID and configuration
dataset_id = "202"  # Custom dataset ID
config = "3d_fullres"  # Configuration type (2d, 3d_fullres, etc.)
fold = "2"  # GPU/device ID or fold

# # # Step 1: Run nnUNetv2_plan_and_preprocess
# print("Running nnUNetv2_plan_and_preprocess...")
# subprocess.run([
#     "nnUNetv2_plan_and_preprocess",
#     "-d", dataset_id,
#     "-c", config,
#     "-pl", "nnUNetPlannerResEncL",
#     "--verify_dataset_integrity"
#  ], check=True)

# Step 2: Optional – Count preprocessed files
preprocessed_dir = Path(f"nnUNet_preprocessed/Dataset202_BraTS/nnUNetPlans_3d_fullres")
if preprocessed_dir.exists():
    print(f"Number of preprocessed files in {preprocessed_dir}: {len(list(preprocessed_dir.iterdir()))}")
else:
    print(f"Preprocessed directory not found: {preprocessed_dir}")

# Check GPU status using PyTorch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"Memory Cached:    {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")
        print(f"Memory Free:      {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i))/1024**3:.2f} GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    print("\nCUDA cache cleared.")

# Step 3: Train the model
print("Starting training...")
subprocess.run([
    "nnUNetv2_train",
    dataset_id,
    config,
    fold,
    "-tr", "nnUNetTrainer_500epochs",
    "-p", "nnUNetResEncUNetLPlans",
], check=True)

## Step 3:  Primus

# print("Starting training...")
# subprocess.run([
#     "nnUNetv2_train",
#     dataset_id,
#     config,
#     fold,
#     "-tr", "nnUNet_PrimusV3S_Trainer",
#     "-p", "nnUNetResEncUNetLPlans"
# ], check=True)

# # Step 3: Train the model
# print("Starting training...")
# subprocess.run([
#    "nnUNetv2_train",
#    dataset_id,
#    config,
#    fold,
#    "-p", "nnUNetResEncUNetLPlans"
# ], check=True)

## Step 3:  Primus

# print("Starting training...")
# subprocess.run([
#   "nnUNetv2_train",
#    dataset_id,
#    config,
#    fold,
#    "-tr", "nnUNet_PrimusV2S_Trainer",
# ], check=True)

