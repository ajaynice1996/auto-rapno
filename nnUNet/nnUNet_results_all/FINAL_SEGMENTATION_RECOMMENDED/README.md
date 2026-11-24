# FINAL_SEGMENTATION_RECOMMENDED

This folder contains a deployment-safe final segmentation ensemble with class-wise model combinations and a single evaluation/inference script.

## Final class-wise ensemble strategy
- ET: `ResEncL + PrimusV2S + ET500_ResEncL` with `adaptive_nogt_variation` (threshold 0.18)
- NET: `ResEncL + MedNeXt_k3_pp + MedNeXt_k5_pp` with `union`
- CC: `ResEncL + MedNeXt_k3_raw + MedNeXt_k3_pp` with `intersection`
- ED: `PrimusV3S + MedNeXt_k5_raw + MedNeXt_k5_pp` with `union`

TC and WT are evaluated from the fused final output:
- TC = labels {1,2,3}
- WT = labels {1,2,3,4}

## Why this is final and fair
- No GT is used for per-case rule selection at inference time.
- ET uses a prediction-variation selector (union vs intersection) based only on model disagreement.
- Empty-case policy for metrics is publication-standard:
  - both empty => Dice=1, IoU=1
  - one empty => Dice=0, IoU=0

## Main script
- `evaluate.py`

## Usage
Run all common cases and evaluate with labels:

```bash
/home/ashar/Documents/auto-rapno/nnUNet/.venv/bin/python nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED/evaluate.py --evaluate --cases all --output-dir nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED
```

Run a subset of cases:

```bash
/home/ashar/Documents/auto-rapno/nnUNet/.venv/bin/python nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED/evaluate.py --evaluate --cases BraTSPED_00003.nii.gz,BraTSPED_00005.nii.gz --output-dir nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED
```

Inference only (no metrics):

```bash
/home/ashar/Documents/auto-rapno/nnUNet/.venv/bin/python nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED/evaluate.py --cases all --output-dir nnUNet_results_all/FINAL_SEGMENTATION_RECOMMENDED
```

## Outputs
- `predictions/*.nii.gz`: final fused segmentation predictions
- `reports/selection_log_per_case.csv`: ET adaptive decision log and class rule details per case
- `reports/metrics_per_case.csv`: per-case metrics (if `--evaluate`)
- `reports/metrics_summary.csv`: mean metrics over cases (if `--evaluate`)
- `reports/publication_final_summary.txt`: publication-ready summary text
