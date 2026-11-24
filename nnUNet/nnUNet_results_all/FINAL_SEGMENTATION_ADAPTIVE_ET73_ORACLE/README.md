# FINAL_SEGMENTATION_ADAPTIVE_ET73_ORACLE

Alternate adaptive final package focused on the ET~0.737 class-wise adaptive configuration from prior analysis.

Class-wise heads (oracle adaptive_ui):
- ET: MedNeXt_k3_pp + MedNeXt_k5_pp + ET500_ResEncL
- NET: nnUNet + PrimusV3S + MedNeXt_k3_raw + MedNeXt_k3_pp
- CC: nnUNet + ResEncL + MedNeXt_k3_pp + MedNeXt_k5_pp
- ED: nnUNet + MedNeXt_k3_pp + MedNeXt_k5_pp

Rule:
- For each case and class, compute union and intersection.
- Select higher Dice; if Dice tie, select higher IoU.
- This selection uses GT (oracle), so it is for analysis/reporting rather than deployment inference.

Run:

```bash
/home/ashar/Documents/auto-rapno/nnUNet/.venv/bin/python nnUNet_results_all/FINAL_SEGMENTATION_ADAPTIVE_ET73_ORACLE/evaluate.py --evaluate --cases all --output-dir nnUNet_results_all/FINAL_SEGMENTATION_ADAPTIVE_ET73_ORACLE
```

Outputs:
- predictions/*.nii.gz
- reports/selection_log_per_case.csv
- reports/metrics_per_case.csv
- reports/metrics_summary.csv
- reports/publication_final_summary.txt
