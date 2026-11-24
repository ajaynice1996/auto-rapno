# Ensemble Combination Report (ET label=1)

## Scope
- Total models evaluated: 10
- Total ensemble combinations evaluated: 367
- Total cases: 52

## Best Overall Ensemble
- Combination: pair:M02+M06:if_ov<0.02_then_inter_else_union
- Mean DSC: 0.694569
- Mean IoU: 0.627834
- Mean HD95 (best ensemble summary): 2.244478

## Worst Overall Ensemble
- Combination: pair:M02+M07:union
- Mean DSC: 0.579750
- Mean IoU: 0.505072

## Best/Worst Individual Models
- Best individual: nnUNet_results__Dataset202_BraTS__nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres__fold_2__validation | DSC=0.663635 | IoU=0.593771 | HD95=4.180372
- Worst individual: nnUNet_results_all__Dataset202_BraTS__nnUNetTrainer__nnUNetPlans__3d_fullres__fold_2__validation | DSC=0.592463 | IoU=0.518310 | HD95=6.814489

## Boundary Condition Guidance (Lower HD95 is Better)
- Best boundary-friendly individual model: nnUNet_results_ET_variant__Dataset202_BraTS__et_eval__predictions__nnUNet_PrimusV3S_Trainer__nnUNetResEncUNetLPlans__3d_fullres | HD95=2.275515
- Worst boundary individual model: nnUNet_results_all__Dataset202_BraTS__nnUNetTrainerV2_MedNeXt_S_kernel5__nnUNetPlansv2.1_trgSp_1x1x1__fold_2__validation_raw_postprocessed | HD95=6.958394
- Best ensemble HD95: 2.244478

## Files Added
- ensemble_candidate_summary_expanded.csv
- best_combinations_per_variant_pair.csv
- COMBINATION_REPORT.md

## Notes
- Ensemble ranking is based on validation fold_2 and ET label only.
- Publishable claims require frozen rule and external test validation.