# ET Ensemble Compilation (Label=1)

Compiled individual model ET metrics and ensemble search results.

## Scope
- Included requested ET sources and all discovered model prediction folders under nnUNet_results_all/Dataset202_BraTS with fold_2 validation predictions.
- Common cases evaluated across all included models: 52
- ET label evaluated: 1

## Files
- model_sources.csv
- individual_model_summary.csv
- individual_model_per_case.csv
- ensemble_candidate_summary.csv
- leaderboard_all_rules.csv
- best_ensemble_predictions/
- best_ensemble_metrics_per_case.csv
- best_ensemble_metrics_summary.csv

## Best Ensemble Rule
- Rule: pair:M02+M06:if_ov<0.02_then_inter_else_union
- Mean DSC: 0.694569
- Mean IoU: 0.627834
- Mean HD95 (mm): 2.244478
- >=0.68 reached: True
- >=0.70 reached: False
- >=0.80 reached: False

## Publishability Guidance
- Current best rule is optimized on this same validation split (risk of overfitting).
- For publication: freeze rule first, evaluate once on an untouched test split, and report CIs.
