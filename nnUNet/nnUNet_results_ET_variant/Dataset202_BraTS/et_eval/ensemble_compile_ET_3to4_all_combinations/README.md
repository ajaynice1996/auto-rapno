# ET Ensemble Expansion (3-model and 4-model, all combinations)

## Summary
- Exhaustive search over all 3-model and 4-model subsets from models M01..M10.
- For each subset, all k-of-n vote thresholds are evaluated.
- Outputs are split into with_M02_allowed and without_M02 folders.
- Individual model outputs are included with M02 intentionally listed last.

## Combination counts
- 3-model subsets: C(10,3)=120, thresholds per subset=3, rules=360
- 4-model subsets: C(10,4)=210, thresholds per subset=4, rules=840
- Total rules evaluated: 1200
- Cases evaluated per rule: 52

## Files
- individual_model_summary_M02_last.csv
- individual_model_per_case_M02_last.csv
- ensemble_3to4_all_combinations_summary.csv
- ensemble_3to4_all_combinations_per_case.csv
- ensemble_candidates_gt_067.csv
- ensemble_candidates_gt_068.csv
- ensemble_candidates_gt_069.csv
- ensemble_candidates_gt_070.csv
- with_M02_allowed/ensemble_3to4_summary_with_M02.csv
- with_M02_allowed/ensemble_3to4_per_case_with_M02.csv
- with_M02_allowed/best_rule_per_case_with_M02.csv
- with_M02_allowed/top100_with_M02.csv
- without_M02/ensemble_3to4_summary_without_M02.csv
- without_M02/ensemble_3to4_per_case_without_M02.csv
- without_M02/best_rule_per_case_without_M02.csv
- without_M02/top100_without_M02.csv
- PUBLISHABLE_OBSERVATIONS.txt
- METHODS_AND_PUBLICATION_NOTES.txt

## How to improve results
1. Use probability-map soft voting and calibrated thresholds.
2. Learn per-model weights on a dedicated tuning split.
3. Add ET-specific postprocessing and false-positive suppression.
4. Optimize a multi-objective criterion (DSC, IoU, HD95).
5. Validate frozen rules on untouched test data.