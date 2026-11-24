# ET Recompute: 2/3/4-Model Union-Intersection-Majority Search

## Scope
- Cases: 52
- Models: 10 (M01, M02, M03, M04, M05, M06, M07, M08, M09, M10)
- Subset sizes: 2, 3, 4
- Rules: union, intersection, vote>=2_of_3, vote>=2_of_4, vote>=3_of_4
- Total rules evaluated: 1290

## Best results
- Overall: 3model:M01+M05+M07:union | DSC=0.515458 | IoU=0.437164
- With M02: 4model:M01+M02+M03+M05:vote>=2_of_4 | DSC=0.513598 | IoU=0.440715
- Without M02: 3model:M01+M05+M07:union | DSC=0.515458 | IoU=0.437164

## Files
- ensemble_2to4_union_intersection_majority_summary.csv
- ensemble_2to4_union_intersection_majority_per_case.csv
- individual_model_summary_M02_last_fixed.csv
- ensemble_candidates_gt_066.csv
- ensemble_candidates_gt_067.csv
- best_per_subsetsize_rule.csv
- top20_hd95_summary.csv
- reference_previous_top_pairwise_rules.csv
- with_M02_allowed/summary_with_M02.csv
- with_M02_allowed/best_rule_per_case_with_M02.csv
- without_M02/summary_without_M02.csv
- without_M02/best_rule_per_case_without_M02.csv
- OBSERVATIONS.txt
- METHODS_AND_IMPROVEMENT.txt

## Improvement directions
1. Keep adaptive pairwise overlap rule family (previously best near 0.694).
2. Add 3-model adaptive overlap/weighted soft-vote rules.
3. Use probability maps and calibration.
4. Tune ET-specific postprocessing and test on held-out split.