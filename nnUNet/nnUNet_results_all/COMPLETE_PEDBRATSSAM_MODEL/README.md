# COMPLETE_PEDBRATSSAM_MODEL
labels_dir=nnUNet_raw/Dataset202_BraTS/labelsTr_all
common_cases=52
all_class_models=8
et_only_models=3
combination_sizes=3,4
rules=union,intersection,adaptive_ui
metric_policy=both_empty->1; one_empty->0
pool_policy=ET(all+ET); others(all-class-only)

Files:
- model_inventory.csv
- individual_models_summary.csv
- ensemble_all_rules_summary.csv
- best_per_label_union.csv
- best_per_label_intersection.csv
- best_per_label_adaptive_ui.csv
- best_per_label_overall.csv
- best_et_by_rule.csv
- publication_ready_best_results.txt
- adaptive_rule_used.txt
- other_observations.txt
- quality_flags.csv