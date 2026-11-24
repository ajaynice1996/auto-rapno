# BraTS Multi-class + WT/TC Ensemble (Up to 4 Models)

Computed from nnUNet_results_all/Dataset202_BraTS validation predictions (52 common cases).

## Levels and regions used
- Atomic levels: 0(background), 1(ET), 2(NET), 3(CC), 4(ED)
- TC = {1,2,3}
- WT = {1,2,3,4}

## Labels evaluated
- ET, NET, CC, ED, TC, WT

## Key files
- individual_models_per_case.csv
- individual_models_summary.csv
- ensemble_all_rules_per_case.csv
- ensemble_all_rules_summary.csv
- best_per_label_union.csv
- best_per_label_intersection.csv
- best_per_label_adaptive_ui.csv
- best_per_label_overall.csv
- adaptive_choice_distribution.csv
- leaderboard_best_union_intersection_adaptive.csv