# Union-only reference-style recomputation (empty-case policy fixed)
labels_dir=nnUNet_raw/Dataset202_BraTS/labelsTr_all
common_cases=52
rule=union_only
combinations=all_3_model_from_6
metric_policy=both_empty->1, one_empty->0