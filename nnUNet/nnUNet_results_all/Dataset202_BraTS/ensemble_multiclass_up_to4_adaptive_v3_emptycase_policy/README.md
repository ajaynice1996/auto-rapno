# Recomputed multiclass up-to-4 with empty-case policy
labels_dir=nnUNet_raw/Dataset202_BraTS/labelsTr_all
common_cases=52
levels_used=0,1,2,3,4
regions=ET,NET,CC,ED,TC={1,2,3},WT={1,2,3,4}
metric_policy=both_empty->1 ; one_empty->0
Note: per-case heavy files intentionally omitted in this optimized recompute; summary and best tables are complete.