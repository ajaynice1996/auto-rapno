A = runs/mri_only_scratch/eval_test
B = runs/mri_plus_seg_scratch/eval_test  (cases: 38)

| metric | A | B | B - A |
|---|---|---|---|
| ROUGE-1 | 0.479 | 0.473 | -0.006 |
| ROUGE-2 | 0.185 | 0.175 | -0.010 |
| ROUGE-L | 0.299 | 0.279 | -0.020 |
| BLEU-4 | 0.123 | 0.117 | -0.006 |
| JSON valid (strict) | 0.947 | 0.868 | -0.079 |
| JSON invalid | 0.000 | 0.000 | +0.000 |
| tumor type accuracy | 0.579 | 0.553 | -0.026 |
| tumor type macro-F1 | 0.280 | 0.524 | +0.244 |
| grade group accuracy | 0.658 | 0.553 | -0.105 |
| laterality accuracy | 0.184 | 0.447 | +0.263 |
| findings mean accuracy | 0.535 | 0.556 | +0.020 |
| locations micro-F1 | 0.644 | 0.655 | +0.011 |
| molecular markers mean accuracy | 0.553 | 0.647 | +0.095 |
| KEY-FINDING ACCURACY | 0.543 | 0.570 | +0.028 |
| combined score | 0.421 | 0.425 | +0.004 |

Paired bootstrap over cases (B - A, 95% CI):
- rougeL: -0.020  [-0.039, -0.001]
- attr_acc: +0.011  [-0.045, +0.063]
