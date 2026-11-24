"""Compare two evaluated runs (e.g. MRI only vs MRI + segmentation) on the same split.

    python compare_runs.py --a runs/mri_only_scratch/eval_test --b runs/mri_plus_seg_scratch/eval_test

Writes comparison.md / comparison.json next to --out (default: runs/comparison_<a>_vs_<b>).
Differences are b - a. The paired bootstrap runs over test cases; with ~38 cases the confidence
intervals are wide, so treat small differences as inconclusive.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from i2r.metrics import paired_bootstrap  # noqa: E402

ROWS = [
    ("ROUGE-1", ("text", "rouge1")), ("ROUGE-2", ("text", "rouge2")), ("ROUGE-L", ("text", "rougeL")), ("BLEU-4", ("text", "bleu4")),
    ("JSON valid (strict)", ("json", "valid")), ("JSON invalid", ("json", "invalid")),
    ("tumor type accuracy", ("attributes", "tumor_type", "accuracy")),
    ("tumor type macro-F1", ("attributes", "tumor_type", "macro_f1")),
    ("grade group accuracy", ("attributes", "grade_group", "accuracy")),
    ("laterality accuracy", ("attributes", "laterality", "accuracy")),
    ("findings mean accuracy", ("attributes", "findings_mean_accuracy")),
    ("locations micro-F1", ("attributes", "locations_micro_f1")),
    ("molecular markers mean accuracy", ("attributes", "molecular_markers_mean_accuracy")),
    ("KEY-FINDING ACCURACY", ("attributes", "key_finding_accuracy")),
    ("combined score", ("score",)),
]


def dig(d, path):
    for p in path:
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="eval folder of run A (baseline, e.g. without segmentation)")
    ap.add_argument("--b", required=True, help="eval folder of run B (e.g. with segmentation)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    ma, mb = json.load(open(A / "metrics.json")), json.load(open(B / "metrics.json"))
    pa, pb = json.load(open(A / "per_case.json")), json.load(open(B / "per_case.json"))
    common = sorted(set(pa) & set(pb))
    if len(common) != len(pa) or len(common) != len(pb):
        print(f"warning: runs share {len(common)} cases (A={len(pa)}, B={len(pb)}); bootstrap uses the shared cases")

    table = [{"metric": n, "A": dig(ma, p), "B": dig(mb, p), "diff": dig(mb, p) - dig(ma, p)} for n, p in ROWS]
    boot = {k: paired_bootstrap([pa[c][k] for c in common], [pb[c][k] for c in common]) for k in ("rougeL", "attr_acc")}
    out = Path(args.out or A.parent.parent / f"comparison_{A.parent.name}_vs_{B.parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"A": str(A), "B": str(B), "n_shared_cases": len(common), "table": table, "paired_bootstrap_b_minus_a": boot},
              open(out / "comparison.json", "w"), indent=1)
    lines = [f"A = {A}", f"B = {B}  (cases: {len(common)})", "", "| metric | A | B | B - A |", "|---|---|---|---|"]
    lines += [f"| {r['metric']} | {r['A']:.3f} | {r['B']:.3f} | {r['diff']:+.3f} |" for r in table]
    lines += ["", "Paired bootstrap over cases (B - A, 95% CI):"]
    lines += [f"- {k}: {v['mean_diff']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in boot.items()]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
