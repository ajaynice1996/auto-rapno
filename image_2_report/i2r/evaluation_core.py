"""Evaluation of a trained run: generation, JSON validation, text / section / key-finding metrics, review sheet."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .data import CaseDataset, scale_tumor_feats
from .engine import (ROOT, baseline_predictions, generate_cases, joined, load_model, load_reports, load_split, score_predictions)
from .report_parser import SECTIONS
from .volume import preprocess_case


def evaluation_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run_dir", required=True, help="training run folder (contains best.pt, config.json, tokenizer)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split_file", default=None, help="defaults to the split file used for training")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/eval_<split>")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_baseline", action="store_true")
    return ap


def override_segmentation(ds, pairs, seg_dir, size):
    """Replace the ground-truth segmentation by predicted label maps (<seg_dir>/<case_id>.nii.gz), e.g. nnU-Net output."""
    for item in ds.items:
        cid = item["case_id"]
        p = Path(seg_dir) / f"{cid}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"predicted segmentation missing: {p}")
        arr = preprocess_case(pairs[cid]["modalities"], str(p), size)
        item["seg"] = torch.from_numpy(arr["seg"])
        item["tumor_feats"] = scale_tumor_feats(arr["tumor_feats"])


def format_report(sections):
    if not sections:
        return ""
    return "\n".join(f"{s.upper()}: {sections.get(s, '')}" for s in SECTIONS)


def write_review_sheet(path, preds, reports):
    cols = ["case_id", "reference_report", "generated_report", "json_status", "accuracy_1to5", "completeness_1to5",
            "hallucination_yes_no", "clinically_significant_error_yes_no", "comments"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in preds:
            gen = format_report(p["generated_sections"]) or p["generated_raw"]
            w.writerow([p["case_id"], format_report(reports[p["case_id"]]["sections"]), gen, p["parse_status"], "", "", "", "", ""])


def summary_text(m, baseline=None):
    def row(name, a, b=None):
        return f"{name:<34}{a:>9.3f}" + (f"{b:>12.3f}" if b is not None else "")

    bm = baseline
    lines = [f"cases: {m['n_cases']}", "", f"{'metric':<34}{'model':>9}" + (f"{'centroid':>12}" if bm else "")]
    for k in ("rouge1", "rouge2", "rougeL", "bleu1", "bleu4"):
        lines.append(row(k, m["text"][k], bm["text"][k] if bm else None))
    A = m["attributes"]
    for k, v in (("tumor_type accuracy", A["tumor_type"]["accuracy"]), ("grade_group accuracy", A["grade_group"]["accuracy"]),
                 ("laterality accuracy", A["laterality"]["accuracy"]), ("findings mean accuracy", A["findings_mean_accuracy"]),
                 ("locations micro-F1", A["locations_micro_f1"]), ("molecular markers mean acc", A["molecular_markers_mean_accuracy"]),
                 ("KEY-FINDING ACCURACY", A["key_finding_accuracy"])):
        b = None
        if bm:
            B = bm["attributes"]
            b = {"tumor_type accuracy": B["tumor_type"]["accuracy"], "grade_group accuracy": B["grade_group"]["accuracy"],
                 "laterality accuracy": B["laterality"]["accuracy"], "findings mean accuracy": B["findings_mean_accuracy"],
                 "locations micro-F1": B["locations_micro_f1"], "molecular markers mean acc": B["molecular_markers_mean_accuracy"],
                 "KEY-FINDING ACCURACY": B["key_finding_accuracy"]}[k]
        lines.append(row(k, v, b))
    lines += ["", "JSON: " + ", ".join(f"{k}={v:.2%}" for k, v in m["json"].items()), "", "section ROUGE-L:"]
    lines += [f"  {s:<18}{v['rougeL']:.3f}" for s, v in m["sections"].items()]
    if "aux_heads" in m:
        lines += ["", "auxiliary-head predictions (image only, no text): " + ", ".join(f"{k}={v:.3f}" for k, v in m["aux_heads"].items())]
    return "\n".join(lines)


def run_evaluation(args, seg_mode, pred_seg_dir=None):
    run_dir = Path(args.run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    if bool(cfg["use_seg"]) != seg_mode:
        script = "evaluation_segmentation.py" if cfg["use_seg"] else "evaluation.py"
        sys.exit(f"This run was trained {'with' if cfg['use_seg'] else 'without'} segmentation; use {script}.")
    model, tok, cfg, device = load_model(run_dir, args.ckpt)
    reports = load_reports(cfg["data_dir"])
    split = load_split(args.split_file or cfg["split_file"])
    cache = Path(cfg["data_dir"]) / f"cache_{cfg['size']}"
    ds = CaseDataset(split[args.split], reports, cache, tok, cfg["max_len"])
    if pred_seg_dir:
        pairs = json.load(open(Path(cfg["data_dir"]) / "pairs.json"))
        override_segmentation(ds, pairs, pred_seg_dir, cfg["size"])
    out = Path(args.out_dir or run_dir / f"eval_{args.split}{'_predseg' if pred_seg_dir else ''}")
    out.mkdir(parents=True, exist_ok=True)

    preds = generate_cases(model, ds, tok, device, args.batch_size, cfg["max_len"])
    metrics, per_case, pred_attrs = score_predictions(reports, preds)
    base_metrics = None
    if not args.no_baseline:
        centroid, bpreds = baseline_predictions(reports, split["train"], split[args.split])
        base_metrics, base_per_case, _ = score_predictions(reports, bpreds, with_aux_heads=False)
        metrics["baseline_centroid_report"] = {"case_id": centroid, **base_metrics}
        json.dump(base_per_case, open(out / "per_case_baseline.json", "w"), indent=1)

    for p, a in zip(preds, pred_attrs):
        p["generated_attributes"] = a
        p["reference_sections"] = reports[p["case_id"]]["sections"]
        p["reference_attributes"] = reports[p["case_id"]]["attributes"]
    json.dump(preds, open(out / "predictions.json", "w"), indent=1)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    json.dump(per_case, open(out / "per_case.json", "w"), indent=1)
    write_review_sheet(out / "radiologist_review_sheet.csv", preds, reports)
    text = summary_text(metrics, base_metrics)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nresults written to {out}")
    return metrics
