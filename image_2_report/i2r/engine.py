"""Shared training / generation / scoring logic used by the train*.py and evaluation*.py entry points."""
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .attributes import AUX_FINDINGS, AUX_LOCATIONS, extract_attributes
from .data import CaseDataset, collate
from .metrics import (attribute_metrics, per_case_attribute_accuracy, rouge_scores, section_text_metrics, text_metrics)
from .model import Image2Report
from .output_parser import parse_generated
from .report_parser import SECTIONS
from .tokenizer import HFTokenizer, ScratchTokenizer

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_reports(data_dir):
    return json.load(open(Path(data_dir) / "reports.json"))


def load_split(split_file):
    return json.load(open(split_file))


def build_tokenizer(cfg, reports, train_ids, run_dir=None):
    if cfg["decoder"] == "scratch":
        tok = ScratchTokenizer.train([reports[c]["target_json"] for c in train_ids], cfg.get("vocab_size", 4000))
    else:
        tok = HFTokenizer(cfg["decoder"])
    if run_dir:
        tok.save(Path(run_dir) / "tokenizer.json")
    return tok


def load_tokenizer(run_dir, cfg):
    p = Path(run_dir) / "tokenizer.json"
    return ScratchTokenizer.load(p) if cfg["decoder"] == "scratch" else HFTokenizer.load(p)


def make_loader(ds, batch_size, shuffle, pad_id):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(collate, pad_id=pad_id), num_workers=0, drop_last=False)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def augment(batch, p_drop=0.1, max_shift=6):
    """GPU augmentation. No left/right flips: laterality words in the report would become wrong."""
    img = batch["image"].float()
    B = img.size(0)
    brain = (img != 0).float()
    gain = 1 + 0.1 * (torch.rand(B, 4, 1, 1, 1, device=img.device) * 2 - 1)
    bias = 0.1 * (torch.rand(B, 4, 1, 1, 1, device=img.device) * 2 - 1)
    img = (img * gain + bias + 0.03 * torch.randn_like(img)) * brain
    mask = batch["modality_mask"].clone()
    drop = (torch.rand_like(mask) < p_drop).float()
    new_mask = mask * (1 - drop)
    new_mask = torch.where(new_mask.sum(1, keepdim=True) == 0, mask, new_mask)
    img = img * new_mask[:, :, None, None, None]
    seg = batch["seg"].clone()
    for i in range(B):
        sh = tuple(int(s) for s in torch.randint(-max_shift, max_shift + 1, (3,)))
        img[i] = torch.roll(img[i], sh, (1, 2, 3))
        seg[i] = torch.roll(seg[i], sh, (0, 1, 2))
    batch = dict(batch)
    batch.update(image=img, modality_mask=new_mask, seg=seg)
    return batch


@torch.no_grad()
def generate_cases(model, ds, tokenizer, device, batch_size=16, max_len=448):
    """Run generation on a dataset. Returns list of prediction dicts (raw text, parsed JSON, aux head output)."""
    model.eval()
    loader = make_loader(ds, batch_size, False, tokenizer.pad_id)
    preds = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids, aux = model.generate(batch, max_len=max_len)
        for i, cid in enumerate(batch["case_id"]):
            parsed = parse_generated(tokenizer.decode(ids[i]))
            preds.append({
                "case_id": cid,
                "generated_raw": parsed["raw_text"],
                "parse_status": parsed["status"],
                "parse_errors": parsed["errors"],
                "generated_sections": parsed["sections"],
                "aux_head_prediction": {k: v[i] for k, v in aux.items()},
            })
    return preds


def sections_for_attributes(pred):
    if pred["generated_sections"]:
        return pred["generated_sections"]
    return {"findings": pred["generated_raw"]}


def joined(sections):
    return " ".join((sections or {}).get(s, "") for s in SECTIONS)


def aux_head_metrics(ref_attrs, preds):
    heads = [p["aux_head_prediction"] for p in preds]
    out = {k: float(np.mean([r[k] == h[k] for r, h in zip(ref_attrs, heads)])) for k in ("tumor_type", "grade_group", "laterality")}
    out["findings_mean_accuracy"] = float(np.mean([r["findings"][f] == h["findings"][f] for r, h in zip(ref_attrs, heads) for f in AUX_FINDINGS]))
    out["locations_mean_accuracy"] = float(np.mean([r["locations"][l] == h["locations"][l] for r, h in zip(ref_attrs, heads) for l in AUX_LOCATIONS]))
    return out


def score_predictions(reports, preds, with_aux_heads=True):
    """Full metric dict + per-case scores for a list of predictions (ordered like the dataset)."""
    refs = [reports[p["case_id"]] for p in preds]
    ref_sections = [r["sections"] for r in refs]
    ref_attrs = [r["attributes"] for r in refs]
    hyp_texts = [joined(p["generated_sections"]) if p["generated_sections"] else p["generated_raw"] for p in preds]
    pred_attrs = [extract_attributes(sections_for_attributes(p)) for p in preds]
    text, per_case_text = text_metrics([joined(s) for s in ref_sections], hyp_texts)
    status = [p["parse_status"] for p in preds]
    metrics = {
        "n_cases": len(preds),
        "json": {k: status.count(k) / len(status) for k in ("valid", "repaired", "partial", "invalid")},
        "text": text,
        "sections": section_text_metrics(ref_sections, [p["generated_sections"] for p in preds]),
        "attributes": attribute_metrics(ref_attrs, pred_attrs),
    }
    if with_aux_heads and preds and "aux_head_prediction" in preds[0]:
        metrics["aux_heads"] = aux_head_metrics(ref_attrs, preds)
    metrics["score"] = float(0.5 * text["rougeL"] + 0.5 * metrics["attributes"]["key_finding_accuracy"])
    per_case = {p["case_id"]: {"rougeL": t["rougeL"], "attr_acc": a}
                for p, t, a in zip(preds, per_case_text, per_case_attribute_accuracy(ref_attrs, pred_attrs))}
    return metrics, per_case, pred_attrs


def centroid_report(reports, train_ids):
    """Training report most similar (ROUGE-1) to all other training reports; used as a trivial baseline."""
    ids = list(train_ids)
    texts = {c: joined(reports[c]["sections"]) for c in ids}
    best, best_s = None, -1
    for c in ids:
        s = np.mean([rouge_scores(texts[o], texts[c])["rouge1"] for o in ids if o != c])
        if s > best_s:
            best, best_s = c, s
    return best


def baseline_predictions(reports, train_ids, case_ids):
    c = centroid_report(reports, train_ids)
    return c, [{"case_id": cid, "generated_raw": reports[c]["target_json"], "parse_status": "valid",
                "parse_errors": [], "generated_sections": reports[c]["sections"]} for cid in case_ids]


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train(cfg, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = load_reports(cfg["data_dir"])
    split = load_split(cfg["split_file"])
    cache = Path(cfg["data_dir"]) / f"cache_{cfg['size']}"
    tok = build_tokenizer(cfg, reports, split["train"], run_dir)
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)

    t0 = time.time()
    train_ds = CaseDataset(split["train"], reports, cache, tok, cfg["max_len"])
    val_ds = CaseDataset(split["val"], reports, cache, tok, cfg["max_len"])
    print(f"loaded train={len(train_ds)} val={len(val_ds)} in {time.time() - t0:.0f}s | use_seg={cfg['use_seg']} decoder={cfg['decoder']}", flush=True)

    model = Image2Report(cfg, tok).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    dec_params = list(model.decoder.lm.parameters()) if cfg["decoder"] != "scratch" else []
    dec_ids = {id(p) for p in dec_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in dec_ids], "lr": cfg["lr"]}]
    if dec_params:
        groups.append({"params": dec_params, "lr": cfg["lm_lr"]})
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])

    loader = make_loader(train_ds, cfg["batch_size"], True, tok.pad_id)
    val_loader = make_loader(val_ds, cfg["batch_size"], False, tok.pad_id)
    total = cfg["epochs"] * len(loader)
    warmup = cfg["warmup_epochs"] * len(loader)
    step, best, bad = 0, -1.0, 0
    log = open(run_dir / "history.jsonl", "w")
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tr = time.time(), {"loss": 0.0, "lm_loss": 0.0, "aux_loss": 0.0}
        for batch in loader:
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = cosine_lr(step, total, warmup, b)
            batch = augment(to_device(batch, device), cfg["modality_dropout"])
            with torch.autocast(**amp):
                out = model(batch, cfg["aux_weight"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for k in tr:
                tr[k] += float(out[k].detach()) / len(loader)
        model.eval()
        vl = {"loss": 0.0, "lm_loss": 0.0, "aux_loss": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(**amp):
                    out = model(to_device(batch, device), cfg["aux_weight"])
                for k in vl:
                    vl[k] += float(out[k].detach()) / len(val_loader)
        rec = {"epoch": epoch, "train": tr, "val": vl, "sec": round(time.time() - t0, 1)}

        if epoch % cfg["val_gen_every"] == 0 or epoch == cfg["epochs"]:
            preds = generate_cases(model, val_ds, tok, device, cfg["batch_size"] * 2, cfg["max_len"])
            m, _, _ = score_predictions(reports, preds)
            rec["val_gen"] = {"score": m["score"], "rougeL": m["text"]["rougeL"], "bleu4": m["text"]["bleu4"],
                              "key_finding_accuracy": m["attributes"]["key_finding_accuracy"], "json_valid": m["json"]["valid"]}
            if m["score"] > best:
                best, bad = m["score"], 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_score": best}, run_dir / "best.pt")
                rec["saved_best"] = True
            else:
                bad += 1
        log.write(json.dumps(rec) + "\n")
        log.flush()
        gen = (" | " + " ".join(f"{k}={v:.3f}" for k, v in rec["val_gen"].items())) if "val_gen" in rec else ""
        print(f"ep {epoch:3d} train {tr['loss']:.3f} (lm {tr['lm_loss']:.3f}) val {vl['loss']:.3f} (lm {vl['lm_loss']:.3f}){gen} {rec['sec']}s", flush=True)
        if cfg["patience"] and bad >= cfg["patience"]:
            print(f"early stopping: no val-score improvement for {bad} generation checks", flush=True)
            break
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, run_dir / "last.pt")
    json.dump({"best_val_score": best, "epochs_run": epoch}, open(run_dir / "train_summary.json", "w"), indent=1)
    print(f"done. best val score {best:.4f} -> {run_dir / 'best.pt'}", flush=True)


def load_model(run_dir, ckpt="best.pt", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(Path(run_dir) / ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    tok = load_tokenizer(run_dir, cfg)
    model = Image2Report(cfg, tok).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tok, cfg, device
