"""Evaluation metrics: report text overlap, section-wise scores, JSON validity and key-finding accuracy."""
import math
import re
from collections import Counter

import numpy as np

from .attributes import AUX_FINDINGS, AUX_LOCATIONS, KEY_MOLECULAR, TRI_STATES
from .report_parser import SECTIONS


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", (text or "").lower())


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(refs, hyps, max_n=4):
    """Corpus BLEU with add-1 smoothing for n>1. refs/hyps: lists of strings. Returns dict bleu1..bleu4."""
    clipped = [0] * max_n
    total = [0] * max_n
    ref_len = hyp_len = 0
    for r, h in zip(refs, hyps):
        rt, ht = tokenize(r), tokenize(h)
        ref_len += len(rt)
        hyp_len += len(ht)
        for n in range(1, max_n + 1):
            hn, rn = _ngrams(ht, n), _ngrams(rt, n)
            clipped[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(len(ht) - n + 1, 0)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    out = {}
    for k in range(1, max_n + 1):
        logs = []
        for n in range(1, k + 1):
            num, den = clipped[n - 1], total[n - 1]
            if n > 1:
                num, den = num + 1, den + 1
            logs.append(math.log(num / den) if num > 0 and den > 0 else -1e9)
        out[f"bleu{k}"] = bp * math.exp(sum(logs) / k)
    return out


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _f(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def rouge_scores(ref, hyp):
    rt, ht = tokenize(ref), tokenize(hyp)
    out = {}
    for n in (1, 2):
        rn, hn = _ngrams(rt, n), _ngrams(ht, n)
        overlap = sum(min(c, rn[g]) for g, c in hn.items())
        out[f"rouge{n}"] = _f(overlap / max(sum(hn.values()), 1), overlap / max(sum(rn.values()), 1))
    lcs = _lcs(rt, ht)
    out["rougeL"] = _f(lcs / max(len(ht), 1), lcs / max(len(rt), 1))
    return out


def text_metrics(ref_texts, hyp_texts):
    per_case = [rouge_scores(r, h) for r, h in zip(ref_texts, hyp_texts)]
    agg = {k: float(np.mean([c[k] for c in per_case])) if per_case else 0.0 for k in ("rouge1", "rouge2", "rougeL")}
    agg.update(corpus_bleu(ref_texts, hyp_texts))
    return agg, per_case


def section_text_metrics(ref_sections, hyp_sections):
    """ROUGE-L / ROUGE-1 per report section (missing generated sections count as empty)."""
    out = {}
    for s in SECTIONS:
        scores = [rouge_scores(r.get(s, ""), (h or {}).get(s, "")) for r, h in zip(ref_sections, hyp_sections)]
        out[s] = {"rougeL": float(np.mean([x["rougeL"] for x in scores])), "rouge1": float(np.mean([x["rouge1"] for x in scores]))}
    return out


def _acc_f1(y_true, y_pred, labels):
    from sklearn.metrics import f1_score
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if y_true else 0.0
    return acc, f1


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def attribute_metrics(ref_attrs, pred_attrs):
    """Compare extracted attributes of references and predictions (lists of attribute dicts)."""
    from .attributes import GRADE_GROUPS, LATERALITIES, TUMOR_TYPES
    res = {}
    for key, labels in (("tumor_type", TUMOR_TYPES), ("grade_group", GRADE_GROUPS), ("laterality", LATERALITIES)):
        acc, f1 = _acc_f1([a[key] for a in ref_attrs], [a[key] for a in pred_attrs], labels)
        res[key] = {"accuracy": acc, "macro_f1": f1}

    # strict WHO grade set match (grade_group above is the lenient version)
    res["who_grade_exact"] = {"accuracy": float(np.mean([a["who_grades"] == b["who_grades"] for a, b in zip(ref_attrs, pred_attrs)]))}

    findings = {}
    for f in AUX_FINDINGS:
        yt, yp = [a["findings"][f] for a in ref_attrs], [a["findings"][f] for a in pred_attrs]
        acc, _ = _acc_f1(yt, yp, TRI_STATES)
        tp = sum(1 for t, p in zip(yt, yp) if t == "present" and p == "present")
        fp = sum(1 for t, p in zip(yt, yp) if t != "present" and p == "present")
        fn = sum(1 for t, p in zip(yt, yp) if t == "present" and p != "present")
        findings[f] = {"accuracy": acc, "present_f1": _f1(tp, fp, fn), "support_present": sum(t == "present" for t in yt)}
    res["findings"] = findings
    res["findings_mean_accuracy"] = float(np.mean([v["accuracy"] for v in findings.values()]))

    loc = {}
    tp_all = fp_all = fn_all = 0
    for l in AUX_LOCATIONS:
        yt, yp = np.array([a["locations"][l] for a in ref_attrs]), np.array([a["locations"][l] for a in pred_attrs])
        tp, fp, fn = int(((yt == 1) & (yp == 1)).sum()), int(((yt == 0) & (yp == 1)).sum()), int(((yt == 1) & (yp == 0)).sum())
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        loc[l] = {"f1": _f1(tp, fp, fn), "accuracy": float((yt == yp).mean())}
    res["locations"] = loc
    res["locations_micro_f1"] = _f1(tp_all, fp_all, fn_all)
    res["locations_mean_accuracy"] = float(np.mean([v["accuracy"] for v in loc.values()]))

    mol = {}
    for m in KEY_MOLECULAR:
        yt, yp = np.array([a["molecular_markers"][m] for a in ref_attrs]), np.array([a["molecular_markers"][m] for a in pred_attrs])
        mol[m] = {"accuracy": float((yt == yp).mean())}
    res["molecular_markers"] = mol
    res["molecular_markers_mean_accuracy"] = float(np.mean([v["accuracy"] for v in mol.values()]))

    res["key_finding_accuracy"] = float(np.mean([
        res["tumor_type"]["accuracy"], res["grade_group"]["accuracy"], res["laterality"]["accuracy"],
        res["findings_mean_accuracy"], res["locations_mean_accuracy"],
    ]))
    return res


def per_case_attribute_accuracy(ref_attrs, pred_attrs):
    """Mean agreement over all headline attributes, per case (used for paired bootstrap)."""
    scores = []
    for a, b in zip(ref_attrs, pred_attrs):
        s = [a["tumor_type"] == b["tumor_type"], a["grade_group"] == b["grade_group"], a["laterality"] == b["laterality"]]
        s += [a["findings"][f] == b["findings"][f] for f in AUX_FINDINGS]
        s += [a["locations"][l] == b["locations"][l] for l in AUX_LOCATIONS]
        scores.append(float(np.mean(s)))
    return scores


def paired_bootstrap(scores_a, scores_b, n_boot=2000, seed=0):
    """Mean difference (b - a) with 95% CI over cases."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(float((b[idx] - a[idx]).mean()))
    return {"mean_diff": float((b - a).mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_b_not_better": float(np.mean(np.asarray(diffs) <= 0))}
