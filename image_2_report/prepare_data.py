"""Step 1 - build the paired image/report dataset.

* parses every report (.docx) into normalised sections + extracted keyword attributes (JSON)
* pairs reports with the 4 MRI modalities (+ segmentation) of Dataset202_BraTS
* creates patient-level splits (70/15/15 stratified, and nnU-Net fold based)
* caches resampled, normalised volumes so training is fast

Usage:  python prepare_data.py            (parse + split + cache)
        python prepare_data.py --no_cache (parse + split only)
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from i2r.attributes import extract_attributes  # noqa: E402
from i2r.report_parser import SECTIONS, parse_report_file  # noqa: E402
from i2r.volume import MODALITIES, preprocess_case  # noqa: E402

STD_NAME = re.compile(r"^BraTSPED_(\d{5})\.docx$")


def normalise_text(t):
    for a, b in {"\u2013": "-", "\u2014": "-", "\u2019": "'", "\u2018": "'", "\u201c": "'", "\u201d": "'",
                 "\u00a0": " ", "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00b1": "+/-"}.items():
        t = t.replace(a, b)
    t = re.sub(r"[^\x20-\x7e]", " ", t)  # drop remaining bullets / private-use glyphs
    return re.sub(r"\s+", " ", t).strip()


def collect_reports(reports_dir, quality):
    """Return {case_id: path}. Irregular file names and cross-folder duplicates are recorded, not used."""
    found = defaultdict(list)
    for f in sorted(Path(reports_dir).rglob("*.docx")):
        m = STD_NAME.match(f.name)
        if f.name.startswith("~$"):
            quality["ignored_lock_files"].append(str(f))
        elif not m:
            quality["ignored_irregular_names"].append(str(f))
        else:
            found[f"BraTSPED_{m.group(1)}"].append(f)
    chosen = {}
    for cid, files in found.items():
        if len(files) == 1:
            chosen[cid] = files[0]
            continue
        num = int(cid[-5:])

        def in_range(p):
            m = re.search(r"(\d+)\s*-\s*(\d+)", p.parent.name)
            return bool(m) and int(m.group(1)) <= num <= int(m.group(2))

        preferred = [p for p in files if in_range(p)] or files
        chosen[cid] = preferred[0]
        quality["duplicate_reports_resolved"][cid] = {"used": str(chosen[cid]), "all": [str(p) for p in files],
                                                      "rule": "file inside the folder whose range contains the id"}
    return chosen


def build_pairs(dataset_dir, case_ids):
    pairs = {}
    for cid in case_ids:
        mods = {m: str(dataset_dir / "imagesTr" / f"{cid}_{i:04d}.nii.gz") for i, m in enumerate(MODALITIES)}
        mods = {m: p for m, p in mods.items() if Path(p).exists()}
        seg = dataset_dir / "labelsTr" / f"{cid}.nii.gz"
        pairs[cid] = {"modalities": mods, "seg": str(seg) if seg.exists() else None}
    return pairs


def make_ratio_split(ids, strata, seed, val=0.15, test=0.15):
    from sklearn.model_selection import train_test_split
    ids = sorted(ids)
    s = [strata[i] for i in ids]
    train_val, test_ids = train_test_split(ids, test_size=test, stratify=s, random_state=seed)
    s2 = [strata[i] for i in train_val]
    train, val_ids = train_test_split(train_val, test_size=val / (1 - test), stratify=s2, random_state=seed)
    return {"train": sorted(train), "val": sorted(val_ids), "test": sorted(test_ids)}


def make_nnunet_fold_split(ids, strata, nnunet_splits, fold, seed, val=0.15):
    from sklearn.model_selection import train_test_split
    idset = set(ids)
    test_ids = sorted(c for c in nnunet_splits[fold]["val"] if c in idset)
    rest = sorted(idset - set(test_ids))
    val_frac = min(val * len(idset) / len(rest), 0.5)
    train, val_ids = train_test_split(rest, test_size=val_frac, stratify=[strata[i] for i in rest], random_state=seed)
    return {"train": sorted(train), "val": sorted(val_ids), "test": test_ids}


def _cache_one(args):
    cid, pair, size, out = args
    out = Path(out)
    if out.exists():
        return cid, "cached"
    arr = preprocess_case(pair["modalities"], pair["seg"], size)
    import numpy as np
    np.savez(out, **arr)
    return cid, "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default=str(ROOT.parent / "REPORTS"))
    ap.add_argument("--dataset_dir", default=str(ROOT.parent / "nnUNet/nnUNet_raw/Dataset202_BraTS"))
    ap.add_argument("--nnunet_splits", default=str(ROOT.parent / "nnUNet/nnUNet_preprocessed/Dataset202_BraTS/splits_final.json"))
    ap.add_argument("--nnunet_fold", type=int, default=2)
    ap.add_argument("--out_dir", default=str(ROOT / "data"))
    ap.add_argument("--size", type=int, default=128, help="cubic size of cached volumes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / f"cache_{args.size}").mkdir(exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    quality = {"ignored_lock_files": [], "ignored_irregular_names": [], "duplicate_reports_resolved": {}}
    report_files = collect_reports(args.reports_dir, quality)
    all_cases = sorted(p.name[: -len("_0000.nii.gz")] for p in (dataset_dir / "imagesTr").glob("*_0000.nii.gz"))
    paired = [c for c in all_cases if c in report_files and (dataset_dir / "labelsTr" / f"{c}.nii.gz").exists()]
    quality["cases_in_dataset"] = len(all_cases)
    quality["cases_with_report"] = len(paired)
    quality["cases_without_report"] = sorted(set(all_cases) - set(report_files))
    quality["reports_without_image"] = sorted(set(report_files) - set(all_cases))

    reports, strata = {}, {}
    for cid in paired:
        r = parse_report_file(report_files[cid])
        sec = {k: normalise_text(v) for k, v in r["sections"].items()}
        attrs = extract_attributes(sec)
        reports[cid] = {
            "case_id": cid,
            "source_file": r["source_file"],
            "patient_id_in_report": r["patient_id_in_report"],
            "raw_text": r["raw_text"],
            "sections": sec,
            "attributes": attrs,
            "target_json": json.dumps({k: sec[k] for k in SECTIONS}, ensure_ascii=False, separators=(",", ":")),
            "parse_warnings": r["parse_warnings"],
        }
        strata[cid] = attrs["tumor_type"]
    quality["reports_with_parse_warnings"] = {c: r["parse_warnings"] for c, r in reports.items() if r["parse_warnings"]}

    pairs = build_pairs(dataset_dir, paired)
    quality["cases_with_missing_modalities"] = {c: sorted(set(MODALITIES) - set(p["modalities"])) for c, p in pairs.items()
                                                if len(p["modalities"]) < len(MODALITIES)}
    quality["tumor_type_counts"] = dict(Counter(strata.values()))
    quality["grade_group_counts"] = dict(Counter(r["attributes"]["grade_group"] for r in reports.values()))

    json.dump(reports, open(out / "reports.json", "w"), indent=1)
    json.dump(pairs, open(out / "pairs.json", "w"), indent=1)

    ratio = make_ratio_split(paired, strata, args.seed)
    json.dump(ratio, open(out / "splits" / "ratio_70_15_15.json", "w"), indent=1)
    nn_path = Path(args.nnunet_splits)
    if nn_path.exists():
        nn = json.load(open(nn_path))
        fold = make_nnunet_fold_split(paired, strata, nn, args.nnunet_fold, args.seed)
        json.dump(fold, open(out / "splits" / f"nnunet_fold{args.nnunet_fold}.json", "w"), indent=1)
        quality["split_sizes_nnunet_fold"] = {k: len(v) for k, v in fold.items()}
    quality["split_sizes_ratio"] = {k: len(v) for k, v in ratio.items()}
    json.dump(quality, open(out / "data_quality_report.json", "w"), indent=1)
    print("paired cases:", len(paired), "| split sizes:", quality["split_sizes_ratio"])
    print("ignored irregular report files:", quality["ignored_irregular_names"])

    if not args.no_cache:
        jobs = [(c, pairs[c], args.size, out / f"cache_{args.size}" / f"{c}.npz") for c in paired]
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (cid, status) in enumerate(ex.map(_cache_one, jobs), 1):
                if i % 25 == 0 or i == len(jobs):
                    print(f"cached {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
