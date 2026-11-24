"""Keyword/rule based extraction of structured attributes from report text.

Used for (a) building the evaluation JSON from reference reports, (b) extracting the same fields from
generated reports so that reference and prediction are compared on identical, clinically meaningful
fields, and (c) auxiliary training targets. Rules are deliberately transparent and heuristic; the
original report text is always kept next to the extracted values.
"""
import re

# tri-state findings: present / absent / not_mentioned
FINDING_TERMS = {
    "enhancement": r"enhanc\w*|contrast uptake",
    "necrosis": r"necro\w*",
    "cystic_change": r"cyst\w*",
    "hemorrhage": r"h(?:a)?emorrhag\w*|haemorrhag\w*|blood products?|susceptibility",
    "calcification": r"calcif\w*",
    "edema": r"oedema|edema",
    "mass_effect": r"mass effect",
    "midline_shift": r"midline shift",
    "hydrocephalus": r"hydrocephalus",
    "restricted_diffusion": r"restrict\w* diffusion|diffusion restriction",
    "exophytic_component": r"exophytic",
    "midline_crossing": r"cross\w* (?:the )?midline|across (?:the )?midline|crossing of the midline|midline[- ]crossing",
}
TRI_STATES = ["not_mentioned", "present", "absent"]

LOCATION_TERMS = {
    "thalamus": r"thalam\w*|pulvinar",
    "midbrain": r"midbrain|mesencephal\w*|cerebral peduncle\w*|hemi-?midbrain|tegmen\w*|tectal|tectum",
    "pons": r"\bpons\b|pontine|pontomedull\w*|ponto-medull\w*|pontomesenceph\w*",
    "medulla": r"medulla\w*|bulb\w*",
    "cerebellum": r"cerebell\w*|vermi\w*",
    "cerebral_hemisphere": r"frontal|parietal|temporal|occipital|insula\w*|hemispher\w*|corona radiata|centrum semiovale",
    "basal_ganglia": r"basal ganglia|caudate|lentiform|putamen|globus pallidus|internal capsule",
    "corpus_callosum": r"corpus callosum|callosal|splenium|genu",
    "hypothalamus_suprasellar": r"hypothalam\w*|suprasellar|chiasm\w*|optic",
    "fourth_ventricle": r"fourth ventric\w*|4th ventric\w*",
    "brainstem_nos": r"brain-?stem",
}

MOLECULAR_TERMS = {
    "H3_K27": r"h3[\s\-]*(?:\.\d\s*)?k27\w*|h3f3a|hist1h3b|h3 k27",
    "ACVR1": r"acvr1",
    "TP53": r"tp53",
    "PDGFRA": r"pdgfra",
    "BRAF": r"braf",
    "IDH": r"\bidh\d?\b",
    "MGMT": r"mgmt",
    "ATRX": r"atrx",
    "NF1": r"\bnf1\b",
    "EGFR": r"egfr",
    "MYCN": r"mycn",
    "FGFR1": r"fgfr1",
    "H3_G34": r"g34",
}

TUMOR_TYPES = ["diffuse_midline_glioma", "brainstem_glioma", "thalamic_glioma", "supratentorial_glioma", "other_unspecified"]
GRADE_GROUPS = ["high", "intermediate", "low", "unspecified"]
LATERALITIES = ["left", "right", "bilateral_or_midline", "unspecified"]

_NEG_CUES = re.compile(
    r"\b(no|not|without|absent|absence|negative|free of|lacking|lack|devoid|denies|neither|nor|none|"
    r"unremarkable|exclud\w*|rule[sd]? out)\b|\bnon[- ]|\bun-?enhanc",
    re.I,
)
_NEG_BREAK = re.compile(r"\b(but|however|although|though|with|showing|demonstrat\w*|there is|there are|and)\b|[;:]", re.I)


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[\.\n])\s+|;\s+", text) if s.strip()]


def _mentions(sentence, pattern):
    """Yield (is_negated) for each mention of pattern in sentence."""
    low = sentence.lower()
    for m in re.finditer(pattern, low):
        before = low[: m.start()]
        # the term itself can carry the negation ("non-enhancing", "non-hemorrhagic", "unenhanced")
        glued = re.search(r"\b(?:non|un)[- ]?$", before) is not None
        # negation cue within the clause that precedes the term (cut at contrast words like 'but', 'with')
        clause = _NEG_BREAK.split(before)[-1] if _NEG_BREAK.search(before) else before
        # keep 'and'/'or' lists inside the scope: "No definite cystic or necrotic areas"
        or_scope = re.split(r"[;,]", before)[-1]
        window = " ".join(or_scope.split()[-9:])
        neg = glued or bool(_NEG_CUES.search(" ".join(clause.split()[-6:]))) or bool(
            re.search(r"\b(no|without|absence of|absent|negative for|free of|lacking)\b(?:\s+\w+){0,6}\s*(?:,|or|and|/)?\s*$", window, re.I)
        )
        # trailing negation: "enhancement is absent", "hemorrhage not seen"
        after = low[m.end(): m.end() + 40]
        if re.match(r"\s*(?:is |are |was )?(?:absent|not (?:seen|identified|present|noted)|negative)", after):
            neg = True
        yield neg


def extract_tri_state(text):
    out = {}
    sents = _sentences(text)
    for name, pat in FINDING_TERMS.items():
        pos = neg = 0
        for s in sents:
            for is_neg in _mentions(s, pat):
                if is_neg:
                    neg += 1
                else:
                    pos += 1
        out[name] = "not_mentioned" if pos + neg == 0 else ("present" if pos >= neg and pos > 0 else "absent")
    return out


def extract_locations(text):
    sents = _sentences(text)
    locs = {}
    for name, pat in LOCATION_TERMS.items():
        hit = False
        for s in sents:
            if any(not n for n in _mentions(s, pat)):
                hit = True
                break
        locs[name] = int(hit)
    return locs


def extract_molecular(text):
    low = text.lower()
    return {k: int(bool(re.search(p, low))) for k, p in MOLECULAR_TERMS.items()}


def extract_tumor_type(type_text, impression_text=""):
    t = (type_text or "").lower()
    both = f"{t} {(impression_text or '').lower()}"
    if re.search(r"diffuse midline glioma|\bdmg\b|h3 ?k27", t):
        return "diffuse_midline_glioma"
    if re.search(r"brain-?stem|pontine|\bdipg\b|ponto|midbrain|medull", t):
        return "brainstem_glioma"
    if re.search(r"thalam|deep gr[ae]y", t):
        return "thalamic_glioma"
    if re.search(r"supratentorial|hemispher|cortical|lobe|frontal|parietal|temporal|occipital", t):
        return "supratentorial_glioma"
    if re.search(r"diffuse midline glioma", both):
        return "diffuse_midline_glioma"
    if re.search(r"brain-?stem|pontine|\bdipg\b", both):
        return "brainstem_glioma"
    if re.search(r"thalam", both):
        return "thalamic_glioma"
    if re.search(r"supratentorial", both):
        return "supratentorial_glioma"
    return "other_unspecified"


def extract_who_grades(grade_text):
    low = (grade_text or "").lower().replace("–", "-").replace("—", "-")
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    nums = []
    for m in re.finditer(r"grade\s*(?:of\s*)?(iv|iii|ii|i|[1-4])(?:\s*(?:-|to|/|or)\s*(?:grade\s*)?(iv|iii|ii|i|[1-4]))?\b", low):
        for g in m.groups():
            if g:
                nums.append(roman.get(g, int(g) if g.isdigit() else 0))
    return sorted(set(n for n in nums if 1 <= n <= 4))


def extract_grade_group(grade_text):
    nums = extract_who_grades(grade_text)
    low = (grade_text or "").lower()
    if nums:
        if min(nums) >= 3:
            return "high"
        if max(nums) <= 2:
            return "low"
        return "intermediate"
    if re.search(r"high[- ]grade|malignant|aggressive", low) and not re.search(r"low[- ]grade", low):
        return "high"
    if re.search(r"low[- ]grade|benign", low) and not re.search(r"high[- ]grade", low):
        return "low"
    if re.search(r"intermediate", low) or (re.search(r"low", low) and re.search(r"high", low)):
        return "intermediate"
    return "unspecified"


def extract_laterality(text):
    low = text.lower()
    left = len(re.findall(r"\bleft\b", low))
    right = len(re.findall(r"\bright\b", low))
    bilat = len(re.findall(r"bilateral|both|symmetric|midline|across the midline", low))
    if left > right and left >= bilat:
        return "left"
    if right > left and right >= bilat:
        return "right"
    if left or right or bilat:
        return "bilateral_or_midline"
    return "unspecified"


def extract_attributes(sections):
    """sections: dict with the canonical section keys. Returns the evaluation attributes."""
    findings_text = " ".join(sections.get(k, "") for k in ("findings", "characterization", "impression", "summary"))
    location_text = " ".join(sections.get(k, "") for k in ("findings", "impression", "summary"))
    return {
        "tumor_type": extract_tumor_type(sections.get("type", ""), sections.get("impression", "")),
        "grade_group": extract_grade_group(sections.get("grade", "")),
        "who_grades": extract_who_grades(sections.get("grade", "")),
        "laterality": extract_laterality(location_text),
        "findings": extract_tri_state(findings_text),
        "locations": extract_locations(location_text),
        "molecular_markers": extract_molecular(sections.get("any_other", "") + " " + sections.get("grade", "")),
    }


# Fields with enough support (>= ~20 reports) to be used as auxiliary targets / headline metrics.
AUX_FINDINGS = [
    "enhancement", "necrosis", "cystic_change", "hemorrhage", "edema",
    "mass_effect", "midline_shift", "hydrocephalus", "exophytic_component",
]
AUX_LOCATIONS = [
    "thalamus", "midbrain", "pons", "medulla", "cerebellum",
    "cerebral_hemisphere", "fourth_ventricle", "brainstem_nos",
]
KEY_MOLECULAR = ["H3_K27", "TP53", "IDH", "ATRX", "MGMT"]
