"""Parse / validate generated report JSON against the report schema.

The raw generated text is always kept. If strict parsing fails we try (1) closing a truncated JSON
string and (2) key-by-key extraction; the result is flagged so downstream code knows how reliable it is.
"""
import json
import re

from .report_parser import SECTIONS

REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PediatricBrainTumorReport",
    "type": "object",
    "required": SECTIONS,
    "additionalProperties": False,
    "properties": {s: {"type": "string", "minLength": 1} for s in SECTIONS},
}


def validate_report_dict(d):
    """Return list of schema violations (empty list = valid)."""
    if not isinstance(d, dict):
        return ["not_an_object"]
    errs = []
    for s in SECTIONS:
        if s not in d:
            errs.append(f"missing:{s}")
        elif not isinstance(d[s], str):
            errs.append(f"not_string:{s}")
        elif not d[s].strip():
            errs.append(f"empty:{s}")
    errs += [f"unexpected:{k}" for k in d if k not in SECTIONS]
    return errs


def _extract_by_keys(text):
    """Pull '"key":"value"' pairs out of broken JSON in canonical key order."""
    out = {}
    positions = []
    for s in SECTIONS:
        m = re.search(rf'"{s}"\s*:\s*"', text)
        if m:
            positions.append((m.start(), m.end(), s))
    positions.sort()
    for i, (_, vstart, key) in enumerate(positions):
        vend = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        val = text[vstart:vend].rstrip()
        val = re.sub(r'"?\s*,?\s*}?\s*$', "", val) if i + 1 == len(positions) else re.sub(r'"\s*,\s*$', "", val)
        out[key] = val.replace('\\"', '"').replace("\\n", " ").strip()
    return out


def parse_generated(text):
    """Return dict(sections, status, errors, raw_text). status: valid | repaired | partial | invalid."""
    raw = text
    errors = []
    try:
        d = json.loads(text)
        errors = validate_report_dict(d)
        if not errors:
            return {"sections": {s: d[s] for s in SECTIONS}, "status": "valid", "errors": [], "raw_text": raw}
    except json.JSONDecodeError as e:
        errors = [f"json_error:{e.msg}"]
        for suffix in ('"}', '}', '"'):
            try:
                d = json.loads(text.rstrip() + suffix)
                if not validate_report_dict(d):
                    return {"sections": {s: d[s] for s in SECTIONS}, "status": "repaired", "errors": errors, "raw_text": raw}
            except json.JSONDecodeError:
                continue
    sections = _extract_by_keys(text)
    missing = [s for s in SECTIONS if not sections.get(s)]
    if sections and not missing:
        return {"sections": {s: sections[s] for s in SECTIONS}, "status": "repaired", "errors": errors, "raw_text": raw}
    if sections:
        return {"sections": {s: sections.get(s, "") for s in SECTIONS}, "status": "partial",
                "errors": errors + [f"missing:{m}" for m in missing], "raw_text": raw}
    return {"sections": None, "status": "invalid", "errors": errors, "raw_text": raw}
