"""Read .docx reports, normalise section headings and keep the original text.

The reports follow a loose 8-section template (Patient ID, Findings, Impression, Type,
Characterization, Grade, Summary, Any other) but heading style varies (numbered / unnumbered,
heading and body in separate paragraphs, everything in one paragraph, extra trailing
paragraphs). This module maps all of those variants onto one canonical dict.
"""
import re
import zipfile

SECTIONS = ["findings", "impression", "type", "characterization", "grade", "summary", "any_other"]

_HEADING_ALIASES = [
    ("patient_id", r"patient\s*id"),
    ("findings", r"findings?"),
    ("impression", r"impressions?"),
    ("type", r"type(?:ison)?"),
    ("characterization", r"characteri[sz]ation"),
    ("grade", r"grade"),
    ("summary", r"summary"),
    ("any_other", r"any\s*other[^:\n]{0,70}"),
]
_HEAD_NAMES = "|".join(f"(?P<{k}>{v})" for k, v in _HEADING_ALIASES)

# Heading at the start of a paragraph: optional numbering, name, then ':' or end of paragraph.
_PARA_HEAD = re.compile(
    rf"^[^\w]*(?:\d+\s*[\.\)]\s*)?\**\s*(?:{_HEAD_NAMES})\s*\**\s*(?::\s*(?P<rest>.*)|$)", re.I | re.S
)
# Numbered heading with a colon appearing inline (single-paragraph reports).
_INLINE_HEAD = re.compile(
    r"(?=[1-8]\s*\.\s*(?:findings?|impressions?|type|characteri[sz]ation|grade|summary|any\s*other)[^:\n]{0,70}:)",
    re.I,
)


def read_docx_paragraphs(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S))
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paras.append(text)
    return paras


def split_sections(paragraphs):
    """Return (sections dict, parse_warnings)."""
    expanded = []
    for p in paragraphs:
        expanded.extend(x.strip() for x in _INLINE_HEAD.split(p) if x.strip())

    out = {k: [] for k in ["patient_id"] + SECTIONS}
    current = None
    for p in expanded:
        m = _PARA_HEAD.match(p)
        if m:
            current = next((k for k in out if m.group(k)), None)
            rest = (m.group("rest") or "").strip()
            if current and rest:
                out[current].append(rest)
        elif current:
            out[current].append(p)
    sections = {k: " ".join(v).strip() for k, v in out.items()}
    warnings = [f"missing_section:{k}" for k in SECTIONS if not sections[k]]
    return sections, warnings


def parse_report_file(path):
    paragraphs = read_docx_paragraphs(path)
    sections, warnings = split_sections(paragraphs)
    return {
        "source_file": str(path),
        "raw_text": "\n".join(paragraphs),
        "sections": {k: sections[k] for k in SECTIONS},
        "patient_id_in_report": sections["patient_id"],
        "parse_warnings": warnings,
    }
