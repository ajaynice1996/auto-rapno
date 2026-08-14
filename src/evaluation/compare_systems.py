from __future__ import annotations

from pathlib import Path


def write_phase2_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("case_id,agentic_status,multimodal_status,radiologist_reference\n", encoding="utf-8")
