from __future__ import annotations

from pathlib import Path

from src.common.io_utils import ensure_dir


def run_registration_placeholder(case_id: str, output_root: Path) -> Path:
    case_dir = output_root / case_id
    ensure_dir(case_dir)
    transform_file = case_dir / "t1_to_flair.mat"
    transform_file.write_text("# placeholder transform\n", encoding="utf-8")
    return transform_file
