from __future__ import annotations

from typing import Dict, List


def summarize_validation(validation_rows: List[Dict[str, object]]) -> Dict[str, int]:
    total = len(validation_rows)
    passed = sum(1 for r in validation_rows if str(r.get("validation_status", "")) == "PASS")
    failed = total - passed
    return {"num_cases": total, "num_validation_pass": passed, "num_validation_fail": failed}
