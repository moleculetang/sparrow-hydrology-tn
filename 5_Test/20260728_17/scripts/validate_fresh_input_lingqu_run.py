from __future__ import annotations

import json
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports"


def main() -> None:
    status = json.loads((OUT / "experiment" / "experiment_status.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "latest_input_lingqu_exclusion_gate.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((OUT / "fresh_input_lingqu_diagnostic.json").read_text(encoding="utf-8"))
    assert status["status"] == "passed" and status["completed_steps"] == 7
    assert gate["passed"]
    assert all(v == 0 for v in gate["lingqu_san_rows_in_model_panel"].values())
    assert diagnostic["common_station_strict_summary"]["common_station_count"] > 0
    print(json.dumps({"status": "PASS", "strict_station_count": diagnostic["all_station_strict_summary"]["fresh"]["station_count"], "common_station_count": diagnostic["common_station_strict_summary"]["common_station_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
