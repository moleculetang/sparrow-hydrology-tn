from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path(r"D:\ProgramData\anaconda3\envs\sparrow\python.exe")


def main() -> None:
    validation = json.loads((RUN_DIR / "validation.json").read_text(encoding="utf-8"))
    gate = json.loads((RUN_DIR / "gate.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(RUN_DIR / "inputs" / "derived" / "chm_pre_v2_daily_by_reach_2006_2018.parquet")
    mass = pd.read_csv(RUN_DIR / "reports" / "daily_monthly_mass_audit.csv")
    area = pd.read_csv(RUN_DIR / "reports" / "area_closure_audit.csv")
    checks = {
        "runtime_exact": str(Path(sys.executable).resolve()).casefold() == str(EXPECTED_PYTHON.resolve()).casefold(),
        "validation_passed": validation["passed"] is True,
        "all_registered_checks_passed": all(validation["checks"].values()),
        "gate_passed": gate["passed"] is True,
        "daily_shape_exact": tuple(daily.shape) == (1_092_040, 5),
        "daily_key_unique": bool(not daily.duplicated(["reach_id", "date"]).any()),
        "daily_period_frozen": str(daily["date"].min())[:10] == "2006-01-01" and str(daily["date"].max())[:10] == "2018-12-31",
        "mass_all_passed": bool(mass["passed"].astype(bool).all()),
        "area_all_passed": bool(area["passed_1pct"].astype(bool).all()),
        "readme_updated": "GATE_0_PASS_READY_FOR_FORCING_SEMANTIC_AUDIT" in (RUN_DIR / "README.md").read_text(encoding="utf-8"),
    }
    result = {"checks": checks, "passed": bool(all(checks.values()))}
    (RUN_DIR / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
