from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
PYTHON = Path(RUNTIME["sys_executable"])
SCENARIOS = ["R3_00", "R3_10", "R3_01", "R3_11"]


def execute(scenario: str) -> dict[str, object]:
    completed_oof = RUN / "outputs" / scenario / "q72_three_fold_oof_predictions.parquet"
    if completed_oof.exists():
        return {"scenario": scenario, "returncode": 0, "status": "existing_complete_oof_reused"}
    env = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = "1"
    command = [str(PYTHON), str(RUN / "scripts" / "run_r3_scenario.py"), scenario]
    completed = subprocess.run(command, cwd=RUN, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (RUN / "logs" / f"{scenario}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (RUN / "logs" / f"{scenario}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return {"scenario": scenario, "returncode": completed.returncode, "status": "executed"}


def main() -> None:
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute, scenario): scenario for scenario in SCENARIOS}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: SCENARIOS.index(str(row["scenario"])))
    payload = {"runtime": RUNTIME, "completed_utc": datetime.now(timezone.utc).isoformat(), "runs": rows}
    payload["passed"] = all(int(row["returncode"]) == 0 for row in rows)
    (RUN / "logs" / "scenario_run_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["passed"]:
        raise RuntimeError(f"One or more scenario runs failed: {rows}")


if __name__ == "__main__":
    main()
