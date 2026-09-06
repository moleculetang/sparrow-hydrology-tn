from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
STEPS = [
    ("backup_manifest", "build_backup_manifest.py"),
    ("correct_spatial", "build_corrected_spatial.py"),
    ("aggregate_chm_pre", "process_chm_pre_v2_corrected.py"),
    ("aggregate_cmfd", "process_cmfd_corrected.py"),
    ("corrected_climate_backbone", "build_corrected_climate_backbone.py"),
    ("build_updated_local_input", "build_updated_input.py"),
    ("verify_s0_reproduction", "verify_s0_reproduction.py"),
    ("run_s1_three_blocked_q72_folds", "run_q72_baseline.py"),
    ("build_model_audit", "build_model_audit.py"),
    ("compare_s0_s1", "compare_corrected_baseline.py"),
    ("write_readme", "write_readme.py"),
    ("validate_corrected_baseline", "validate_baseline.py"),
]


def main() -> None:
    logs = RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, script_name in STEPS:
        started = perf_counter()
        completed = subprocess.run(
            [sys.executable, str(RUN / "scripts" / script_name)], cwd=RUN,
            text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
        )
        elapsed = perf_counter() - started
        (logs / f"{step}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs / f"{step}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        rows.append({"step": step, "script": f"scripts/{script_name}", "returncode": completed.returncode, "elapsed_seconds": round(elapsed, 3)})
        if completed.returncode != 0:
            (logs / "run_all_manifest.json").write_text(json.dumps({"runtime": RUNTIME, "steps": rows, "completed": False}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"{step} failed; inspect logs/{step}.stderr.log")
    (logs / "run_all_manifest.json").write_text(json.dumps({"runtime": RUNTIME, "steps": rows, "completed": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runtime": RUNTIME, "steps": rows, "completed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
