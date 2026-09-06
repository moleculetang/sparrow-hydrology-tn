from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
LOGS = RUN / "logs"
REPORTS = RUN / "reports" / "experiment"

STEPS = [
    ("source_manifest", "build_input_manifest.py"),
    ("main_workflow", "run_main_workflow.py"),
    ("blocked_q72", "run_blocked_q72_screening.py"),
    ("bad_station_diagnosis", "build_bad_station_numeric_diagnosis.py"),
    (
        "stage2_nonlinear_recession_diagnostics",
        "build_stage2_nonlinear_recession_diagnostics.py",
    ),
    (
        "stage2_nonlinear_recession_gate",
        "compare_stage2_nonlinear_recession.py",
    ),
]


def run_step(label: str, script_name: str) -> dict[str, object]:
    started = datetime.now()
    proc = subprocess.run(
        [sys.executable, str(RUN / "scripts" / script_name)],
        cwd=str(RUN),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ended = datetime.now()
    step_dir = LOGS / "experiment_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"{label}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (step_dir / f"{label}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {
        "step": label,
        "script": script_name,
        "returncode": int(proc.returncode),
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "elapsed_seconds": round((ended - started).total_seconds(), 3),
        "stdout_log": str(step_dir / f"{label}.stdout.log"),
        "stderr_log": str(step_dir / f"{label}.stderr.log"),
    }


def write_status(rows: list[dict[str, object]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        REPORTS / "experiment_step_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_status = (
        "passed"
        if len(rows) == len(STEPS) and all(int(row["returncode"]) == 0 for row in rows)
        else "failed"
    )
    payload = {
        "run_id": RUN.name,
        "status": final_status,
        "completed_steps": len(rows),
        "expected_steps": len(STEPS),
        "steps": rows,
    }
    (REPORTS / "experiment_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {RUN.name} Experiment Log",
        "",
        f"Status: {final_status}",
        "",
    ]
    lines.extend(
        f"- {row['step']}: returncode={row['returncode']}, "
        f"elapsed={row['elapsed_seconds']}s"
        for row in rows
    )
    (LOGS / "run_experiment_log.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, script in STEPS:
        row = run_step(label, script)
        rows.append(row)
        write_status(rows)
        print(
            f"{label}: returncode={row['returncode']}, "
            f"elapsed={row['elapsed_seconds']}s",
            flush=True,
        )
        if int(row["returncode"]) != 0:
            print(f"FAILED {label}; see {row['stderr_log']}", flush=True)
            return int(row["returncode"])
    print((LOGS / "run_experiment_log.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
