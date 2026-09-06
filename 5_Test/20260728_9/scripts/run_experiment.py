from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
LOGS = RUN / "logs"
REPORTS = RUN / "reports" / "experiment"
HEARTBEATS = LOGS / "heartbeats"
HEARTBEAT_SECONDS = 30

STEPS = [
    ("runtime_controller_preflight", "test_runtime_controller.py"),
    ("input_manifest", "build_input_manifest.py"),
    (
        "minimal_monotone_residual_pilot",
        "run_minimal_monotone_residual_pilot.py",
    ),
    (
        "minimal_monotone_residual_validation",
        "validate_minimal_monotone_residual_pilot.py",
    ),
]

TIMEOUT_SECONDS = {
    "minimal_monotone_residual_pilot": 300,
}
DEFAULT_TIMEOUT_SECONDS = 300


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Some managed Windows sessions allow taskkill to return without
        # terminating the direct process.  Always enforce termination of the
        # Popen child as a second boundary.
        if proc.poll() is None:
            proc.kill()
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def stream_pipe(
    source: IO[str],
    destination: IO[str],
    last_output: dict[str, float],
    lock: threading.Lock,
) -> None:
    try:
        for line in iter(source.readline, ""):
            destination.write(line)
            destination.flush()
            with lock:
                last_output["monotonic"] = time.monotonic()
    finally:
        source.close()


def status_payload(
    rows: list[dict[str, object]],
    *,
    status: str,
    current_step: str | None,
    child_pid: int | None,
    started_at: str,
    message: str = "",
) -> dict[str, object]:
    return {
        "run_id": RUN.name,
        "phase_id": "bounded_time_gate_sharpening_iteration",
        "reference_run": "20260727_6",
        "status": status,
        "started_at": started_at,
        "last_heartbeat": iso_now(),
        "current_step": current_step,
        "child_pid": child_pid,
        "completed_steps": len(rows),
        "expected_steps": len(STEPS),
        "message": message,
        "steps": rows,
    }


def write_status(
    rows: list[dict[str, object]],
    *,
    status: str,
    current_step: str | None,
    child_pid: int | None,
    started_at: str,
    message: str = "",
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        REPORTS / "experiment_step_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    atomic_json(
        REPORTS / "experiment_status.json",
        status_payload(
            rows,
            status=status,
            current_step=current_step,
            child_pid=child_pid,
            started_at=started_at,
            message=message,
        ),
    )


def append_heartbeat(
    label: str,
    proc: subprocess.Popen[str],
    elapsed_seconds: float,
    seconds_since_output: float,
    timeout_seconds: int,
) -> None:
    HEARTBEATS.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": RUN.name,
        "step": label,
        "timestamp": iso_now(),
        "child_pid": proc.pid,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_since_output": round(seconds_since_output, 3),
        "timeout_seconds": timeout_seconds,
        "stalled_output_warning": bool(seconds_since_output >= 600),
    }
    with (HEARTBEATS / f"{label}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_step(
    label: str,
    script_name: str,
    rows: list[dict[str, object]],
    run_started_at: str,
) -> dict[str, object]:
    started_wall = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    timeout_seconds = TIMEOUT_SECONDS.get(label, DEFAULT_TIMEOUT_SECONDS)
    step_dir = LOGS / "experiment_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / f"{label}.stdout.log"
    stderr_path = step_dir / f"{label}.stderr.log"
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )

    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_file, (
        stderr_path.open("w", encoding="utf-8", buffering=1)
    ) as stderr_file:
        proc = subprocess.Popen(
            [sys.executable, str(RUN / "scripts" / script_name)],
            cwd=str(RUN),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        write_status(
            rows,
            status="running",
            current_step=label,
            child_pid=proc.pid,
            started_at=run_started_at,
            message=f"started {label}",
        )
        last_output = {"monotonic": started_monotonic}
        lock = threading.Lock()
        threads = [
            threading.Thread(
                target=stream_pipe,
                args=(proc.stdout, stdout_file, last_output, lock),
                daemon=True,
            ),
            threading.Thread(
                target=stream_pipe,
                args=(proc.stderr, stderr_file, last_output, lock),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        next_heartbeat = started_monotonic
        timed_out = False
        while proc.poll() is None:
            now = time.monotonic()
            elapsed = now - started_monotonic
            with lock:
                seconds_since_output = now - last_output["monotonic"]
            if now >= next_heartbeat:
                append_heartbeat(
                    label,
                    proc,
                    elapsed,
                    seconds_since_output,
                    timeout_seconds,
                )
                write_status(
                    rows,
                    status="running",
                    current_step=label,
                    child_pid=proc.pid,
                    started_at=run_started_at,
                    message=(
                        f"{label} elapsed={elapsed:.1f}s "
                        f"silence={seconds_since_output:.1f}s"
                    ),
                )
                print(
                    f"HEARTBEAT {label}: elapsed={elapsed:.1f}s "
                    f"silence={seconds_since_output:.1f}s pid={proc.pid}",
                    flush=True,
                )
                next_heartbeat = now + HEARTBEAT_SECONDS
            if elapsed > timeout_seconds:
                timed_out = True
                terminate_process_tree(proc)
                break
            time.sleep(1)

        for thread in threads:
            thread.join(timeout=5)
        returncode = int(proc.returncode if proc.returncode is not None else -9)

    ended_wall = datetime.now().astimezone()
    return {
        "step": label,
        "script": script_name,
        "returncode": returncode,
        "status": "timed_out" if timed_out else (
            "passed" if returncode == 0 else "failed"
        ),
        "started_at": started_wall.isoformat(timespec="seconds"),
        "ended_at": ended_wall.isoformat(timespec="seconds"),
        "elapsed_seconds": round(
            (ended_wall - started_wall).total_seconds(), 3
        ),
        "timeout_seconds": timeout_seconds,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def write_markdown_log(rows: list[dict[str, object]], final_status: str) -> None:
    lines = [
        f"# {RUN.name} Experiment Log",
        "",
        f"Status: {final_status}",
        "",
    ]
    lines.extend(
        f"- {row['step']}: status={row['status']}, "
        f"returncode={row['returncode']}, "
        f"elapsed={row['elapsed_seconds']}s"
        for row in rows
    )
    (LOGS / "run_experiment_log.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_started_at = iso_now()
    rows: list[dict[str, object]] = []
    write_status(
        rows,
        status="running",
        current_step=None,
        child_pid=None,
        started_at=run_started_at,
        message="experiment initialized",
    )
    for label, script in STEPS:
        row = run_step(label, script, rows, run_started_at)
        rows.append(row)
        if row["status"] != "passed":
            final_status = str(row["status"])
            write_status(
                rows,
                status=final_status,
                current_step=None,
                child_pid=None,
                started_at=run_started_at,
                message=f"stopped after {label}",
            )
            write_markdown_log(rows, final_status)
            print(
                f"STOPPED {label}: status={final_status}; "
                f"see {row['stderr_log']}",
                flush=True,
            )
            return returncode_for(row)

    write_status(
        rows,
        status="passed",
        current_step=None,
        child_pid=None,
        started_at=run_started_at,
        message="all experiment steps passed",
    )
    write_markdown_log(rows, "passed")
    print((LOGS / "run_experiment_log.md").read_text(encoding="utf-8"))
    return 0


def returncode_for(row: dict[str, object]) -> int:
    if row["status"] == "timed_out":
        return 124
    value = int(row["returncode"])
    return value if value != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
