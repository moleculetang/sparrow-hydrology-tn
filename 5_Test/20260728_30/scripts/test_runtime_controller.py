from __future__ import annotations

import json
from pathlib import Path

import run_experiment as runner


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "preflight"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    original_timeout = runner.TIMEOUT_SECONDS.get("runtime_timeout_probe")
    original_heartbeat = runner.HEARTBEAT_SECONDS
    original_write_status = runner.write_status
    runner.TIMEOUT_SECONDS["runtime_timeout_probe"] = 2
    runner.HEARTBEAT_SECONDS = 1
    runner.write_status = lambda *_args, **_kwargs: None
    try:
        row = runner.run_step(
            "runtime_timeout_probe",
            "runtime_timeout_probe.py",
            [],
            runner.iso_now(),
        )
    finally:
        if original_timeout is None:
            runner.TIMEOUT_SECONDS.pop("runtime_timeout_probe", None)
        else:
            runner.TIMEOUT_SECONDS["runtime_timeout_probe"] = original_timeout
        runner.HEARTBEAT_SECONDS = original_heartbeat
        runner.write_status = original_write_status

    heartbeat_path = (
        RUN / "logs" / "heartbeats" / "runtime_timeout_probe.jsonl"
    )
    heartbeat_rows = (
        [
            json.loads(line)
            for line in heartbeat_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if heartbeat_path.exists()
        else []
    )
    passed = bool(
        row["status"] == "timed_out"
        and int(row["returncode"]) != 0
        and 2 <= float(row["elapsed_seconds"]) <= 8
        and len(heartbeat_rows) >= 2
    )
    payload = {
        "run_id": RUN.name,
        "test": "runtime_controller_timeout_and_heartbeat",
        "passed": passed,
        "probe_result": row,
        "heartbeat_rows": len(heartbeat_rows),
    }
    (OUT / "runtime_controller_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
