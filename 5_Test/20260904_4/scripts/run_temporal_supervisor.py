"""Memory-safe three-worker supervisor for temporal multistart checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
LOGS = ROOT / "5_Test/20260904_4/logs"
WORK = ROOT / "5_Test/20260904_4/work"


def complete(capacity: str, fold: str, variant: int) -> bool:
    base = WORK / f"{capacity.lower()}_{fold.lower()}_start{variant}"
    return Path(str(base) + ".json").exists() and Path(str(base) + ".npz").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", nargs="+", choices=["H7", "H14", "H22"], required=True)
    parser.add_argument("--workers", type=int, default=1, choices=[1])
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(capacity, fold, variant) for capacity in args.capacities for fold in ["T1", "T2", "T3"] for variant in range(5)]
    jobs = [job for job in jobs if not complete(*job)]
    running: dict[subprocess.Popen, tuple[tuple[str, str, int], object, object]] = {}
    failures = []
    started = time.perf_counter()
    while jobs or running:
        while jobs and len(running) < args.workers:
            job = jobs.pop(0)
            capacity, fold, variant = job
            stem = f"{capacity.lower()}_{fold.lower()}_start{variant}"
            stdout = (LOGS / f"{stem}.stdout.log").open("w", encoding="utf-8")
            stderr = (LOGS / f"{stem}.stderr.log").open("w", encoding="utf-8")
            command = [sys.executable, str(HERE / "run_temporal_start.py"), "--capacity", capacity, "--fold", fold, "--variant", str(variant)]
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            running[process] = (job, stdout, stderr)
            print(json.dumps({"event": "started", "job": job, "pid": process.pid, "active": len(running), "remaining": len(jobs)}), flush=True)
        time.sleep(10)
        for process in list(running):
            code = process.poll()
            if code is None:
                continue
            job, stdout, stderr = running.pop(process)
            stdout.close(); stderr.close()
            ok = code == 0 and complete(*job)
            if not ok:
                failures.append({"job": job, "returncode": code})
            print(json.dumps({"event": "finished", "job": job, "returncode": code, "ok": ok, "active": len(running), "remaining": len(jobs)}), flush=True)
        if int(time.perf_counter() - started) % 60 < 10:
            print(json.dumps({"event": "heartbeat", "elapsed_seconds": time.perf_counter() - started, "active": len(running), "remaining": len(jobs), "failures": failures}), flush=True)
    if failures:
        raise RuntimeError(f"Temporal start failures: {failures}")
    print(json.dumps({"status": "PASS_TEMPORAL_MULTISTART_CHECKPOINTS", "capacities": args.capacities, "elapsed_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
