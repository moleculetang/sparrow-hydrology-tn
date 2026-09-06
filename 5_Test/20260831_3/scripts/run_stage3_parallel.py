"""Launch at most six independent objective/fold workers with resumable shards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_3"
WORK = RUN / "work"
LOGS = RUN / "logs"
WORKER = RUN / "scripts/run_stage3_worker.py"
PYTHON = Path(r"D:\ProgramData\anaconda3\envs\sparrow\python.exe")
OBJECTIVES = ["O0_LOG_T4", "O1_HET_T4", "O2_DYN_BALANCED"]
FOLDS = ["T1", "T2", "T3"]
# Six workers exceeded the registered 28-GiB aggregate RSS during the first
# live run.  Four is the audited safe concurrency; the contract value is an
# upper bound, not a target.
MAX_WORKERS = 4


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    pending = []
    for objective in OBJECTIVES:
        for fold in FOLDS:
            prefix = f"{objective.lower()}_{fold.lower()}"
            checkpoint = WORK / f"{prefix}_checkpoint.json"
            if checkpoint.exists() and json.loads(checkpoint.read_text(encoding="utf-8")).get("status") == "SHARD_COMPLETE":
                continue
            pending.append((objective, fold, prefix))
    active: list[tuple[subprocess.Popen, object, object, str, str, str]] = []
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    while pending or active:
        while pending and len(active) < MAX_WORKERS:
            objective, fold, prefix = pending.pop(0)
            stdout = (LOGS / f"{prefix}.stdout.log").open("w", encoding="utf-8")
            stderr = (LOGS / f"{prefix}.stderr.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [str(PYTHON), str(WORKER), "--objective", objective, "--fold", fold],
                cwd=str(ROOT), env=environment, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            active.append((process, stdout, stderr, objective, fold, prefix))
            print(json.dumps({"event": "started", "objective": objective, "fold": fold, "pid": process.pid}), flush=True)
        time.sleep(2.0)
        next_active = []
        for process, stdout, stderr, objective, fold, prefix in active:
            code = process.poll()
            if code is None:
                next_active.append((process, stdout, stderr, objective, fold, prefix)); continue
            stdout.close(); stderr.close()
            print(json.dumps({"event": "finished", "objective": objective, "fold": fold, "exit_code": code}), flush=True)
            if code != 0:
                raise RuntimeError(f"Stage3 worker failed: {objective} {fold}; see {LOGS / (prefix + '.stderr.log')}")
        active = next_active
    print(json.dumps({"status": "ALL_STAGE3_SHARDS_COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
