"""Launch four safe Stage-4 shard workers."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_4"
LOGS = RUN / "logs"
PYTHON = Path(r"D:\ProgramData\anaconda3\envs\sparrow\python.exe")
WORKER = RUN / "scripts/run_stage4_worker.py"

parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["screen", "refined_screen", "reference_screen", "full"], default="screen")
args = parser.parse_args(); LOGS.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy(); environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
processes = []
for shard in range(4):
    stdout = (LOGS / f"{args.phase}_s{shard:02d}.stdout.log").open("w", encoding="utf-8")
    stderr = (LOGS / f"{args.phase}_s{shard:02d}.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(WORKER), "--phase", args.phase, "--shard", str(shard), "--shards", "4"],
        cwd=str(ROOT), env=environment, stdout=stdout, stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    processes.append((process, stdout, stderr, shard)); print(json.dumps({"event": "started", "phase": args.phase, "shard": shard, "pid": process.pid}), flush=True)
failed = []
for process, stdout, stderr, shard in processes:
    code = process.wait(); stdout.close(); stderr.close()
    print(json.dumps({"event": "finished", "phase": args.phase, "shard": shard, "exit_code": code}), flush=True)
    if code != 0: failed.append(shard)
if failed: raise RuntimeError(f"Stage4 failed shards: {failed}")
print(json.dumps({"status": "ALL_STAGE4_SHARDS_COMPLETE", "phase": args.phase}), flush=True)
