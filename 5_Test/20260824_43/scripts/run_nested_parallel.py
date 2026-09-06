"""Run four memory-bounded Stage43 nested workers with resumable shards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_43"
SCRIPT = RUN / "scripts/run_nested_worker.py"
LOGS = RUN / "logs"
SHARDS = RUN / "outputs/nested_shards"
PYTHON = Path(r"D:\ProgramData\anaconda3\envs\sparrow\python.exe")
HARD_GIB = 12.0


def completed_count(candidate: str, shard: int) -> int:
    path = SHARDS / f"{candidate}_s{shard:02d}of02_parameters.parquet"
    return len(pd.read_parquet(path)) if path.exists() else 0


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_rss_gib(process: subprocess.Popen[str]) -> float:
    if process.poll() is not None:
        return 0.0
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, process.pid)
    if not handle:
        return 0.0
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return float(counters.WorkingSetSize / 1024**3) if ok else 0.0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int, subprocess.Popen[str], object]] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for candidate in ("kfast", "active"):
        for shard in (0, 1):
            log_path = LOGS / f"nested_{candidate}_s{shard:02d}.log"
            log = log_path.open("a", encoding="utf-8")
            command = [str(PYTHON), str(SCRIPT), "--candidate", candidate, "--shard-index", str(shard), "--shards", "2"]
            process = subprocess.Popen(
                command, cwd=str(RUN), stdout=log, stderr=subprocess.STDOUT,
                text=True, creationflags=creationflags,
            )
            jobs.append((candidate, shard, process, log))
    started = time.perf_counter()
    try:
        while True:
            running = [job for job in jobs if job[2].poll() is None]
            total_rss = sum(process_rss_gib(job[2]) for job in running)
            status = {
                "elapsed_seconds": time.perf_counter() - started,
                "running": len(running), "total_worker_tree_rss_gib": total_rss,
                "completed_folds": {f"{candidate}_s{shard}": completed_count(candidate, shard) for candidate, shard, _, _ in jobs},
            }
            print(json.dumps(status), flush=True)
            if total_rss > HARD_GIB:
                for _, _, process, _ in running:
                    process.terminate()
                raise MemoryError(f"nested workers exceeded {HARD_GIB} GiB")
            failures = [(candidate, shard, process.returncode) for candidate, shard, process, _ in jobs if process.poll() not in (None, 0)]
            if failures:
                for _, _, process, _ in running:
                    process.terminate()
                raise RuntimeError(f"nested worker failure: {failures}")
            if not running:
                break
            time.sleep(60)
    finally:
        for _, _, _, log in jobs:
            log.close()
    print(json.dumps({"status": "PASS_ALL_NESTED_WORKERS", "elapsed_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
