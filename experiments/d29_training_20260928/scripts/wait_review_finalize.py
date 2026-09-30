"""Bounded local companion: finalize the frozen review queue when it ends.

This process is not a scheduler. It performs no fitting and never overrides a
failed queue or acceptance gate. Its sole purpose is to preserve the requested
post-training audit if the interactive terminal disconnects.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import write_json


def process_alive(pid: int) -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise OSError(error, "Cannot inspect review queue")
    try:
        status = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "Cannot inspect review queue")
        return status.value == 259
    finally:
        kernel.CloseHandle(handle)


def main() -> None:
    out = ROOT / "outputs"
    identity = json.loads((out / "review_queue_identity.json").read_text(encoding="utf-8"))
    clock = json.loads((ROOT / "config/clock.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(clock["deadline"])
    write_json(out / "review_finalizer_identity.json", {
        "pid": os.getpid(), "queue_pid": identity["pid"],
        "started": datetime.now(deadline.tzinfo).isoformat(),
        "bounded_by": deadline.isoformat(), "not_scheduled": True,
    })
    stop_path = out / "review_queue_stop.json"
    while not stop_path.exists():
        if datetime.now(deadline.tzinfo) >= deadline:
            raise RuntimeError("FINAL_DEADLINE_WITHOUT_QUEUE_STOP")
        if not process_alive(identity["pid"]):
            raise RuntimeError("QUEUE_EXITED_WITHOUT_STOP_RECEIPT")
        time.sleep(30)
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    if stop["reason"] not in ("finite_queue_exhausted", "dispatch_deadline"):
        raise RuntimeError("COMMON_QUEUE_FAILURE_REQUIRES_REVIEW: " + stop["reason"])
    with (out / "review_finalizer_console.log").open("ab") as log:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/finalize_completed_campaign.py")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("FINALIZATION_FAILED: " + str(result.returncode))
    write_json(out / "review_finalizer_completed.json", {
        "completed": datetime.now(deadline.tzinfo).isoformat(),
        "queue_reason": stop["reason"], "finalization_exit_code": result.returncode,
        "expert_interpretation_review_pending": True,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(ROOT / "outputs/review_finalizer_error.json", {
            "error": repr(exc), "checkpoints_preserved": True,
            "no_silent_exclusion": True,
        })
        raise
