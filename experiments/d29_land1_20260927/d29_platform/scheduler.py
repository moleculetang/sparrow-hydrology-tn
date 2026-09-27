"""Dry-run scheduling, resource hysteresis, and recoverable per-path logs.

This module has no subprocess launcher and cannot dispatch formal fitting.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import time


def resource_gate(metrics: dict, *, paused: bool = False, reserve_bytes: int = 0) -> dict:
    names = ("cpu_percent", "physical_percent", "commit_percent")
    if any(name not in metrics or not math.isfinite(float(metrics[name])) for name in names):
        return {"dispatch_allowed": False, "paused": True, "reason": "resource_measurement_unavailable"}
    values = [float(metrics[name]) for name in names]
    if any(v < 0 or v > 100 for v in values):
        return {"dispatch_allowed": False, "paused": True, "reason": "invalid_resource_measurement"}
    if reserve_bytes < 0:
        raise ValueError("negative memory reserve")
    if reserve_bytes and (metrics.get("physical_available_bytes", -1) < reserve_bytes or
                          metrics.get("commit_available_bytes", -1) < reserve_bytes):
        return {"dispatch_allowed": False, "paused": True, "reason": "peak_memory_reserve"}
    if max(values) >= 90:
        return {"dispatch_allowed": False, "paused": True, "reason": "resource_at_or_above_90"}
    if paused and max(values) >= 85:
        return {"dispatch_allowed": False, "paused": True, "reason": "waiting_for_all_resources_below_85"}
    return {"dispatch_allowed": True, "paused": False, "reason": "resource_gate_clear"}


@contextmanager
def _path_lock(path: Path, timeout: float = 30.0):
    """Lock is local to one path. Stale locks are reported, never auto-deleted."""
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError) as exc:
            # Windows can report EACCES for a just-unlinked/delete-pending
            # O_EXCL lock during concurrent handoff. Bounded retry is safe;
            # persistent ACL/ownership failures still terminate explicitly.
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"checkpoint lock unavailable; inspect owner/permissions: {path}") from exc
            time.sleep(0.01)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode())
        os.close(fd)
        yield
    finally:
        path.unlink()


class PathLedger:
    """Serialized append and checkpoint; restores cumulative calls and activity."""
    def __init__(self, directory: str | Path, path_id: str):
        if not path_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in path_id):
            raise ValueError("unsafe path identity")
        self.directory = Path(directory) / path_id
        self.log = self.directory / "events.jsonl"
        self.checkpoint = self.directory / "checkpoint.json"
        self.lock = self.directory / "ledger.lock"

    def _read_locked(self):
        if not self.checkpoint.exists():
            state = {"sequence": 0, "calls": 0, "active_seconds": 0.0, "status": "registered"}
        else:
            state = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        if self.log.exists():
            # A flushed event may precede an interrupted checkpoint replace.
            lines = self.log.read_text(encoding="utf-8").splitlines()
            for line in lines:
                event = json.loads(line)
                if event["sequence"] > state["sequence"]:
                    state = event
        return state

    def restore(self):
        with _path_lock(self.lock):
            return self._read_locked()

    def append(self, *, event: str, calls_delta: int = 0, active_seconds_delta: float = 0.0,
               status: str = "dry_run", details=None):
        if calls_delta < 0 or active_seconds_delta < 0 or not math.isfinite(active_seconds_delta):
            raise ValueError("cumulative budgets cannot decrease")
        with _path_lock(self.lock):
            previous = self._read_locked()
            state = {"sequence": previous["sequence"] + 1,
                     "calls": previous["calls"] + calls_delta,
                     "active_seconds": previous["active_seconds"] + active_seconds_delta,
                     "event": event, "status": status, "details": details or {},
                     "unix_time": time.time()}
            with self.log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary = self.checkpoint.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.checkpoint)
            return state


def dry_run_path(path_config: dict, gate: dict, resource_decision: dict) -> dict:
    return {"path_id": path_config["path_id"], "configuration": path_config["configuration"],
            "input_gate": gate, "resource_gate": resource_decision,
            "status": "blocked" if not gate["allowed"] else "registered_not_dispatched",
            "dispatch": False, "fit_calls": 0,
            "performance_prerequisites": [], "science_start_count": 0}
