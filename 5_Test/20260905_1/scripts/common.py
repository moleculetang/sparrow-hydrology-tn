"""Runtime, artifact and memory helpers used only by the 20260905 program."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from runtime_environment import assert_sparrow_runtime

RUNTIME = assert_sparrow_runtime()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_replace(part, path):
    """Retry transient Windows sharing/access locks; keep old target intact.

    Bounded retries do not change permissions or fall back to non-atomic writes.
    A persistent permission failure still propagates to the experiment runner.
    """
    for attempt in range(10):
        try:
            os.replace(part, path)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 9:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=json_default, allow_nan=False) + '\n', encoding='utf-8')
    atomic_replace(part, path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    frame.to_parquet(part, index=False)
    atomic_replace(part, path)


def memory_info():
    """Native Windows memory counters; no optional package or process enumeration."""
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('ullTotalPhys', 'ullAvailPhys',
                'ullTotalPageFile', 'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual',
                'ullAvailExtendedVirtual')]
    pmc, ms = PMC(), MS()
    pmc.cb, ms.dwLength = ctypes.sizeof(pmc), ctypes.sizeof(ms)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError()
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(ms)):
        raise ctypes.WinError()
    return {'pid': os.getpid(), 'rss_gib': pmc.WorkingSetSize / 2**30,
            'peak_rss_gib': pmc.PeakWorkingSetSize / 2**30,
            'system_total_gib': ms.ullTotalPhys / 2**30,
            'system_available_gib': ms.ullAvailPhys / 2**30}


def memory_guard(limit_gib=15):
    result = memory_info()
    if result['rss_gib'] > limit_gib:
        raise MemoryError(f'Worker RSS exceeds registered {limit_gib} GiB: {result}')
    return result
