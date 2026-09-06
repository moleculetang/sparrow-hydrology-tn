from __future__ import annotations

import os
from pathlib import Path
import sys


EXPECTED = Path(r"D:\ProgramData\anaconda3\envs\sparrow")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _norm(value: str | Path) -> str:
    return str(Path(value).resolve()).rstrip("\\/").casefold()


def assert_sparrow_runtime() -> dict[str, object]:
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    threads = {name: os.environ.get(name, "") for name in THREAD_VARS}
    problems: list[str] = []
    if _norm(prefix) != _norm(EXPECTED):
        problems.append(f"sys.prefix={prefix}")
    if _norm(executable.parent) != _norm(EXPECTED):
        problems.append(f"sys.executable={executable}")
    if not conda_prefix or _norm(conda_prefix) != _norm(EXPECTED):
        problems.append(f"CONDA_PREFIX={conda_prefix!r}")
    if conda_name.casefold() != "sparrow":
        problems.append(f"CONDA_DEFAULT_ENV={conda_name!r}")
    if any(value != "1" for value in threads.values()):
        problems.append(f"thread_limits={threads}")
    if problems:
        raise RuntimeError(
            "Use conda --no-plugins run -n sparrow with OMP/MKL/OPENBLAS threads set to 1. "
            + "; ".join(problems)
        )
    return {
        "environment_name": "sparrow",
        "expected_prefix": str(EXPECTED),
        "sys_prefix": str(prefix),
        "sys_executable": str(executable),
        "conda_prefix": conda_prefix,
        "conda_default_env": conda_name,
        "thread_limits": threads,
    }
