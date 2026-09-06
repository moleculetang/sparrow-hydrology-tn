"""Experiment-local Conda runtime purity gate."""

from __future__ import annotations

import os
from pathlib import Path
import sys


EXPECTED_PREFIX = Path(r"D:\ProgramData\anaconda3\envs\sparrow")


def _norm(value: str | Path) -> str:
    return str(Path(value).resolve()).rstrip("\\/").casefold()


def assert_sparrow_runtime() -> dict[str, str]:
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    problems: list[str] = []
    if _norm(prefix) != _norm(EXPECTED_PREFIX):
        problems.append(f"sys.prefix={prefix}")
    if _norm(executable.parent) != _norm(EXPECTED_PREFIX):
        problems.append(f"sys.executable={executable}")
    if not conda_prefix or _norm(conda_prefix) != _norm(EXPECTED_PREFIX):
        problems.append(f"CONDA_PREFIX={conda_prefix!r}")
    if conda_name.casefold() != "sparrow":
        problems.append(f"CONDA_DEFAULT_ENV={conda_name!r}")
    if problems:
        raise RuntimeError(
            "Use 'conda --no-plugins run -n sparrow python <script>'. "
            + "; ".join(problems)
        )
    return {
        "environment_name": "sparrow",
        "expected_prefix": str(EXPECTED_PREFIX),
        "sys_prefix": str(prefix),
        "sys_executable": str(executable),
        "conda_prefix": conda_prefix,
        "conda_default_env": conda_name,
    }
