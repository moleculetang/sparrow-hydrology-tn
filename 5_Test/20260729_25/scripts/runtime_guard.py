"""Experiment-local runtime purity gate.

This module deliberately lives inside the test folder so the experiment does
not depend on, or require edits to, SPARROW mainline files.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


EXPECTED_ENV_NAME = "sparrow"
EXPECTED_PREFIX = Path(r"D:\ProgramData\anaconda3\envs\sparrow")


def _normalized(path: str | Path) -> str:
    return str(Path(path).resolve()).rstrip("\\/").casefold()


def assert_sparrow_runtime() -> dict[str, str]:
    """Fail before numerical imports unless Conda activated ``sparrow``."""
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    problems: list[str] = []

    if _normalized(prefix) != _normalized(EXPECTED_PREFIX):
        problems.append(f"sys.prefix={prefix}")
    if _normalized(executable.parent) != _normalized(EXPECTED_PREFIX):
        problems.append(f"sys.executable={executable}")
    if not conda_prefix:
        problems.append("CONDA_PREFIX is missing")
    elif _normalized(conda_prefix) != _normalized(EXPECTED_PREFIX):
        problems.append(f"CONDA_PREFIX={conda_prefix}")
    if conda_name.casefold() != EXPECTED_ENV_NAME:
        problems.append(f"CONDA_DEFAULT_ENV={conda_name!r}")

    if problems:
        raise RuntimeError(
            "SPARROW experiment runtime purity violation. Use "
            "'conda --no-plugins run -n sparrow python <script>'. Details: "
            + "; ".join(problems)
        )

    return {
        "environment_name": EXPECTED_ENV_NAME,
        "expected_prefix": str(EXPECTED_PREFIX),
        "sys_prefix": str(prefix),
        "sys_executable": str(executable),
        "conda_prefix": conda_prefix,
        "conda_default_env": conda_name,
    }
