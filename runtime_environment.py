"""Mandatory runtime guard for SPARROW model and experiment calculations."""

from __future__ import annotations

import os
from pathlib import Path
import sys


EXPECTED_ENV_NAME = "sparrow"
EXPECTED_PREFIX = Path(
    r"D:\ProgramData\anaconda3\envs\sparrow"
)


def _normalized(path: str | Path) -> str:
    return str(Path(path).resolve()).rstrip("\\/").casefold()


def assert_sparrow_runtime() -> dict[str, str]:
    """Fail before numerical imports unless Conda activated the sparrow env."""
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    conda_prefix_text = os.environ.get("CONDA_PREFIX", "")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    problems = []
    if _normalized(prefix) != _normalized(EXPECTED_PREFIX):
        problems.append(f"sys.prefix={prefix}")
    if _normalized(executable.parent) != _normalized(EXPECTED_PREFIX):
        problems.append(f"sys.executable={executable}")
    if not conda_prefix_text:
        problems.append("CONDA_PREFIX is missing (environment not activated)")
    elif _normalized(conda_prefix_text) != _normalized(EXPECTED_PREFIX):
        problems.append(f"CONDA_PREFIX={conda_prefix_text}")
    if conda_name.casefold() != EXPECTED_ENV_NAME:
        problems.append(f"CONDA_DEFAULT_ENV={conda_name!r}")
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "SPARROW runtime purity violation. Use "
            "'conda --no-plugins run -n sparrow python <script>'. "
            f"Details: {details}"
        )
    return {
        "environment_name": EXPECTED_ENV_NAME,
        "expected_prefix": str(EXPECTED_PREFIX),
        "sys_prefix": str(prefix),
        "sys_executable": str(executable),
        "conda_prefix": conda_prefix_text,
        "conda_default_env": conda_name,
    }

