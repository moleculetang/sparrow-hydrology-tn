from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_33"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"This program must run in the sparrow environment: {sys.executable}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name)
        if value not in (None, "1"):
            raise RuntimeError(f"{name} must be 1, got {value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ), encoding="utf-8")

