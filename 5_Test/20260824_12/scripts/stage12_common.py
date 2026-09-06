from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_12"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
EPS = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ),
        encoding="utf-8",
    )


def require_sparrow() -> None:
    import sys

    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")

