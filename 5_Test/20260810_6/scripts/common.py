from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUN = Path(__file__).resolve().parents[1]
CONFIG_PATH = RUN / "config" / "model.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_utf8_csv(path: Path, **kwargs: Any):
    import pandas as pd

    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
