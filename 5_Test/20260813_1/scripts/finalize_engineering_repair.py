from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    artifacts = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        if path.name == "input_manifest.json":
            continue
        artifacts.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    matrix_rows = []
    for scenario in ["B0", "B1"]:
        for folder in sorted((ROOT / "outputs" / scenario / "blocked_folds").iterdir()):
            matrix_dir = folder / "reports" / "design_matrix"
            for name in ["observation_design_matrix.npz", "augmented_map_design_matrix.npz"]:
                path = matrix_dir / name
                try:
                    matrix = sparse.load_npz(path)
                    row = {
                    "scenario": scenario,
                    "fold_id": folder.name,
                    "artifact": name,
                    "rows": matrix.shape[0],
                    "columns": matrix.shape[1],
                    "nnz": matrix.nnz,
                    "readable": True,
                    "error": "",
                    }
                except Exception as exc:
                    row = {
                        "scenario": scenario,
                        "fold_id": folder.name,
                        "artifact": name,
                        "rows": None,
                        "columns": None,
                        "nnz": None,
                        "readable": False,
                        "error": repr(exc),
                    }
                matrix_rows.append(row)
    pd.DataFrame(matrix_rows).to_csv(
        ROOT / "reports" / "design_matrix_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failed = [row for row in matrix_rows if not row["readable"]]
    if failed:
        raise RuntimeError(f"Unreadable design matrices: {failed}")
    manifest = {
        "runtime": {
            "sys_prefix": sys.prefix,
            "sys_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "frozen_parent": str(Path(r"E:\SPARROW\5_Test\20260812_5")),
        "artifacts": artifacts,
    }
    (ROOT / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
