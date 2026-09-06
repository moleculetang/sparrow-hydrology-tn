from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "s0_reproduction"
BACKUP = RUN / "backup_before_correction" / "legacy_baseline" / "outputs" / "q72_three_fold_oof_predictions.parquet"
PARENT = RUN.parent / "20260805_2" / "outputs" / "q72_three_fold_oof_predictions.parquet"
KEYS = ["fold_id", "station_name", "reach_id", "year", "month"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists() or not PARENT.exists():
        raise FileNotFoundError("S0 backup or frozen parent OOF is missing")
    backup = pd.read_parquet(BACKUP).sort_values(KEYS).reset_index(drop=True)
    parent = pd.read_parquet(PARENT).sort_values(KEYS).reset_index(drop=True)
    same_columns = list(backup.columns) == list(parent.columns)
    same_keys = backup[KEYS].equals(parent[KEYS]) if len(backup) == len(parent) else False
    numeric_columns = sorted(set(backup.select_dtypes(include=[np.number]).columns) & set(parent.columns))
    max_numeric_difference = 0.0
    if same_keys and numeric_columns:
        left = backup[numeric_columns].to_numpy(float)
        right = parent[numeric_columns].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            max_numeric_difference = float(np.max(np.abs(left[finite] - right[finite])))
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            max_numeric_difference = float("inf")
    result = {
        "run_id": RUN.name,
        "runtime": RUNTIME,
        "backup_path": str(BACKUP.relative_to(RUN)),
        "parent_reference_path": str(PARENT),
        "backup_sha256": sha256(BACKUP),
        "parent_sha256": sha256(PARENT),
        "rows": int(len(backup)),
        "stations": int(backup["station_name"].nunique()),
        "folds": int(backup["fold_id"].nunique()),
        "columns_identical": same_columns,
        "keys_identical": same_keys,
        "max_numeric_difference": max_numeric_difference,
        "sha256_identical": sha256(BACKUP) == sha256(PARENT),
    }
    result["passed"] = bool(
        result["rows"] == 8738
        and result["stations"] == 110
        and result["folds"] == 3
        and same_columns
        and same_keys
        and max_numeric_difference <= 1.0e-8
        and result["sha256_identical"]
    )
    result["decision"] = "S0_REPRODUCTION_PASS" if result["passed"] else "S0_REPRODUCTION_FAIL"
    (REPORT / "gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "summary.md").write_text(
        "\n".join([
            "# S0矫正前基线复现门禁", "",
            f"- 判定：`{result['decision']}`",
            f"- OOF：{result['rows']}行、{result['stations']}站、{result['folds']}折",
            f"- 行键完全一致：{same_keys}",
            f"- 最大数值差：{max_numeric_difference:.3g}",
            f"- Parquet SHA-256完全一致：{result['sha256_identical']}", "",
            "S0只验证本目录矫正前快照能够逐字节复现冻结的20260805_2结果；不参与S1拟合。",
        ]) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
