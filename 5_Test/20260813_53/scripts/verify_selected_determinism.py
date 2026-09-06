from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare(reference_path: Path, reproduced_path: Path, keys: list[str]) -> dict[str, object]:
    a = pd.read_parquet(reference_path).sort_values(keys).reset_index(drop=True)
    b = pd.read_parquet(reproduced_path).sort_values(keys).reset_index(drop=True)
    same_keys = a[keys].equals(b[keys])
    same_actual = bool(np.allclose(a.actual, b.actual, rtol=0, atol=0, equal_nan=True))
    max_diff = float(np.max(np.abs(a.predict.to_numpy(float) - b.predict.to_numpy(float))))
    return {
        "reference_rows": len(a), "reproduced_rows": len(b), "keys_identical": same_keys,
        "actual_identical": same_actual, "max_abs_prediction_difference_cfs": max_diff,
        "reference_sha256": sha256(reference_path), "reproduced_sha256": sha256(reproduced_path),
        "pass": len(a) == len(b) and same_keys and same_actual and max_diff <= 1e-8,
    }


def main() -> None:
    oof = compare(
        ROOT / "reference_selected" / "oof.parquet",
        ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet",
        ["comid", "q_site", "year", "month", "fold_id"],
    )
    validation = compare(
        ROOT / "reference_selected" / "validation.parquet",
        ROOT / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet",
        ["comid", "q_site", "year", "month"],
    )
    payload = {
        "terminal": "S111_DETERMINISTIC_REPRODUCTION_PASS" if oof["pass"] and validation["pass"] else "S111_DETERMINISTIC_REPRODUCTION_FAILURE",
        "oof": oof, "validation_2019_2022": validation,
    }
    (ROOT / "reports" / "deterministic_reproduction_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["terminal"].endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
