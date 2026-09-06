from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reference_path = ROOT / "reference_baseline" / "P1" / "q72_three_fold_oof_predictions.parquet"
    reproduced_path = ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
    reference = pd.read_parquet(reference_path).sort_values(KEY).reset_index(drop=True)
    reproduced = pd.read_parquet(reproduced_path).sort_values(KEY).reset_index(drop=True)
    same_keys = reference[KEY].equals(reproduced[KEY])
    same_actual = bool(
        np.allclose(reference["actual"], reproduced["actual"], rtol=0.0, atol=0.0, equal_nan=True)
    )
    max_abs_prediction_difference = float(
        np.max(np.abs(reference["predict"].to_numpy(float) - reproduced["predict"].to_numpy(float)))
    )
    payload = {
        "terminal": "P1_INDEPENDENT_REPRODUCTION_PASS"
        if len(reference) == 8822 and len(reproduced) == 8822 and same_keys and same_actual
        and max_abs_prediction_difference <= 1e-8
        else "P1_INDEPENDENT_REPRODUCTION_FAILURE",
        "reference_rows": int(len(reference)),
        "reproduced_rows": int(len(reproduced)),
        "keys_identical": same_keys,
        "actual_identical": same_actual,
        "max_abs_prediction_difference_cfs": max_abs_prediction_difference,
        "reference_oof_sha256": sha256(reference_path),
        "reproduced_oof_sha256": sha256(reproduced_path),
        "input_sha256": sha256(ROOT / "inputs" / "parent_indata.parquet"),
        "component_sha256": sha256(ROOT / "scripts" / "components" / "q72_prior_semantics_component.py"),
    }
    (ROOT / "reports" / "reproduction_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["terminal"].endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
