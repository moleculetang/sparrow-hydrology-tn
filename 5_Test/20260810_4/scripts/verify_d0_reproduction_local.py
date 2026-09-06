from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "outputs" / "q72_three_fold_oof_predictions.parquet"
DEFAULT_CANDIDATE = RUN / "outputs" / "scenarios" / "S00" / "q72_three_fold_oof_predictions.parquet"
DEFAULT_OUT = RUN / "reports" / "d0_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(REFERENCE).rename(
        columns={
            "reach_id": "comid",
            "station_name": "q_site",
            "observed_cfs": "actual",
            "predicted_cfs": "predict",
        }
    )
    candidate = pd.read_parquet(candidate_path)
    keys = ["comid", "q_site", "year", "month", "fold_id"]
    reference = reference.sort_values(keys).reset_index(drop=True)
    candidate = candidate.sort_values(keys).reset_index(drop=True)
    key_equal = reference[keys].astype(str).equals(candidate[keys].astype(str))

    rows = []
    for column in sorted(set(reference.columns).intersection(candidate.columns)):
        if column in keys or not pd.api.types.is_numeric_dtype(reference[column]):
            continue
        a = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(candidate[column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        delta = np.abs(a - b)
        delta[both_nan] = 0.0
        rows.append(
            {
                "column": column,
                "max_abs_difference": float(np.nanmax(delta)) if len(delta) else 0.0,
                "different_rows_gt_1e_8": int(np.nansum(delta > 1e-8)),
            }
        )
    differences = pd.DataFrame(rows)
    differences.to_csv(out / "column_differences.csv", index=False, encoding="utf-8-sig")
    max_difference = float(differences["max_abs_difference"].max()) if len(differences) else 0.0
    gate = {
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "reference_stations": int(reference["q_site"].nunique()),
        "candidate_stations": int(candidate["q_site"].nunique()),
        "folds_equal": sorted(reference["fold_id"].unique().tolist()) == sorted(candidate["fold_id"].unique().tolist()),
        "keys_equal": bool(key_equal),
        "compared_numeric_columns": int(len(differences)),
        "max_abs_difference": max_difference,
    }
    gate["status"] = (
        "PASS"
        if gate["reference_rows"] == 8738
        and gate["candidate_rows"] == 8738
        and gate["keys_equal"]
        and gate["folds_equal"]
        and max_difference <= 1e-8
        else "FAIL"
    )
    gate["candidate_path"] = str(candidate_path)
    (out / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
