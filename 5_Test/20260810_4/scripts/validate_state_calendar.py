from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
INPUT = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "indata.parquet"
TOPOLOGY = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "topology" / "topology_edges.csv"
NEW_COMPONENT = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
OLD_COMPONENT = RUN.parent / "20260810_1" / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
OUT = RUN / "reports" / "state_calendar"

PARAMS = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    return module


def max_abs_difference(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    return float(np.nanmax(diff)) if len(diff) else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = load_module("q72_parent_observed_only", OLD_COMPONENT)
    new = load_module("q72_repaired_calendar", NEW_COMPONENT)

    forcing = new.load_forcing_panel()
    old_raw = old.load_observed_panel()
    old_featured = old.prepare_design(old.add_hydrologic_features(old_raw, **PARAMS))
    observed_control = new.build_featured_observation_panel(
        forcing, **PARAMS, state_calendar_mode="observed_only"
    )
    full_calendar = new.build_featured_observation_panel(
        forcing, **PARAMS, state_calendar_mode="full_forcing"
    )

    keys = ["comid", "year", "month", "q_site"]
    old_keys = old_featured[keys].astype(str)
    control_keys = observed_control[keys].astype(str)
    key_equal = old_keys.equals(control_keys)

    numeric_rows = []
    for column in sorted(set(old_featured.columns).intersection(observed_control.columns)):
        if column in keys or not pd.api.types.is_numeric_dtype(old_featured[column]):
            continue
        numeric_rows.append(
            {
                "column": column,
                "max_abs_difference": max_abs_difference(old_featured[column], observed_control[column]),
            }
        )
    differences = pd.DataFrame(numeric_rows)
    differences.to_csv(OUT / "observed_only_feature_differences.csv", index=False, encoding="utf-8-sig")

    observed = forcing.loc[new.observation_mask(forcing), ["comid", "year", "month"]].copy()
    observed["period"] = observed["year"].astype(int) * 12 + observed["month"].astype(int)
    gap_rows = []
    for comid, group in observed.sort_values(["comid", "period"]).groupby("comid", sort=False):
        gap = group["period"].diff().dropna().astype(int) - 1
        for skipped in gap[gap > 0]:
            gap_rows.append({"comid": int(comid), "skipped_forcing_months": int(skipped)})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT / "observed_internal_gaps.csv", index=False, encoding="utf-8-sig")

    forcing_keys = forcing[["comid", "year", "month"]]
    expected_grid = int(forcing["comid"].nunique()) * 204
    full_keys = full_calendar[keys].astype(str)
    observed_key_set = set(map(tuple, control_keys.to_numpy()))
    full_key_set = set(map(tuple, full_keys.to_numpy()))
    max_control_diff = float(differences["max_abs_difference"].max()) if len(differences) else 0.0

    gate = {
        "runtime": RUNTIME,
        "forcing_rows": int(len(forcing)),
        "forcing_reaches": int(forcing["comid"].nunique()),
        "forcing_months_per_reach_expected": 204,
        "forcing_expected_grid_rows": expected_grid,
        "forcing_unique_keys": int(len(forcing_keys.drop_duplicates())),
        "observed_model_rows": int(len(observed_control)),
        "full_calendar_model_rows_after_mask": int(len(full_calendar)),
        "observed_only_key_equal_parent": bool(key_equal),
        "observed_only_max_numeric_difference": max_control_diff,
        "full_calendar_same_observed_key_set": bool(full_key_set == observed_key_set),
        "no_q_rows_in_full_calendar_model_rows": int((~new.observation_mask(full_calendar)).sum()),
        "internal_gap_events": int(len(gaps)),
        "skipped_forcing_months": int(gaps["skipped_forcing_months"].sum()) if len(gaps) else 0,
        "max_internal_gap_months": int(gaps["skipped_forcing_months"].max()) if len(gaps) else 0,
    }
    gate["status"] = (
        "PASS"
        if gate["forcing_rows"] == 46920
        and gate["forcing_unique_keys"] == 46920
        and gate["observed_model_rows"] == 21440
        and gate["full_calendar_model_rows_after_mask"] == 21440
        and gate["observed_only_key_equal_parent"]
        and gate["observed_only_max_numeric_difference"] <= 1e-12
        and gate["full_calendar_same_observed_key_set"]
        and gate["no_q_rows_in_full_calendar_model_rows"] == 0
        else "FAIL"
    )
    (OUT / "state_calendar_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
