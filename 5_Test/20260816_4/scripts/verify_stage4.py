from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_4")


def main() -> None:
    oof = pd.read_parquet(ROOT / "outputs" / "candidate_oof_predictions_2018_2021.parquet")
    eta = pd.read_parquet(ROOT / "outputs" / "candidate_fold_eta_parameters.parquet")
    panel = pd.read_parquet(ROOT / "outputs" / "groundwater_aquifer_decade_panel.parquet")
    states = pd.read_parquet(ROOT / "outputs" / "candidate_state_summary_2010_2018.parquet")
    engineering = json.loads((ROOT / "reports" / "candidate_engineering_and_mass_audit.json").read_text(encoding="utf-8"))
    counts = oof.groupby("model_id").size()
    checks = {
        "exactly_63_candidates": int(oof.model_id.nunique()) == 63 and len(engineering) == 63,
        "oof_4097_each": bool(len(counts) == 63 and counts.eq(4097).all()),
        "oof_keys_unique": not oof.duplicated(["model_id", "station_key", "year", "month"]).any(),
        "four_eta_folds_each": bool(eta.groupby("model_id").size().eq(4).all()),
        "eta_in_bounds": bool(eta[["eta_quick", "eta_gw"]].ge(0).all().all() and eta[["eta_quick", "eta_gw"]].le(1).all().all()),
        "eta_solver_success": bool(eta.success.all()),
        "state_230_each": bool(states.groupby("model_id").size().eq(230).all()),
        "engineering_nonnegative": all(float(item["minimum_state_or_flux_kg_n"]) >= -1e-9 for item in engineering.values()),
        "engineering_mass_relative": all(float(item["max_relative_mass_balance_error"]) <= 1e-12 for item in engineering.values()),
        "T0_T1_mutually_exclusive": all(bool(item["T0_T1_mutually_exclusive"]) for item in engineering.values()),
        "gw_pattern_eta_excluded": bool((panel.eta_gw_included == False).all()),  # noqa: E712
        "aquifer_decade_unique": not panel.duplicated(["model_id", "aquifer_id", "decade"]).any(),
        "locked_2022_absent": int(oof.year.max()) == 2021,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    result = {"pass": True, "checks": checks}
    (ROOT / "reports" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
