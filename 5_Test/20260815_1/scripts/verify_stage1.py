from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_1")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
CANON = Path(r"E:\SPARROW\5_Test\20260814_6\outputs\structural_canonical_main_interface_2006_2022.parquet")
EXPECTED_CANON_SHA = "a0b47562bae42129a6b55571e28cc5fdfc1001ee740f9c9287e6f2f70db065c2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}

    climate = pd.read_parquet(OUT / "q72_monthly_climatology_2006_2015.parquet")
    check("climatology_shape", len(climate) == 2760 and climate.reach_id.nunique() == 230 and climate.month.nunique() == 12, {"rows": len(climate), "reaches": climate.reach_id.nunique()})
    check("climatology_no_development_leakage", set(climate.climatology_start_year) == {2006} and set(climate.climatology_end_year) == {2015}, sorted(climate.climatology_end_year.unique().tolist()))
    check("bypass_bounded", climate.quick_bypass_fraction.between(-1e-12, 1 + 1e-12).all(), {"min": float(climate.quick_bypass_fraction.min()), "max": float(climate.quick_bypass_fraction.max())})
    contact_error = float((climate.soil_contact_water_mm - climate.soil_overflow_to_quick_mm - climate.gw_recharge_mm).abs().max())
    check("soil_contact_identity", contact_error <= 1e-12, contact_error)

    tn = pd.read_parquet(OUT / "tn_observation_registry_2016_2022.parquet")
    check("tn_unique_positive", not tn.duplicated(["station_key", "year", "month"]).any() and (tn.tn_mg_l > 0).all(), {"rows": len(tn), "stations": tn.station_key.nunique()})
    check("tn_locked_label", set(tn.loc[tn.year == 2022, "observation_role"]) == {"locked_final"}, int((tn.year == 2022).sum()))
    folds = pd.read_parquet(OUT / "tn_fold_registry.parquet")
    check("folds_exclude_2022", not folds.year.eq(2022).any() and set(folds.fold_id) == {"F1", "F2", "F3", "F4"}, {"rows": len(folds), "years": sorted(folds.year.unique().tolist())})

    wg = pd.read_parquet(OUT / "watergap_recharge_by_reach_2006_2022.parquet")
    check("watergap_complete", len(wg) == 46920 and wg.reach_id.nunique() == 230 and not wg.watergap_recharge_mm.isna().any(), {"rows": len(wg), "min": float(wg.watergap_recharge_mm.min()), "max": float(wg.watergap_recharge_mm.max())})
    weights = pd.read_parquet(OUT / "watergap_catchment_overlap_weights.parquet")
    weight_error = float((weights.groupby("reach_id").weight.sum() - 1).abs().max())
    check("watergap_equal_area_weights", weight_error <= 1e-12 and weights.reach_id.nunique() == 230, {"closure_error": weight_error, "reaches": weights.reach_id.nunique()})

    diagnostics = json.loads((REPORTS / "external_hydrology_diagnostics.json").read_text(encoding="utf-8"))
    allowed = {"consistent", "non_identifying", "contradictory"}
    statuses = [
        diagnostics["watergap_recharge_vs_q72_recharge"]["status"],
        diagnostics["groundwater_level_vs_q72_response_state"]["status"],
        diagnostics["hydrology_external_consistency"],
    ]
    check("external_status_vocabulary", set(statuses).issubset(allowed), statuses)
    check("external_diagnostic_isolation", diagnostics["decision_role"] == "diagnostic_only_not_in_N_loss_Q72_calibration_or_mu_selection" and diagnostics["s6_groundwater_status_remains"] == "non_identifying", diagnostics["decision_role"])

    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parent_hashes_unchanged", start == end, {"files": len(start)})
    check("canonical_hash", sha256(CANON) == EXPECTED_CANON_SHA, sha256(CANON))
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)

    failed = [name for name, value in checks.items() if not value["pass"]]
    audit = {"scenario_id": "20260815_1", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenario_id": audit["scenario_id"], "pass": audit["pass"], "failed": failed}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
