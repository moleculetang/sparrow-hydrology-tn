from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_3")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}

    data = pd.read_parquet(OUT / "m0_reach_month_local_n_1961_2022.parquet")
    check("coverage", len(data) == 171120 and data.reach_id.nunique() == 230 and len(data[["year", "month"]].drop_duplicates()) == 744, {"rows": len(data), "reaches": data.reach_id.nunique()})
    check("unique_keys", not data.duplicated(["reach_id", "year", "month"]).any(), None)
    check("m0_mobile_has_no_carryover", float(data.m0_mobile_state_end_kg_n.abs().max()) == 0.0, float(data.m0_mobile_state_end_kg_n.abs().max()))
    check("bypass_bounded", data.quick_bypass_fraction_recomputed.between(0, 1).all(), [float(data.quick_bypass_fraction_recomputed.min()), float(data.quick_bypass_fraction_recomputed.max())])
    positive = data.positive_legacy_eligible_n_surplus_kg_n_month
    bypass_identity = float((data.direct_current_quick_n_input_kg_n - positive * data.quick_bypass_fraction_recomputed).abs().max())
    check("direct_current_quick_identity", bypass_identity <= 1e-8, bypass_identity)
    contact = data.soil_overflow_to_quick_mm + data.gw_recharge_mm
    contact_error = float((data.soil_contact_water_mm - contact).abs().max())
    check("only_overflow_plus_recharge_flushes_soil_n", contact_error <= 1e-12, contact_error)
    check("zero_contact_zero_soil_release", float(data.loc[contact <= 1e-12, "m0_current_mobile_released_kg_n"].abs().max()) == 0.0 if (contact <= 1e-12).any() else True, int((contact <= 1e-12).sum()))
    check("nonnegative_fluxes_and_states", (data[["direct_current_quick_n_input_kg_n", "quick_path_n_input_kg_n", "gw_path_n_input_kg_n", "quick_n_release_kg_n", "base_n_release_kg_n", "quick_n_state_end_kg_n", "base_n_state_end_kg_n", "m0_same_month_unmobilized_sink_kg_n"]] >= -1e-10).all().all(), None)
    balance = float(data.m0_system_mass_balance_error_kg_n.abs().max())
    check("row_mass_balance", balance <= 1e-8, balance)
    release_identity = float((data.q_local_total_mm - data.quick_release_mm - data.gw_discharge_mm).abs().max())
    check("water_release_identity", release_identity <= 1e-12, release_identity)
    check("no_point_source_mass_fabrication", data.point_source_tn_kg_n_month.isna().all(), None)
    spinup = json.loads((REPORTS / "spinup_audit.json").read_text(encoding="utf-8"))
    check("pre1961_periodic_spinup", spinup["converged"] and spinup["terminal_max_abs_delta_kg_n"] <= 1e-10, spinup)
    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parents_unchanged", start == end, start)
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)
    failed = [name for name, value in checks.items() if not value["pass"]]
    audit = {"scenario_id": "20260815_3", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_3", "pass": not failed, "failed": failed}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
