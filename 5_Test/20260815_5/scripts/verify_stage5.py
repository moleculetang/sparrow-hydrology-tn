from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW\5_Test\20260815_5")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    checks: dict[str, dict[str, object]] = {}
    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}
    decision = json.loads((REPORTS / "delivery_structure_decision.json").read_text(encoding="utf-8"))
    audits = json.loads((REPORTS / "candidate_mass_and_cohort_audit.json").read_text(encoding="utf-8"))
    expected = {"T0", *{f"T1_mu_{x:03d}m" for x in [12, 36, 60, 96, 144, 240]}}
    check("all_month_unit_candidates", set(audits) == expected, sorted(audits))
    check("no_year_unit_parameter", all("year" not in key.lower() for key in decision if key != "locked_2022_used_for_selection"), list(decision))
    max_abs = max(v["max_abs_mass_balance_error_kg_n"] for v in audits.values())
    max_rel = max(v["max_relative_mass_balance_error"] for v in audits.values())
    check("all_candidate_mass_balance", max_abs <= 1e-6 and max_rel <= 1e-14, {"max_abs_kg_n": max_abs, "max_relative": max_rel})
    check("all_spinups_converged", all(v["spinup"]["converged"] for v in audits.values()), {k: v["spinup"] for k, v in audits.items()})
    check("all_pool_masses_nonnegative", min(v["minimum_pool_mass_kg_n"] for v in audits.values()) >= -1e-10, min(v["minimum_pool_mass_kg_n"] for v in audits.values()))
    check("T0_exactly_nested", decision["T0_source_reference_max_abs_difference_kg_n"] <= 1e-8, decision["T0_source_reference_max_abs_difference_kg_n"])
    oof = pd.read_parquet(OUT / "delivery_candidate_oof_predictions_2018_2021.parquet")
    check("development_only_selection", set(oof.year) == {2018, 2019, 2020, 2021} and not decision["locked_2022_used_for_selection"], sorted(set(oof.year)))
    check("same_oof_keys", oof.groupby("model_id").size().nunique() == 1 and not oof.duplicated(["model_id", "station_key", "year", "month"]).any(), oof.groupby("model_id").size().to_dict())
    boot = pd.read_parquet(OUT / "delivery_structure_bootstrap_distributions.parquet")
    check("bootstrap_contract", len(boot) == 6 * 2 * 10000 and set(boot.block) == {"station_key", "terminal_tree_id"}, {"rows": len(boot), "comparisons": len(boot.candidate.unique())})
    selected = pd.read_parquet(OUT / "selected_delivery_model_reach_month_1961_2022.parquet")
    check("selected_full_history", len(selected) == 171120 and selected.reach_id.nunique() == 230 and selected.model_id.iloc[0] == decision["selected_delivery_model_id"], {"rows": len(selected), "model": selected.model_id.iloc[0]})
    cohort_path = OUT / "selected_delivery_model_cohort_state_end_2022.parquet"
    check("cohort_parquet_valid", pq.read_metadata(cohort_path).num_rows > 0, pq.read_metadata(cohort_path).num_rows)
    cohort = pd.read_parquet(cohort_path)
    check("cohort_has_transit_pool_when_applicable", set(cohort.pool).issubset({"son", "mobile", "quick", "transit", "base"}), sorted(set(cohort.pool)))
    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parents_unchanged", start == end, start)
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)
    failed = [k for k, v in checks.items() if not v["pass"]]
    default = lambda x: x.item() if hasattr(x, "item") else str(x)
    audit = {"scenario_id": "20260815_5", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_5", "pass": not failed, "failed": failed, "selected": decision["selected_delivery_model_id"]}, ensure_ascii=False, indent=2, default=default))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
