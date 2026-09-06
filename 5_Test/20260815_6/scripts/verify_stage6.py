from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_6")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    checks: dict[str, dict[str, object]] = {}
    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}
    routed = pd.read_parquet(OUT / "r0_routed_tn_1961_2022.parquet")
    audit = json.loads((REPORTS / "network_mass_balance_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "routing_decision.json").read_text(encoding="utf-8"))
    check("coverage", len(routed) == 171120 and routed.reach_id.nunique() == 230 and len(routed[["year", "month"]].drop_duplicates()) == 744, {"rows": len(routed), "reaches": routed.reach_id.nunique()})
    check("unique_keys", not routed.duplicated(["reach_id", "year", "month"]).any(), None)
    check("full_terminal_trees", routed.terminal_tree_id.nunique() == 14 and audit["summary"]["full_terminal_tree_count"] == 14, int(routed.terminal_tree_id.nunique()))
    check("terminal_network_mass_balance", audit["summary"]["max_terminal_mass_abs_error_kg_n"] <= 1e-6 and audit["summary"]["max_terminal_mass_relative_error"] <= 1e-14, {"max_abs": audit["summary"]["max_terminal_mass_abs_error_kg_n"], "max_relative": audit["summary"]["max_terminal_mass_relative_error"]})
    check("age_fractions_bounded_nested", routed[["fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"]].ge(-1e-12).all().all() and routed[["fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"]].le(1+1e-12).all().all() and (routed.fraction_memory_gt1y + 1e-12 >= routed.fraction_memory_gt5y).all() and (routed.fraction_memory_gt5y + 1e-12 >= routed.fraction_memory_gt10y).all(), routed[["fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"]].agg(["min", "max"]).to_dict())
    check("R0_only_no_R1", decision["routing_model_id"] == "R0_same_month_mass_conserving" and decision["R1_status"] == "not_run_no_reliable_tau_r" and decision["new_fitted_parameter_count"] == 0, decision)
    check("point_source_not_fabricated", set(routed.point_source_tn_status) == {"not_available_not_fabricated"}, None)
    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parents_unchanged", start == end, start)
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)
    failed = [k for k,v in checks.items() if not v["pass"]]
    default = lambda x: x.item() if hasattr(x, "item") else str(x)
    result = {"scenario_id": "20260815_6", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_6", "pass": not failed, "failed": failed}, ensure_ascii=False, indent=2, default=default))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
