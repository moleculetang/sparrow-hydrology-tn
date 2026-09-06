from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_4")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}

    decision = json.loads((REPORTS / "source_structure_decision.json").read_text(encoding="utf-8"))
    audits = json.loads((REPORTS / "candidate_mass_and_cohort_audit.json").read_text(encoding="utf-8"))
    expected = {"M0", "S0", *{f"S1_tau_{x:03d}m" for x in [12, 36, 60, 96, 144, 240, 480]}}
    check("all_nested_candidates_evaluated", set(audits) == expected, sorted(audits))
    max_balance = max(v["max_abs_mass_balance_error_kg_n"] for v in audits.values())
    max_relative_balance = max(v["max_relative_mass_balance_error"] for v in audits.values())
    check("all_candidate_mass_balance", max_balance <= 1e-6 and max_relative_balance <= 1e-14, {"max_abs_kg_n": max_balance, "max_relative": max_relative_balance})
    check("all_spinups_converged", all(v["spinup"]["converged"] for v in audits.values()), {k: v["spinup"] for k, v in audits.items()})
    check("all_pools_nonnegative", min(v["minimum_pool_mass_kg_n"] for v in audits.values()) >= -1e-10, min(v["minimum_pool_mass_kg_n"] for v in audits.values()))
    routed = pd.read_parquet(OUT / "candidate_routed_reach_month_2016_2022.parquet")
    check("routed_coverage", len(routed) == len(expected) * 230 * 84 and routed.reach_id.nunique() == 230, {"rows": len(routed), "models": routed.model_id.nunique()})
    check("full_topology_terminal_tree_count", routed.terminal_tree_id.nunique() == 14 and decision["full_topology_terminal_tree_count"] == 14, int(routed.terminal_tree_id.nunique()))
    age_cols = ["fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"]
    check("age_fractions_bounded_and_nested", routed[age_cols].ge(-1e-12).all().all() and routed[age_cols].le(1 + 1e-12).all().all() and (routed.fraction_memory_gt1y + 1e-12 >= routed.fraction_memory_gt5y).all() and (routed.fraction_memory_gt5y + 1e-12 >= routed.fraction_memory_gt10y).all(), routed[age_cols].agg(["min", "max"]).to_dict())
    oof = pd.read_parquet(OUT / "source_candidate_oof_predictions_2018_2021.parquet")
    check("tn_participating_terminal_tree_blocks", oof.terminal_tree_id.nunique() == decision["tn_participating_terminal_tree_count"] and oof.terminal_tree_id.nunique() > 1, int(oof.terminal_tree_id.nunique()))
    check("oof_years_and_locked_year", set(oof.year.unique()) == {2018, 2019, 2020, 2021} and 2022 not in set(oof.year), sorted(oof.year.unique()))
    check("all_candidates_same_oof_keys", oof.groupby("model_id").size().nunique() == 1 and not oof.duplicated(["model_id", "station_key", "year", "month"]).any(), oof.groupby("model_id").size().to_dict())
    boot = pd.read_parquet(OUT / "source_structure_bootstrap_distributions.parquet")
    expected_comparisons = 1 + 7 + 7
    check("bootstrap_contract", len(boot) == expected_comparisons * 2 * 10000 and set(boot.block) == {"station_key", "terminal_tree_id"}, {"rows": len(boot), "comparisons": len(boot[["candidate", "reference"]].drop_duplicates())})
    selected = pd.read_parquet(OUT / "selected_source_model_reach_month_1961_2022.parquet")
    check("selected_full_history", len(selected) == 171120 and selected.reach_id.nunique() == 230 and selected.model_id.nunique() == 1 and selected.model_id.iloc[0] == decision["selected_source_model_id"], {"rows": len(selected), "model": selected.model_id.iloc[0]})
    cohort = pd.read_parquet(OUT / "selected_source_model_cohort_state_end_2022.parquet")
    final = selected.sort_values(["reach_id", "year", "month"]).groupby("reach_id").tail(1)
    closure = 0.0
    for pool in ["son", "mobile", "quick", "base"]:
        expected_mass = final.set_index("reach_id")[f"{pool}_state_end_kg_n"]
        subset = cohort.loc[cohort.pool.eq(pool)]
        actual = {
            int(rid): float(np.sum(group.cohort_mass_kg_n.to_numpy(float), dtype=np.longdouble))
            for rid, group in subset.groupby("reach_id", sort=False)
        }
        actual_mass = pd.Series(actual, dtype=float).reindex(expected_mass.index, fill_value=0.0)
        closure = max(closure, float((actual_mass - expected_mass).abs().max()))
    check("selected_final_cohort_closure", closure <= 1e-8, closure)
    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parents_unchanged", start == end, start)
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)
    failed = [k for k, v in checks.items() if not v["pass"]]
    audit = {"scenario_id": "20260815_4", "pass": not failed, "failed": failed, "checks": checks}
    json_default = lambda value: value.item() if hasattr(value, "item") else str(value)
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_4", "pass": not failed, "failed": failed, "selected": decision["selected_source_model_id"]}, ensure_ascii=False, indent=2, default=json_default))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
