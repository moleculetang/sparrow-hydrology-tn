from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_6")


def main() -> None:
    decision = json.loads((ROOT / "reports" / "regionalization_decision.json").read_text(encoding="utf-8"))
    diagnostics = pd.read_csv(ROOT / "reports" / "regionalization_signal_diagnostics.csv")
    groups = pd.read_csv(ROOT / "reports" / "terminal_tree_hydrogeo_groups.csv")
    if decision["regionalization_authorized"]:
        metrics = pd.read_csv(ROOT / "reports" / "regional_candidate_metrics.csv")
        candidate_rule = bool((metrics.fast_mu_month <= metrics.medium_mu_month).all() and (metrics.medium_mu_month <= metrics.slow_mu_month).all())
        candidate_count_ok = len(metrics) > 0
    else:
        candidate_rule = True
        candidate_count_ok = not (ROOT / "reports" / "regional_candidate_metrics.csv").exists()
    checks = {
        "all_14_terminal_trees_hydrogeo_profiled": len(groups) == 14 and groups.terminal_tree_id.nunique() == 14,
        "station_bearing_tree_signal_coverage": int(diagnostics.n_terminal_trees.min()) >= 4,
        "both_gradient_methods_tested": set(diagnostics.gradient_method) == {"local_gradient", "path_gradient"},
        "all_soil_admissible_taus_tested": set(diagnostics.soil_tau_month.astype(int)) == set(map(int, decision["soil_admissible_tau_month"])),
        "authorization_matches_contract": bool(decision["regionalization_authorized"] == (decision["soil_memory_status"] == "bounded" and decision["all_soil_tau_signal_stable"] and decision["local_path_tree_group_agreement_fraction"] >= decision["minimum_required_group_agreement_fraction"])),
        "monotone_regional_candidates_only": candidate_rule,
        "conditional_candidate_count_valid": candidate_count_ok,
        "registered_mu_only": decision["registered_mu_only"] is True,
        "locked_2022_absent": decision["locked_2022_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    result = {"pass": True, "checks": checks}
    (ROOT / "reports" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
