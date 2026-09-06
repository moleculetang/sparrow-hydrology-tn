from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_5")


def main() -> None:
    table = pd.read_csv(ROOT / "reports" / "progressive_candidate_adjudication.csv")
    boot = pd.read_parquet(ROOT / "outputs" / "candidate_bootstrap_distributions.parquet")
    ci = pd.read_csv(ROOT / "reports" / "candidate_bootstrap_ci.csv")
    decision = json.loads((ROOT / "reports" / "legacy_attribution_decision.json").read_text(encoding="utf-8"))
    expected_admissible = table.engineering_pass & table.soil_structural_admissible & table.hydrogeo_pass & table.river_noninferior
    checks = {
        "all_63_candidates_adjudicated": len(table) == 63 and table.model_id.nunique() == 63,
        "bootstrap_rows_exact": len(boot) == 63 * 2 * 10000,
        "bootstrap_blocks_exact": set(boot.block) == {"station_key", "terminal_tree_id"},
        "ci_rows_63": len(ci) == 63 and ci.model_id.nunique() == 63,
        "strict_noninferiority_rule": bool((ci.river_noninferior == ((ci.station_ci_upper < 0.01) & (ci.terminal_tree_ci_upper < 0.01))).all()),
        "progressive_gate_exact": bool((table.final_structural_admissible == expected_admissible).all()),
        "controls_not_formal_soil_candidates": bool((~table.loc[table.source_structure.isin(["M0", "S0"]), "soil_structural_admissible"]).all()),
        "gw_context_not_hard_gate": bool((table.gw_context_is_hard_gate == False).all()),  # noqa: E712
        "decision_count_matches": int(table.final_structural_admissible.sum()) == int(decision["n_admissible"]),
        "no_gate_relaxation": decision["empty_set_gate_relaxation_applied"] is False and decision["grid_expanded"] is False,
        "locked_2022_absent": decision["locked_2022_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    result = {"pass": True, "checks": checks}
    (ROOT / "reports" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
