from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    spin = pd.read_parquet(OUT / "operator_spinup_audit.parquet")
    pred = pd.read_parquet(OUT / "candidate_oof_predictions_2018_2021.parquet")
    eng = pd.read_parquet(OUT / "operator_engineering_audit.parquet")
    matrix = pd.read_parquet(OUT / "operator_model_gate_matrix.parquet")
    f00 = json.loads((REPORTS / "f00_all_model_oof_reproduction_audit.json").read_text(encoding="utf-8"))
    states = json.loads((REPORTS / "f00_representative_state_flux_reproduction_audit.json").read_text(encoding="utf-8"))
    semantics = json.loads((REPORTS / "q72_flux_semantics_audit.json").read_text(encoding="utf-8"))
    complete = json.loads((REPORTS / "completion_audit.json").read_text(encoding="utf-8"))
    checks = {
        "parent_hashes_unchanged": bool(complete["parent_hashes_unchanged"]),
        "q72_semantics": semantics["status"] == "PASS",
        "forty_eight_independent_spinups": bool(len(spin) == 48 and spin.candidate_id.nunique() == 48),
        "all_spinups_converged": bool(spin.converged.all() and spin.terminal_max_abs_delta_kg_n.le(1e-9).all()),
        "f00_all_oof_reproduced": f00["status"] == "PASS",
        "f00_three_full_states_reproduced": states["status"] == "PASS",
        "candidate_count": bool(pred.candidate_id.nunique() == 48),
        "two_layers": set(pred.layer) == {"P1", "P2"},
        "oof_keys": bool(pred.groupby(["candidate_id", "layer"]).size().eq(4097).all()),
        "mass_balance": bool(eng.max_relative_mass_balance_error.le(1e-12).all()),
        "nonnegative": bool(eng.minimum_state_or_flux_kg_n.ge(-1e-12).all()),
        "thirty_six_alternative_model_gates": bool(len(matrix) == 36),
        "no_2022": bool(pred.year.max() == 2021),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "checks": checks}
    (REPORTS / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
