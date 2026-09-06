from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_3")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    registry = pd.read_parquet(OUT / "wwtp_scenario_registry.parquet")
    pred = pd.read_parquet(OUT / "wwtp_candidate_oof_predictions_2018_2019.parquet")
    params = pd.read_parquet(OUT / "candidate_fold_readout_parameters.parquet")
    source = json.loads((REPORTS / "wwtp_source_nonoverlap_audit.json").read_text(encoding="utf-8"))
    complete = json.loads((REPORTS / "completion_audit.json").read_text(encoding="utf-8"))
    checks = {
        "parent_hashes_unchanged": bool(complete["parent_hashes_unchanged"]),
        "source_nonoverlap_pass": source["status"] == "PASS",
        "five_scenarios": registry.scenario_id.nunique() == 5,
        "forcing_2006_2019_only": registry.year.min() == 2006 and registry.year.max() == 2019,
        "no_negative_mass": bool(registry.local_wwtp_tn_kg_n.ge(0).all()),
        "sixty_candidates": pred.candidate_id.nunique() == 60,
        "two_layers": set(pred.layer) == {"P1", "P2"},
        "native_oof_1711": bool(pred.groupby(["candidate_id", "layer"]).size().eq(1711).all()),
        "years_2018_2019": set(pred.year) == {2018, 2019},
        "independent_readouts": params.groupby(["candidate_id", "layer"]).fold_id.nunique().eq(2).all(),
        "wwtp_not_eta_scaled_contract": json.loads((REPORTS / "readout_training_contract.json").read_text(encoding="utf-8"))["WWTP_not_scaled_by_eta"],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "checks": checks}
    (REPORTS / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
