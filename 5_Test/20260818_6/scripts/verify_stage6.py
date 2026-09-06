from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_6")


def main() -> None:
    manifest = json.loads((ROOT / "reports" / "conditional_final_model_manifest.json").read_text(encoding="utf-8"))
    models = pd.read_parquet(ROOT / "outputs" / "final_formal_model_registry.parquet")
    components = pd.read_parquet(ROOT / "outputs" / "final_component_registry.parquet")
    checks = {
        "twelve_unique_models": bool(len(models) == 12 and models.model_id.nunique() == 12),
        "only_F00": bool(models.source_water_operator.eq("F00").all()),
        "WWTP_not_included": bool(not components.loc[components.component.eq("municipal WWTP PS1"), "included"].iloc[0]),
        "no_unique_mu": bool(not manifest["operational_model"]["unique_mu_selected"]),
        "no_unmonitored_reach_claim": manifest["generalization_boundary"]["unmonitored_reach_claim"] == "not supported",
        "OOF_2018_2021": manifest["time_boundaries"]["OOF_evaluation_years"] == [2018, 2019, 2020, 2021],
        "WWTP_unavailable_after_2019": manifest["time_boundaries"]["2020_2022_wwtp_forcing_status"] == "unavailable_not_filled",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    (ROOT / "reports" / "independent_verification.json").write_text(
        json.dumps({"status": status, "checks": checks}, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError("STAGE6_INDEPENDENT_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
