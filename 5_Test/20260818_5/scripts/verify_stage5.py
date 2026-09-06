from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_5")


def main() -> None:
    decision = json.loads((ROOT / "reports" / "remaining_process_diagnostic_decision.json").read_text(encoding="utf-8"))
    negative = pd.read_parquet(ROOT / "outputs" / "negative_surplus_model_summary.parquet")
    registry = pd.read_parquet(ROOT / "outputs" / "residual_diagnostic_registry.parquet")
    temperature = pd.read_parquet(ROOT / "outputs" / "temperature_model_gates.parquet")
    pon = pd.read_parquet(ROOT / "outputs" / "pon_model_gates.parquet")
    checks = {
        "diagnostic_only": decision["diagnostic_only"] and not decision["model_changed"],
        "twelve_models_negative": negative.model_id.nunique() == 12,
        "twelve_models_temperature": temperature.model_id.nunique() == 12,
        "twelve_models_pon": pon.model_id.nunique() == 12,
        "four_folds": bool(registry.fold_id.nunique() == 4),
        "eight_terminal_trees": bool(registry.terminal_tree_id.nunique() == 8),
        "no_2022": bool(registry.year.max() == 2021),
        "parent_verification_pass": json.loads((ROOT / "reports" / "verification.json").read_text(encoding="utf-8"))["status"] == "PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    (ROOT / "reports" / "independent_verification.json").write_text(
        json.dumps({"status": status, "checks": checks}, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError("STAGE5_INDEPENDENT_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
