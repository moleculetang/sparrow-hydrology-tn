"""Validate Stage-15 static feature artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260826_15")


def main() -> None:
    qa = json.loads((RUN / "reports" / "multiscale_attribute_qa.json").read_text(encoding="utf-8"))
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    registry = json.loads((RUN / "reports" / "feature_registry.json").read_text(encoding="utf-8"))
    local = pd.read_parquet(RUN / "outputs" / "local_static_features_standardized.parquet")
    multi = pd.read_parquet(RUN / "outputs" / "multiscale_static_features_standardized.parquet")
    checks = {
        "qa_pass": qa["all_checks_pass"],
        "230_unique_reaches": len(local) == 230 and len(multi) == 230 and local.reach_id.nunique() == 230 and multi.reach_id.nunique() == 230,
        "local_7": len(registry["local_features"]) == 7 and local.shape[1] == 8,
        "multiscale_22": len(registry["multiscale_features"]) == 22 and multi.shape[1] == 23,
        "finite": bool(np.isfinite(local.iloc[:, 1:].to_numpy()).all() and np.isfinite(multi.iloc[:, 1:].to_numpy()).all()),
        "no_Q_or_TN": not contract["discharge_read"] and not contract["TN_read"],
        "no_training": not contract["candidate_training_performed"],
        "stage16_next": contract["authorized_successor"] == "20260826_16",
    }
    result = {"stage": "20260826_15", "checks": checks, "all_checks_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 15 validation failed: {checks}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
