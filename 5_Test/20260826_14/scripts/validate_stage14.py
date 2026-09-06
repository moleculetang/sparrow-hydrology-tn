"""Independent artifact validation for Stage 14."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260826_14")


def main() -> None:
    numerical = json.loads((RUN / "reports" / "numerical_validation.json").read_text(encoding="utf-8"))
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    environment = json.loads((RUN / "reports" / "environment_lock.json").read_text(encoding="utf-8"))
    bridge = pd.read_parquet(RUN / "outputs" / "frozen_parent_bridge_reproduction.parquet")
    state = pd.read_parquet(RUN / "outputs" / "torch_numpy_state_flux_reproduction.parquet")
    gradient = pd.read_parquet(RUN / "outputs" / "autograd_finite_difference_check.parquet")
    checks = {
        "numerical_all_pass": numerical["all_checks_pass"],
        "no_candidate_training": not contract["candidate_training_performed"],
        "no_TN": not contract["TN_read"],
        "stage15_only_next": contract["authorized_successor"] == "20260826_15",
        "dedicated_environment": "hydro_dpl_20260826" in environment["python_executable"],
        "unsafe_openmp_override_absent": not environment["unsafe_KMP_DUPLICATE_LIB_OK_present"],
        "eight_bridge_fields": len(bridge) == 8,
        "nine_state_flux_fields": len(state) == 9,
        "eight_gradients": len(gradient) == 8 and bool(gradient.finite.all()),
    }
    result = {"stage": "20260826_14", "checks": checks, "all_checks_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 14 artifact validation failed: {checks}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
