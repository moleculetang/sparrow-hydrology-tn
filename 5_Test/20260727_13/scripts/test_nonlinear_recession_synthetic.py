from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]
SCRIPTS = RUN / "scripts"
MODEL_PATH = (
    SCRIPTS / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
)
OUT = (
    RUN
    / "reports"
    / "preflight"
    / "nonlinear_recession_synthetic_test.json"
)


def load_model():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "nonlinear_recession_test_model", MODEL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    model = load_model()
    capacity = 240.0
    rate = 0.10
    storages = np.array([0.0, 1.0, 12.0, 60.0, 120.0, 239.0, 240.0])
    reference = np.array(
        [
            model.nonlinear_recession_release_mm(
                value, capacity, rate, 1.0
            )
            for value in storages
        ]
    )
    candidate = np.array(
        [
            model.nonlinear_recession_release_mm(
                value,
                capacity,
                rate,
                model.NONLINEAR_RECESSION_EXPONENT,
            )
            for value in storages
        ]
    )
    inflow = np.linspace(0.0, 12.0, len(storages))
    next_storage = storages + inflow - candidate
    balance_residual = next_storage - storages - inflow + candidate

    interior = (storages > 0.0) & (storages < capacity)
    checks = {
        "candidate_exponent_exact": bool(
            model.NONLINEAR_RECESSION_EXPONENT == 1.5
        ),
        "finite_nonnegative_release": bool(
            np.isfinite(candidate).all() and (candidate >= 0.0).all()
        ),
        "release_bounded_by_storage": bool((candidate <= storages).all()),
        "monotonic_with_storage": bool((np.diff(candidate) >= 0.0).all()),
        "candidate_not_above_linear": bool(
            (candidate <= reference + 1.0e-12).all()
        ),
        "candidate_strictly_below_linear_below_capacity": bool(
            (candidate[interior] < reference[interior]).all()
        ),
        "same_release_at_zero_and_capacity": bool(
            np.allclose(candidate[[0, -1]], reference[[0, -1]])
        ),
        "one_step_mass_balance_exact": bool(
            np.max(np.abs(balance_residual)) <= 1.0e-12
        ),
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, ensure_ascii=False, indent=2))

    payload = {
        "status": "passed",
        "checks": checks,
        "capacity_mm": capacity,
        "release_rate": rate,
        "reference_exponent": 1.0,
        "candidate_exponent": float(
            model.NONLINEAR_RECESSION_EXPONENT
        ),
        "storage_mm": storages.tolist(),
        "linear_release_mm": reference.tolist(),
        "candidate_release_mm": candidate.tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
