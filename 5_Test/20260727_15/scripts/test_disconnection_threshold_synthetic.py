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
    / "disconnection_threshold_synthetic_test.json"
)


def load_model():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "disconnection_threshold_test_model", MODEL_PATH
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
    threshold = (
        model.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION * capacity
    )
    storages = np.array(
        [0.0, 12.0, 59.999, 60.0, 60.001, 90.0, 180.0, 240.0]
    )
    reference = rate * storages
    candidate = np.array(
        [
            model.threshold_slow_release_mm(
                value,
                capacity,
                rate,
                model.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION,
            )
            for value in storages
        ]
    )
    inflow = np.linspace(0.0, 14.0, len(storages))
    next_storage = storages + inflow - candidate
    balance_residual = next_storage - storages - inflow + candidate
    below_or_equal = storages <= threshold
    above = storages > threshold

    checks = {
        "fixed_threshold_fraction_exact": bool(
            model.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION == 0.25
        ),
        "threshold_storage_exact": bool(threshold == 60.0),
        "finite_nonnegative_release": bool(
            np.isfinite(candidate).all() and (candidate >= 0.0).all()
        ),
        "release_bounded_by_storage": bool((candidate <= storages).all()),
        "zero_release_below_or_at_threshold": bool(
            np.allclose(candidate[below_or_equal], 0.0)
        ),
        "positive_release_above_threshold": bool(
            (candidate[above] > 0.0).all()
        ),
        "candidate_not_above_linear_reference": bool(
            (candidate <= reference + 1.0e-12).all()
        ),
        "monotonic_with_storage": bool((np.diff(candidate) >= 0.0).all()),
        "continuous_at_threshold": bool(candidate[3] == 0.0),
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
        "threshold_fraction": (
            model.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION
        ),
        "threshold_storage_mm": threshold,
        "storage_mm": storages.tolist(),
        "linear_reference_release_mm": reference.tolist(),
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
