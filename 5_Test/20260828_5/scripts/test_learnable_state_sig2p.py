"""Preflight tests for the learnable state-consistent SIG2P operator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_5"
sys.path[:0] = [
    str(RUN / "scripts"),
    str(ROOT / "5_Test" / "20260827_7" / "scripts"),
    str(ROOT / "5_Test" / "20260826_27" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from state_consistent_sig2p import simulate_state_consistent_dyn2p  # noqa: E402


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(260828)
    n_time, n_reach = 40, 4
    precipitation = torch.rand((n_time, n_reach), dtype=torch.float64) * 12.0
    precipitation[:5] = 0.0
    pet = torch.full((n_time, n_reach), 2.5, dtype=torch.float64)
    api3 = torch.zeros_like(precipitation)
    api30 = torch.zeros_like(precipitation)
    sin_doy = torch.sin(torch.arange(n_time, dtype=torch.float64) * 2.0 * torch.pi / 365.25)
    cos_doy = torch.cos(torch.arange(n_time, dtype=torch.float64) * 2.0 * torch.pi / 365.25)
    physical = torch.tensor([150.0, 1.5, 0.7, 1.2, 8.0, 1.5, 8.0, 70.0], dtype=torch.float64)
    initial = torch.tensor([[50.0, 2.0, 20.0]] * n_reach, dtype=torch.float64)
    static = torch.zeros((n_reach, 3), dtype=torch.float64)
    center = torch.zeros(11, dtype=torch.float64)
    scale = torch.ones(11, dtype=torch.float64)
    score = torch.tensor([-1.0, -0.25, 0.5, 1.25], dtype=torch.float64)
    return precipitation, pet, api3, api30, sin_doy, cos_doy, physical, initial, static, center, scale, score


def main() -> None:
    torch.set_default_dtype(torch.float64)
    values = make_inputs()
    common = values[:-1]
    score = values[-1]
    old = simulate_state_consistent_dyn2p(
        *common, None, score, 0.0, collect_storage=True, collect_aet=True,
        collect_internal_fluxes=True,
    )
    new = simulate_learnable_sig2p(
        *common, None, score, 0.0, collect_storage=True, collect_aet=True,
        collect_internal_fluxes=True,
    )
    identity_error = max(
        float(torch.max(torch.abs(old.components_mm_day - new.components_mm_day))),
        float(torch.max(torch.abs(old.storage_mm - new.storage_mm))),
        float(torch.max(torch.abs(old.aet_mm_day - new.aet_mm_day))),
        float(torch.max(torch.abs(old.percolation_to_lower_mm_day - new.percolation_to_lower_mm_day))),
    )

    raw_lambda = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    learnable = simulate_learnable_sig2p(
        *common, None, score, torch.sigmoid(raw_lambda), collect_storage=True,
        collect_internal_fluxes=True,
    )
    objective = learnable.components_mm_day[:, :, 1].sum()
    objective.backward()
    gradient = float(raw_lambda.grad)

    storage = learnable.storage_mm
    components = learnable.components_mm_day
    percolation = learnable.percolation_to_lower_mm_day
    lower_previous = torch.cat((common[7][:, 2][None, :], storage[:-1, :, 2]), dim=0)
    lower_balance = lower_previous + percolation - components[:, :, 1] - storage[:, :, 2]
    lower_balance_error = float(torch.max(torch.abs(lower_balance)))
    component_closure = float(torch.max(torch.abs(components.sum(dim=2) - components[:, :, 0] - components[:, :, 1])))
    all_nonnegative = bool((components >= -1.0e-12).all() and (storage >= -1.0e-12).all() and (percolation >= -1.0e-12).all())

    # A one-day pulse must enter the lower storage before being released over later days.
    pulse = list(make_inputs())
    pulse[0] = torch.zeros_like(pulse[0])
    pulse[0][5, :] = 30.0
    pulse[1] = torch.zeros_like(pulse[1])
    pulse[7] = torch.zeros_like(pulse[7])
    pulse[7][:, 0] = 150.0
    pulse_result = simulate_learnable_sig2p(
        *pulse[:-1], None, pulse[-1], torch.tensor(1.0), collect_storage=True,
        collect_internal_fluxes=True,
    )
    delayed_slow = float(pulse_result.components_mm_day[6:, :, 1].sum())
    checks = {
        "lambda_zero_exact_parent": identity_error == 0.0,
        "tensor_lambda_gradient_finite": bool(torch.isfinite(raw_lambda.grad)),
        "tensor_lambda_gradient_nonzero": abs(gradient) > 1.0e-12,
        "mass_error_le_1e_10_mm": float(learnable.maximum_mass_error_mm) <= 1.0e-10,
        "lower_store_balance_le_1e_10_mm": lower_balance_error <= 1.0e-10,
        "component_closure_le_1e_12_mm": component_closure <= 1.0e-12,
        "states_and_fluxes_nonnegative": all_nonnegative,
        "pulse_has_delayed_slow_response": delayed_slow > 0.0,
    }
    report = {
        "stage": "20260828_5",
        "status": "PASS_LEARNABLE_OPERATOR_PREFLIGHT" if all(checks.values()) else "FAIL_LEARNABLE_OPERATOR_PREFLIGHT",
        "metrics": {
            "lambda_zero_identity_error": identity_error,
            "lambda_gradient": gradient,
            "mass_error_mm": float(learnable.maximum_mass_error_mm),
            "lower_store_balance_error_mm": lower_balance_error,
            "component_closure_mm": component_closure,
            "pulse_delayed_slow_volume_mm": delayed_slow,
        },
        "checks": checks,
    }
    path = RUN / "reports" / "learnable_operator_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
