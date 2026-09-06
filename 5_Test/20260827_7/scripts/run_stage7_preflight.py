"""Preflight tests for the registered state-consistent SIG2P-S/P operator."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_7"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE5 = ROOT / "5_Test" / "20260827_5"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [str(RUN / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts")]

from state_consistent_sig2p import simulate_state_consistent_dyn2p  # noqa: E402
from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage25 import FORCING, SCALING, STATIC, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["program_amendment"]["old_baseline_mutation_forbidden"] is not True:
        raise RuntimeError("Program amendment is incomplete")

    selected_seed = 260827
    model_path = STAGE2 / "outputs" / f"dyn2p_alpha05_seed_{selected_seed}_lock.pt"
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(selected_seed)
    model.load_state_dict(saved["model_state"])
    model.eval()
    physical = raw_to_physical(saved["raw_parameters"].to(torch.float64))

    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
    static_all = torch.from_numpy(
        pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy()
    )

    # Exact parent identity on real forcings and all 230 reaches.
    dates = pd.date_range("2006-01-01", "2006-03-31", freq="D")
    forcing = pd.read_parquet(
        FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"]
    )
    forcing.date = pd.to_datetime(forcing.date)
    reach_ids = np.arange(1, 231, dtype=int)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    initial = torch.zeros((230, 3), dtype=torch.float64)
    score = torch.linspace(-2.0, 2.0, 230, dtype=torch.float64)
    with torch.no_grad():
        parent = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static_all, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
        identity = simulate_state_consistent_dyn2p(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static_all, center, scale, model.gate, score, 0.0,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    exact_components = bool(torch.equal(parent.components_mm_day, identity.components_mm_day))
    exact_states = bool(torch.equal(parent.storage_mm, identity.storage_mm))
    exact_final = bool(torch.equal(parent.final_state_mm, identity.final_state_mm))

    # Controlled two-Reach pulse for direction and storage-memory semantics.
    n_time = 40
    p_syn = torch.zeros((n_time, 2), dtype=torch.float64)
    pet_syn = torch.zeros_like(p_syn)
    api_syn = torch.zeros_like(p_syn)
    sin_syn = torch.zeros(n_time, dtype=torch.float64)
    cos_syn = torch.ones(n_time, dtype=torch.float64)
    initial_syn = torch.tensor([[100.0, 50.0, 0.0], [100.0, 50.0, 0.0]], dtype=torch.float64)
    static_syn = torch.zeros((2, static_all.shape[1]), dtype=torch.float64)
    scores_syn = torch.tensor([1.0, -1.0], dtype=torch.float64)
    physical_syn = physical.unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        base_syn = simulate_state_consistent_dyn2p(
            p_syn, pet_syn, api_syn, api_syn, sin_syn, cos_syn, physical_syn, initial_syn,
            static_syn, center, scale, None, torch.zeros(2), 0.0, force_parent=True,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
        adjusted_syn = simulate_state_consistent_dyn2p(
            p_syn, pet_syn, api_syn, api_syn, sin_syn, cos_syn, physical_syn, initial_syn,
            static_syn, center, scale, None, scores_syn, 0.5, force_parent=True,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    p0 = base_syn.percolation_to_lower_mm_day[0]
    p1 = adjusted_syn.percolation_to_lower_mm_day[0]
    positive_direction = bool(p1[0] > p0[0])
    negative_direction = bool(p1[1] < p0[1])
    fast_closure = float(torch.max(torch.abs(
        adjusted_syn.components_mm_day[:, :, 0]
        + adjusted_syn.percolation_to_lower_mm_day
        - adjusted_syn.unadjusted_fast_mm_day
        - adjusted_syn.unadjusted_percolation_mm_day
    )))
    multiday_release = bool(torch.count_nonzero(adjusted_syn.components_mm_day[1:, 0, 1] > 1.0e-12) >= 2)
    lower_storage_created = bool(adjusted_syn.storage_mm[0, 0, 2] > 0.0)
    nonnegative = bool(
        torch.all(identity.components_mm_day >= 0)
        and torch.all(identity.storage_mm >= 0)
        and torch.all(adjusted_syn.components_mm_day >= 0)
        and torch.all(adjusted_syn.storage_mm >= 0)
        and torch.all(adjusted_syn.percolation_to_lower_mm_day >= 0)
    )
    maximum_mass_error = max(float(identity.maximum_mass_error_mm), float(adjusted_syn.maximum_mass_error_mm))

    checks = {
        "lambda_zero_exact_components": exact_components,
        "lambda_zero_exact_states": exact_states,
        "lambda_zero_exact_final_state": exact_final,
        "positive_score_increases_percolation": positive_direction,
        "negative_score_decreases_percolation": negative_direction,
        "fast_percolation_group_closure_le_1e_10": fast_closure <= 1.0e-10,
        "lower_storage_created_by_adjusted_percolation": lower_storage_created,
        "lower_store_pulse_has_multiday_release": multiday_release,
        "all_states_and_fluxes_nonnegative": nonnegative,
        "land_mass_error_le_1e_10_mm": maximum_mass_error <= 1.0e-10,
        "2019_2022_observations_not_read": True,
        "four_station_observations_not_read": True,
        "TN_not_read": True,
    }
    decision = {
        "stage": "20260827_7",
        "status": "PASS_STATE_OPERATOR_PREFLIGHT" if all(checks.values()) else "FAIL_STATE_OPERATOR_PREFLIGHT",
        "checks": checks,
        "metrics": {
            "maximum_mass_error_mm": maximum_mass_error,
            "fast_percolation_group_closure_max_abs_mm": fast_closure,
            "synthetic_parent_percolation_positive_reach_mm": float(p0[0]),
            "synthetic_adjusted_percolation_positive_reach_mm": float(p1[0]),
            "synthetic_parent_percolation_negative_reach_mm": float(p0[1]),
            "synthetic_adjusted_percolation_negative_reach_mm": float(p1[1]),
        },
        "input_hashes": {
            "contract": sha256(RUN / "experiment_contract.json"),
            "operator": sha256(RUN / "scripts" / "state_consistent_sig2p.py"),
            "parent_model": sha256(model_path),
            "forcing": sha256(FORCING),
        },
        "authorized_successor": "20260827_8" if all(checks.values()) else None,
    }
    write_json(REPORTS / "stage7_preflight_decision.json", decision)
    write_json(REPORTS / "validation.json", {"all_checks_pass": all(checks.values()), "checks": checks})
    pd.DataFrame({
        "reach_role": ["positive_score", "negative_score"],
        "regionalized_score": scores_syn.numpy(),
        "parent_percolation_mm_day": p0.numpy(),
        "adjusted_percolation_mm_day": p1.numpy(),
        "adjusted_lower_storage_end_day1_mm": adjusted_syn.storage_mm[0, :, 2].numpy(),
    }).to_parquet(OUT / "synthetic_operator_diagnostics.parquet", index=False)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
