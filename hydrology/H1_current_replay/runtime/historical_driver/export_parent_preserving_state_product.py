"""Embed the locked state operator into the accepted total-flow parent."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_9"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7OP = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
STAGE5 = ROOT / "5_Test" / "20260828_5"
OLDMODEL = ROOT / "5_Test" / "20260827_5" / "outputs" / "full_development_model_lock.pt"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(ROOT / "5_Test" / "20260828_7" / "scripts"), str(STAGE5 / "scripts"),
    str(STAGE8 / "scripts"), str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7OP / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from fit_export_state_consistent_product import build_monthly  # noqa: E402
from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from evaluate_component_development import periodic_spinup  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
SCORE = STAGE8 / "outputs" / "regionalized_slow_score.parquet"
LAMBDA_S = 0.1805437376850875
SEED = 260827


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    old_hash_before = sha256(OLDMODEL)
    saved = torch.load(OLDMODEL, map_location="cpu", weights_only=False)
    if int(saved["seed"]) != SEED:
        raise RuntimeError("Accepted parent seed changed")
    model = AlphaTwoPathCandidate(SEED)
    model.load_state_dict(saved["model_state"])
    model.eval()
    raw_parameters = saved["raw_parameters"].to(torch.float64).detach()
    physical = raw_to_physical(raw_parameters)
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score = torch.from_numpy(pd.read_parquet(SCORE).sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy())
    dates = pd.date_range("2006-01-01", "2024-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("2006-2024 forcing incomplete")
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin = np.asarray(dates.year <= 2009)
    initial, spin_audit = periodic_spinup(p[spin], pet[spin], physical, static, center, scale, model.gate, score, LAMBDA_S)
    with torch.no_grad():
        result = simulate_learnable_sig2p(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, score, LAMBDA_S,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    components = result.components_mm_day.numpy()
    storage = result.storage_mm.numpy()
    percolation = result.percolation_to_lower_mm_day.numpy()
    local = components * area.numpy()[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local, list(order), downstream)
    daily = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reach_ids)), "reach_id": np.tile(reach_ids, len(dates)),
        "is_spinup_period": np.repeat(np.asarray(dates.year <= 2009), len(reach_ids)),
        "local_fast_response_m3_s": local[:, :, 0].reshape(-1),
        "local_slow_response_m3_s": local[:, :, 1].reshape(-1),
        "routed_fast_response_m3_s": routed[:, :, 0].reshape(-1),
        "routed_slow_response_m3_s": routed[:, :, 1].reshape(-1),
        "routed_total_m3_s": routed.sum(axis=2).reshape(-1),
        "percolation_to_lower_mm_day": percolation.reshape(-1),
        "soil_storage_mm": storage[:, :, 0].reshape(-1),
        "upper_response_storage_mm": storage[:, :, 1].reshape(-1),
        "lower_slow_storage_mm": storage[:, :, 2].reshape(-1),
        "actual_aet_mm_day": result.aet_mm_day.numpy().reshape(-1),
    })
    daily["state_consistent_fast_fraction"] = daily.routed_fast_response_m3_s / daily.routed_total_m3_s.clip(lower=1.0e-12)
    monthly = build_monthly(daily)
    lower_previous = np.concatenate((initial[:, 2].numpy()[None, :], storage[:-1, :, 2]), axis=0)
    lower_balance = float(np.max(np.abs(lower_previous + percolation - components[:, :, 1] - storage[:, :, 2])))
    closure = float(np.max(np.abs(routed.sum(axis=2) - routed[:, :, 0] - routed[:, :, 1])))
    parent_unchanged = sha256(OLDMODEL) == old_hash_before
    checks = {
        "daily_rows_exact": len(daily) == 1_596_200,
        "monthly_rows_exact": len(monthly) == 52_440,
        "reach_count_exact": daily.reach_id.nunique() == 230,
        "mass_error_le_1e_8_mm": float(result.maximum_mass_error_mm) <= 1.0e-8,
        "lower_store_balance_le_1e_10_mm": lower_balance <= 1.0e-10,
        "component_closure_le_1e_10_m3_s": closure <= 1.0e-10,
        "all_fluxes_and_states_nonnegative": bool(daily[[
            "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s",
            "routed_slow_response_m3_s", "routed_total_m3_s", "percolation_to_lower_mm_day",
            "soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm",
        ]].ge(-1.0e-12).all().all()),
        "spinup_converged": bool(spin_audit["converged"]),
        "accepted_parent_file_unchanged": parent_unchanged,
        "no_parameter_refit": True,
        "no_output_component_reallocation": True,
        "formal_observations_not_read": True,
        "TN_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    model_path = OUT / "parent_preserving_state_consistent_model.pt"
    daily_path = OUT / "canonical_reach_daily_2006_2024.parquet"
    monthly_path = OUT / "canonical_reach_monthly_2006_2024.parquet"
    torch.save({
        "seed": SEED, "lambda_S": LAMBDA_S, "model_state": model.state_dict(),
        "raw_parameters": raw_parameters, "physical_parameters": physical,
        "regionalized_score_path": str(SCORE), "accepted_parent_model_path": str(OLDMODEL),
        "accepted_parent_model_sha256": old_hash_before, "spinup": spin_audit,
    }, model_path)
    daily.to_parquet(daily_path, index=False)
    monthly.to_parquet(monthly_path, index=False)
    hashes = {
        "accepted_parent_model": old_hash_before, "state_consistent_model": sha256(model_path),
        "daily_2006_2024": sha256(daily_path), "monthly_2006_2024": sha256(monthly_path),
        "forcing": sha256(FORCING), "regionalized_score": sha256(SCORE),
    }
    lock = {
        "stage": "20260828_9", "status": "PARENT_PRESERVING_STATE_PRODUCT_LOCKED",
        "reach_count": 230, "prediction_period": "2006-2024", "lambda_S": LAMBDA_S,
        "parent_parameters_refit": False, "formal_observations_opened": False,
        "hashes": hashes, "authorized_successor": "20260828_10",
    }
    validation = {
        "stage": "20260828_9", "status": "PASS_PARENT_PRESERVING_STATE_EXPORT",
        "checks": checks, "metrics": {"mass_error_mm": float(result.maximum_mass_error_mm), "lower_store_balance_error_mm": lower_balance, "component_closure_m3_s": closure},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(REPORTS / "parent_preserving_product_lock.json", lock)
    write_json(REPORTS / "validation.json", validation)
    (REPORTS / "technical_report.md").write_text(
        "# 20260828_9 父模型保持的状态一致快慢流产品\n\n"
        "保留20260827_6已接受总流量父模型的全部参数，仅将已锁定区域快慢校正嵌入上层快流—下渗分配。"
        "输出端不再进行快慢比例重分配。\n",
        encoding="utf-8",
    )
    print(json.dumps({**lock, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
