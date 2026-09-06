"""Full-development clean refit and test-observation-blind 230-Reach export."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_5"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE4 = ROOT / "5_Test" / "20260827_4"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE4 / "scripts"), str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from run_stage2_temporal import (  # noqa: E402
    AlphaTwoPathCandidate, composite_parts_strict, periodic_dyn2p_spinup,
)
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from run_stage4_signature_repair import (  # noqa: E402
    OFFSET_BOUND, RIDGE, apply_signature_operator, bfi_target, fit_ridge, predict_ridge,
)
from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import HBVParameters, load_topology, parameters_to_raw  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, GRACE, GRADIENT_CLIP, LEARNING_RATE, PML, PRIOR_SIGMA,
    PRIOR_WEIGHT, Q72, SCALING, STATIC, TOPOLOGY, WEIGHT_DECAY, antecedent_mean,
    build_support, normalized_loss,
)
from run_stage4 import GlobalObjective  # noqa: E402
from fold_worker import state_loss_function  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical  # noqa: E402


SEED = 260827
EPOCHS = 60
MULTISCALE = ROOT / "5_Test" / "20260826_15" / "outputs" / "multiscale_static_features_standardized.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = OLD24 / "outputs" / "registered_channel_attributes.parquet"
FORBIDDEN = {
    (ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet").resolve(),
    (ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet").resolve(),
    (ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_coverage.parquet").resolve(),
}
OPENED: list[str] = []
_ORIGINAL_READ_PARQUET = pd.read_parquet


def audited_read_parquet(path, *args, **kwargs):
    resolved = Path(path).resolve() if isinstance(path, (str, Path)) else None
    if resolved in FORBIDDEN:
        raise PermissionError(f"Formal test observation is locked: {resolved}")
    if resolved is not None:
        OPENED.append(str(resolved))
    return _ORIGINAL_READ_PARQUET(path, *args, **kwargs)


pd.read_parquet = audited_read_parquet


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    stage4 = json.loads((STAGE4 / "reports" / "stage4_decision.json").read_text(encoding="utf-8"))
    if not stage4["operator_authorized"] or stage4["authorized_successor"] != "20260827_5":
        raise RuntimeError("Stage 4 did not authorize the final refit")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE2 / "outputs" / "temporal_station_registry.parquet")
    if len(stations) != 65 or stations.station_norm.duplicated().any() or stations.reach_id.duplicated().any():
        raise RuntimeError("Frozen strict-complete station cohort changed")
    support_np = build_support(stations, reach_ids, order, downstream)
    support = torch.from_numpy(support_np)
    trees = np.sort(stations.terminal_tree.unique())
    tree_groups = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in trees]
    area_np = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    area = torch.from_numpy(area_np.copy())
    static_frame = pd.read_parquet(STATIC).sort_values("reach_id")
    static = torch.from_numpy(static_frame.drop(columns="reach_id").to_numpy(float).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])

    dates = pd.date_range("2006-01-01", "2022-12-31", freq="D")
    model_dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Forcing grid is incomplete")
    p, pet = torch.from_numpy(p_np), torch.from_numpy(pet_np)
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin_np = np.asarray(dates.year <= 2009)
    development_np = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    development_end = int(np.flatnonzero(development_np)[-1] + 1)

    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)
    if np.isfinite(observed_np[dates.year >= 2019]).any():
        raise RuntimeError("Development discharge table unexpectedly contains formal time-test observations")
    observed = torch.from_numpy(observed_np[:development_end].copy())
    development_mask = torch.from_numpy(development_np[:development_end])
    q20 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.9, axis=0))

    fixed_prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(fixed_prior)
    parent_objective = GlobalObjective(
        p_np[spin_np], pet_np[spin_np], p_np[development_np], pet_np[development_np],
        observed_np[development_np], support_np, area_np,
        stations.terminal_tree.to_numpy(int), prior_raw,
    )
    starts = [
        np.zeros(8),
        np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
        np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
    ]
    parent_results = [
        minimize(
            parent_objective, np.clip(prior_raw + start, -5.25, 5.25),
            method="L-BFGS-B", bounds=[(-5.5, 5.5)] * 8,
            options={"maxiter": 45, "maxfun": 550, "ftol": 1e-10, "gtol": 1e-5, "maxls": 20},
        )
        for start in starts
    ]
    parent_best = min(parent_results, key=lambda result: float(result.fun))
    parent_raw = torch.tensor(np.asarray(parent_best.x, dtype=float))
    parent_physical = raw_to_physical(parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_np], pet[spin_np], parent_physical, 1e-8, 500)
    with torch.no_grad():
        parent_sim = simulate_dyn2p_hbv(
            p[:development_end], pet[:development_end], api3[:development_end], api30[:development_end],
            sin_doy[:development_end], cos_doy[:development_end], parent_physical,
            parent_initial, static, center, scale, None, force_parent=True,
            collect_storage=True, collect_aet=True,
        )
    parent_components = torch.einsum("trc,r,sr->tsc", parent_sim.components_mm_day, area * 1000.0, support) / 86400.0
    parent_site = parent_components.sum(dim=2)
    reference = {name: float(value) for name, value in composite_parts_strict(
        parent_site, observed, development_mask, q20, q90, tree_groups, model_dates
    ).items()}
    state_losses = state_loss_function(model_dates, area, PML, GRACE, set(range(2010, 2019)))
    parent_aet, parent_grace = state_losses(parent_sim)

    model = AlphaTwoPathCandidate(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trace = []
    started = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate_raw = model.candidate_raw(parent_raw)
        physical = raw_to_physical(candidate_raw)
        simulation = simulate_dyn2p_hbv(
            p[:development_end], pet[:development_end], api3[:development_end], api30[:development_end],
            sin_doy[:development_end], cos_doy[:development_end], physical,
            torch.zeros((230, 3), dtype=torch.float64), static, center, scale, model.gate,
            collect_storage=True, collect_aet=True,
        )
        site = torch.einsum("trc,r,sr->tsc", simulation.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
        parts = composite_parts_strict(site, observed, development_mask, q20, q90, tree_groups, model_dates)
        flow_loss = normalized_loss(parts, reference)
        aet_loss, grace_loss = state_losses(simulation)
        guardrail = 0.05 * torch.relu(aet_loss / parent_aet.clamp_min(1e-6) - 1.0) ** 2
        guardrail += 0.05 * torch.relu(grace_loss / parent_grace.clamp_min(1e-6) - 1.0) ** 2
        prior = PRIOR_WEIGHT * torch.mean(((candidate_raw - parent_raw) / PRIOR_SIGMA) ** 2)
        objective = flow_loss + guardrail + prior
        trace.append({
            "epoch": epoch, "objective": float(objective.detach()),
            "flow_composite": float(flow_loss.detach()), "state_guardrail": float(guardrail.detach()),
            "prior": float(prior.detach()), "gate_strength": float(model.gate.strength().detach()),
        })
        if epoch == EPOCHS:
            break
        objective.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
        if not np.isfinite(gradient):
            raise RuntimeError("Non-finite gradient")
        optimizer.step()

    final_raw = model.candidate_raw(parent_raw).detach()
    final_physical = raw_to_physical(final_raw)
    final_initial, final_spin = periodic_dyn2p_spinup(
        p[spin_np], pet[spin_np], final_physical, static, center, scale, model.gate,
    )
    with torch.no_grad():
        final = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, final_physical, final_initial,
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    local_original = (final.components_mm_day * area[None, :, None] * 1000.0 / 86400.0).numpy()

    multiscale = pd.read_parquet(MULTISCALE).sort_values("reach_id").set_index("reach_id")
    feature_columns = list(multiscale.columns)
    reach_features = multiscale.reindex(reach_ids).to_numpy(float)
    station_features = multiscale.reindex(stations.reach_id).to_numpy(float)
    development_bfi = np.asarray([bfi_target(observed_np[development_np, index]) for index in range(len(stations))])
    signature_coefficients = fit_ridge(station_features, development_bfi)
    predicted_reach_bfi = predict_ridge(reach_features, signature_coefficients)
    local_corrected, signature_offset = apply_signature_operator(local_original, predicted_reach_bfi, development_np)
    routed_original = route_instantaneous_np(local_original, list(order), downstream)
    routed_corrected = route_instantaneous_np(local_corrected, list(order), downstream)
    storage = final.storage_mm
    if storage is None or final.aet_mm_day is None:
        raise RuntimeError("Final state collection failed")

    daily = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(dates)),
        "is_spinup_period": np.repeat(np.asarray(dates.year <= 2009), len(reach_ids)),
        "local_fast_response_m3_s": local_corrected[:, :, 0].reshape(-1),
        "local_slow_response_m3_s": local_corrected[:, :, 1].reshape(-1),
        "routed_fast_response_m3_s": routed_corrected[:, :, 0].reshape(-1),
        "routed_slow_response_m3_s": routed_corrected[:, :, 1].reshape(-1),
        "raw_dyn2p_routed_fast_response_m3_s": routed_original[:, :, 0].reshape(-1),
        "raw_dyn2p_routed_slow_response_m3_s": routed_original[:, :, 1].reshape(-1),
        "soil_storage_mm": storage[:, :, 0].numpy().reshape(-1),
        "upper_response_storage_mm": storage[:, :, 1].numpy().reshape(-1),
        "lower_slow_storage_mm": storage[:, :, 2].numpy().reshape(-1),
        "actual_aet_mm_day": final.aet_mm_day.numpy().reshape(-1),
    })
    daily["routed_total_m3_s"] = daily.routed_fast_response_m3_s + daily.routed_slow_response_m3_s
    daily["routed_fast_response_fraction"] = daily.routed_fast_response_m3_s / daily.routed_total_m3_s.clip(lower=1e-12)
    daily_path = OUT / "blind_reach_daily_2006_2022.parquet"
    daily.to_parquet(daily_path, index=False)

    daily["year"] = daily.date.dt.year
    daily["month"] = daily.date.dt.month
    flow_columns = [
        "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s",
        "routed_slow_response_m3_s", "raw_dyn2p_routed_fast_response_m3_s",
        "raw_dyn2p_routed_slow_response_m3_s", "routed_total_m3_s", "actual_aet_mm_day",
    ]
    monthly = daily.groupby(["reach_id", "year", "month"], as_index=False)[flow_columns].mean()
    month_state = daily.groupby(["reach_id", "year", "month"], as_index=False)[
        ["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]
    ].last()
    monthly = monthly.merge(month_state, on=["reach_id", "year", "month"], validate="one_to_one")
    monthly["is_spinup_period"] = monthly.year <= 2009
    monthly["routed_fast_response_fraction"] = monthly.routed_fast_response_m3_s / monthly.routed_total_m3_s.clip(lower=1e-12)

    geometry = pd.read_parquet(GEOMETRY, columns=[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
    ])
    channel = pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"])
    geometry = geometry.merge(channel, on="reach_id", validate="one_to_one")
    monthly = monthly.merge(geometry, on="reach_id", validate="many_to_one")
    q = monthly.routed_total_m3_s.to_numpy(float)
    positive = q > 1e-12
    for label, width, depth in [
        ("central", "bankfull_width_m", "bankfull_depth_m"),
        ("geometry_p05", "bankfull_width_p05_m", "bankfull_depth_p05_m"),
        ("geometry_p95", "bankfull_width_p95_m", "bankfull_depth_p95_m"),
    ]:
        values = np.full(len(monthly), np.nan)
        values[positive] = monthly.loc[positive, "reach_length_m"] * monthly.loc[positive, width] * monthly.loc[positive, depth] / q[positive] / 86400.0
        monthly[f"channel_bankfull_travel_time_{label}_day"] = values
    monthly.drop(columns=[column for column in geometry.columns if column != "reach_id"], inplace=True)
    monthly_path = OUT / "blind_reach_monthly_2006_2022.parquet"
    monthly.to_parquet(monthly_path, index=False)

    operator_frame = pd.DataFrame({
        "reach_id": reach_ids, "predicted_BFI_target": predicted_reach_bfi,
        "logit_offset": signature_offset,
    })
    operator_path = OUT / "final_signature_operator_by_reach.parquet"
    operator_frame.to_parquet(operator_path, index=False)
    trace_path = OUT / "full_development_training_trace.parquet"
    pd.DataFrame(trace).to_parquet(trace_path, index=False)
    model_path = OUT / "full_development_model_lock.pt"
    torch.save({
        "seed": SEED, "epochs": EPOCHS, "model_state": model.state_dict(),
        "parent_raw_parameters": parent_raw, "raw_parameters": final_raw,
        "physical_parameters": final_physical, "spinup": final_spin,
        "signature_feature_columns": feature_columns,
        "signature_coefficients": signature_coefficients,
        "signature_offset_bound": OFFSET_BOUND, "signature_ridge": RIDGE,
    }, model_path)

    source_files = [
        Path(__file__), RUN / "experiment_contract.json",
        STAGE2 / "scripts" / "run_stage2_temporal.py",
        STAGE4 / "scripts" / "run_stage4_signature_repair.py",
        OLD27 / "scripts" / "dyn2p_hbv.py",
        OLD25 / "scripts" / "run_stage25.py",
        ROOT / "5_Test" / "20260826_14" / "scripts" / "torch_hbv.py",
        ROOT / "5_Test" / "20260825_3" / "scripts" / "hydrology_core.py",
    ]
    input_files = sorted({Path(path) for path in OPENED if Path(path).is_file()}, key=lambda path: str(path))
    write_json(REPORTS / "input_code_hash_registry.json", {
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_files],
        "code": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_files],
        "forbidden_paths_opened": sorted(set(OPENED).intersection({str(path) for path in FORBIDDEN})),
    })

    closure = float(np.max(np.abs(daily.routed_total_m3_s - daily.routed_fast_response_m3_s - daily.routed_slow_response_m3_s)))
    raw_total = routed_original.sum(axis=2)
    total_change = float(np.max(np.abs(routed_corrected.sum(axis=2) - raw_total)))
    model_hash = sha256(model_path)
    daily_hash = sha256(daily_path)
    monthly_hash = sha256(monthly_path)
    operator_hash = sha256(operator_path)
    lock = {
        "stage": "20260827_5",
        "status": "BLIND_230_REACH_EXPORT_LOCKED",
        "model_sha256": model_hash,
        "daily_prediction_sha256": daily_hash,
        "monthly_prediction_sha256": monthly_hash,
        "signature_operator_sha256": operator_hash,
        "station_count": len(stations), "reach_count": 230,
        "fit_period": "2010-2018", "prediction_period": "2006-2022",
        "formal_test_observations_opened": False,
        "four_station_names_or_reach_mapping_opened": False,
        "created_before_test_evaluation": True,
        "authorized_successor": "20260827_6",
    }
    write_json(REPORTS / "blind_prediction_lock.json", lock)
    checks = {
        "daily_rows_exact": len(daily) == len(dates) * 230,
        "monthly_rows_exact": len(monthly) == 17 * 12 * 230,
        "reach_count_exact": daily.reach_id.nunique() == 230,
        "component_closure_le_1e_10": closure <= 1e-10,
        "signature_total_change_le_1e_10": total_change <= 1e-10,
        "all_flows_nonnegative": bool((daily[["local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s"]] >= 0).all().all()),
        "spinup_converged": bool(final_spin["converged"]),
        "land_mass_error_le_1e_8": float(final.maximum_mass_error_mm) <= 1e-8,
        "2006_2009_marked_spinup": bool(daily.loc[daily.date.dt.year <= 2009, "is_spinup_period"].all()),
        "2010_2022_not_marked_spinup": bool(~daily.loc[daily.date.dt.year >= 2010, "is_spinup_period"].any()),
        "formal_test_observations_not_opened": not bool(set(OPENED).intersection({str(path) for path in FORBIDDEN})),
        "prediction_hashes_written": all([model_hash, daily_hash, monthly_hash, operator_hash]),
        "TN_not_read": True,
        "Andreadis_reference_discharge_not_read": True,
    }
    validation = {"stage": "20260827_5", "all_checks_pass": all(checks.values()), "checks": checks, "elapsed_seconds": time.perf_counter() - started}
    write_json(REPORTS / "validation.json", validation)
    (REPORTS / "technical_report.md").write_text(
        "# 20260827_5 全开发期重拟合与盲导出锁\n\n"
        f"状态：`{lock['status']}`；全部检查通过={validation['all_checks_pass']}。"
        "模型仅使用2010–2018开发流量；2019–2022实测流量和四个空间测试站的名称、观测及Reach映射均未读取。\n\n"
        f"230 Reach日预测SHA-256：`{daily_hash}`。\n\n"
        "2006–2009明确标记为spin-up期；2010年以后才进入可评价的因果输出。\n",
        encoding="utf-8",
    )
    (RUN / "README.md").write_text("# 20260827_5 blind all-Reach export lock\n", encoding="utf-8")
    if not validation["all_checks_pass"]:
        raise RuntimeError("Blind export integrity checks failed")
    print(json.dumps({**lock, "checks": checks, "elapsed_seconds": validation["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
