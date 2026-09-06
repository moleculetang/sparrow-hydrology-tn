"""Full-development proximal update and blind 2006-2024 all-Reach export."""

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


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_7"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7OP = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
STAGE5 = ROOT / "5_Test" / "20260828_5"
STAGE6 = ROOT / "5_Test" / "20260828_6"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE5 / "scripts"), str(STAGE8 / "scripts"),
    str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7OP / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from train_component_worker import aggregate_fraction, component_parts, tree_balanced_mse  # noqa: E402
from evaluate_component_development import periodic_spinup  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from run_stage8_candidates import aggregate_monthly, composite_parts_available, monthly_station_loss  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support, normalized_loss  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = OLD24 / "outputs" / "registered_channel_attributes.parquet"
EPOCHS = 20
LEARNING_RATE = 0.002
PROXIMAL_WEIGHT = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["year"] = work.date.dt.year
    work["month"] = work.date.dt.month
    means = [
        "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s",
        "routed_total_m3_s", "percolation_to_lower_mm_day", "actual_aet_mm_day",
    ]
    monthly = work.groupby(["reach_id", "year", "month"], as_index=False)[means].mean()
    states = work.groupby(["reach_id", "year", "month"], as_index=False)[
        ["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]
    ].last()
    monthly = monthly.merge(states, on=["reach_id", "year", "month"], validate="one_to_one")
    monthly["is_spinup_period"] = monthly.year <= 2009
    monthly["state_consistent_fast_fraction"] = monthly.routed_fast_response_m3_s / monthly.routed_total_m3_s.clip(lower=1.0e-12)
    geometry = pd.read_parquet(GEOMETRY, columns=[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
    ]).merge(pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"]), on="reach_id", validate="one_to_one")
    monthly = monthly.merge(geometry, on="reach_id", validate="many_to_one")
    positive = monthly.routed_total_m3_s.to_numpy(float) > 1.0e-12
    for label, width, depth in [
        ("central", "bankfull_width_m", "bankfull_depth_m"),
        ("geometry_p05", "bankfull_width_p05_m", "bankfull_depth_p05_m"),
        ("geometry_p95", "bankfull_width_p95_m", "bankfull_depth_p95_m"),
    ]:
        value = np.full(len(monthly), np.nan)
        value[positive] = (
            monthly.loc[positive, "reach_length_m"] * monthly.loc[positive, width]
            * monthly.loc[positive, depth] / monthly.loc[positive, "routed_total_m3_s"] / 86400.0
        )
        monthly[f"channel_bankfull_travel_time_{label}_day"] = value
    monthly.drop(columns=[column for column in geometry.columns if column != "reach_id"], inplace=True)
    return monthly


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    decision6 = json.loads((STAGE6 / "reports" / "component_development_decision.json").read_text(encoding="utf-8"))
    lock6 = json.loads((STAGE6 / "reports" / "component_candidate_lock.json").read_text(encoding="utf-8"))
    if decision6["status"] != "PASS_COMPONENT_SUPERVISED_DEVELOPMENT" or decision6["authorized_successor"] != "20260828_7":
        raise RuntimeError("Stage 6 did not authorize full export")
    seed = int(lock6["selected_seed"])
    lambda_s = float(lock6["selected_lambda_S"])
    gamma_effective = float(lock6["selected_gamma"]) * float(lock6["selected_projection_alpha"])
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    if len(stations) != 91 or stations.reach_id.nunique() != 91:
        raise RuntimeError("91-station registry changed")
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    station_trees = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in sorted(stations.terminal_tree.unique())]
    reach_tree = np.asarray([terminal[int(reach)] for reach in reach_ids])
    reach_trees = [torch.from_numpy(np.flatnonzero(reach_tree == tree)) for tree in sorted(np.unique(reach_tree))]
    dates = pd.date_range("2006-01-01", "2024-12-31", freq="D")
    model_dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("2006-2024 forcing is incomplete")
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    model_end = len(model_dates)
    development_np = np.asarray((model_dates.year >= 2010) & (model_dates.year <= 2018))
    development_mask = torch.from_numpy(development_np)
    spin_np = np.asarray(dates.year <= 2009)
    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score = torch.from_numpy(pd.read_parquet(STAGE8 / "outputs" / "regionalized_slow_score.parquet").sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy())

    daily = pd.concat([
        pd.read_parquet(FIREWALL / "daily_observations_2010_2016.parquet"),
        pd.read_parquet(FIREWALL / "daily_observations_2017_2018.parquet"),
    ], ignore_index=True)
    daily.date = pd.to_datetime(daily.date)
    daily = daily.drop_duplicates(["date", "station_norm"], keep="last")
    observed_np = daily.loc[daily.station_norm.isin(stations.station_norm)].pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=model_dates, columns=stations.station_norm).to_numpy(np.float64)
    observed = torch.from_numpy(observed_np.copy())
    q20 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.9, axis=0))
    periods = pd.period_range("2010-01", "2018-12", freq="M")
    monthly = pd.concat([
        pd.read_parquet(FIREWALL / "monthly_observations_2010_2016.parquet"),
        pd.read_parquet(FIREWALL / "monthly_observations_2017_2018.parquet"),
    ], ignore_index=True).drop_duplicates(["year", "month", "station_norm"], keep="last")
    keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    monthly_np = monthly.loc[monthly.station_norm.isin(stations.station_norm)].pivot(index=["year", "month"], columns="station_norm", values="q_m3s").reindex(index=keys, columns=stations.station_norm).to_numpy(np.float64)
    monthly_observed = torch.from_numpy(monthly_np.copy())
    monthly_mask = torch.ones(len(periods), dtype=torch.bool)

    checkpoint6 = torch.load(Path(lock6["selected_checkpoint"]), map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(seed)
    model.load_state_dict(checkpoint6["model_state"])
    physical_raw = checkpoint6["raw_parameters"].to(torch.float64).detach()
    physical = raw_to_physical(physical_raw)
    initial_gate = {name: value.detach().clone() for name, value in model.gate.named_parameters()}

    teacher_saved = torch.load(STAGE8 / "outputs" / "sig2p_sp_seed_260827_lambda_0p0.pt", map_location="cpu", weights_only=False)
    teacher_model = AlphaTwoPathCandidate(260827)
    teacher_model.load_state_dict(teacher_saved["model_state"])
    teacher_model.eval()
    with torch.no_grad():
        teacher_parent = simulate_learnable_sig2p(
            p[:model_end], pet[:model_end], api3[:model_end], api30[:model_end],
            sin_doy[:model_end], cos_doy[:model_end],
            raw_to_physical(teacher_saved["raw_parameters"].to(torch.float64)),
            torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
            teacher_model.gate, score, 0.0,
        ).components_mm_day
    teacher_total = teacher_parent.sum(dim=2)
    teacher_parent_fraction = teacher_parent[:, :, 1] / teacher_total.clamp_min(1.0e-12)
    teacher_fraction = torch.sigmoid(
        torch.log(teacher_parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8))
        - torch.log1p(-teacher_parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8)) + score[None, :]
    )
    teacher_components = torch.stack((teacher_total * (1.0 - teacher_fraction), teacher_total * teacher_fraction), dim=2)
    teacher_components = torch.where(teacher_total[:, :, None] > 1.0e-12, teacher_components, torch.zeros_like(teacher_components))
    teacher_site = torch.einsum("trc,r,sr->tsc", teacher_parent, area * 1000.0, support).sum(dim=2) / 86400.0
    daily_reference = {name: float(value) for name, value in composite_parts_available(teacher_site, observed, development_mask, q20, q90, station_trees, model_dates).items()}
    monthly_reference = float(monthly_station_loss(aggregate_monthly(teacher_site, model_dates, periods), monthly_observed, monthly_mask, station_trees))
    pm = aggregate_fraction(teacher_parent, model_dates, development_np, "monthly")
    tm = aggregate_fraction(teacher_components, model_dates, development_np, "monthly")
    pa = aggregate_fraction(teacher_parent, model_dates, development_np, "annual")
    ta = aggregate_fraction(teacher_components, model_dates, development_np, "annual")
    component_reference = {"monthly": float(tree_balanced_mse(pm - tm, reach_trees)), "annual": float(tree_balanced_mse(pa - ta, reach_trees))}

    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    trace = []
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        simulation = simulate_learnable_sig2p(
            p[:model_end], pet[:model_end], api3[:model_end], api30[:model_end],
            sin_doy[:model_end], cos_doy[:model_end], physical,
            torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
            model.gate, score, lambda_s,
        )
        site = torch.einsum("trc,r,sr->tsc", simulation.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
        flow = 0.75 * normalized_loss(composite_parts_available(site, observed, development_mask, q20, q90, station_trees, model_dates), daily_reference)
        flow += 0.25 * monthly_station_loss(aggregate_monthly(site, model_dates, periods), monthly_observed, monthly_mask, station_trees) / max(monthly_reference, 1.0e-8)
        component, detail = component_parts(simulation.components_mm_day, teacher_components, model_dates, development_np, reach_trees, component_reference)
        proximal_terms = []
        for name, value in model.gate.named_parameters():
            proximal_terms.append(torch.mean((value - initial_gate[name]) ** 2))
        proximal = torch.stack(proximal_terms).mean()
        objective = flow + gamma_effective * component + PROXIMAL_WEIGHT * proximal
        trace.append({
            "epoch": epoch, "objective": float(objective.detach()), "flow": float(flow.detach()),
            "component": float(component.detach()), "component_monthly_MSE": float(detail["monthly_MSE"].detach()),
            "proximal": float(proximal.detach()), "gate_strength": float(model.gate.strength().detach()),
        })
        if epoch == EPOCHS:
            break
        objective.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0))
        if not np.isfinite(gradient):
            raise RuntimeError("Non-finite full-development gradient")
        optimizer.step()

    model.eval()
    initial, spin_audit = periodic_spinup(p[spin_np], pet[spin_np], physical, static, center, scale, model.gate, score, lambda_s)
    with torch.no_grad():
        final = simulate_learnable_sig2p(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, score, lambda_s,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    components = final.components_mm_day.numpy()
    storage = final.storage_mm.numpy()
    percolation = final.percolation_to_lower_mm_day.numpy()
    local = components * area.numpy()[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local, list(order), downstream)
    daily_product = pd.DataFrame({
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
        "actual_aet_mm_day": final.aet_mm_day.numpy().reshape(-1),
    })
    daily_product["state_consistent_fast_fraction"] = daily_product.routed_fast_response_m3_s / daily_product.routed_total_m3_s.clip(lower=1.0e-12)
    monthly_product = build_monthly(daily_product)

    lower_previous = np.concatenate((initial[:, 2].numpy()[None, :], storage[:-1, :, 2]), axis=0)
    lower_balance_error = float(np.max(np.abs(lower_previous + percolation - components[:, :, 1] - storage[:, :, 2])))
    component_closure = float(np.max(np.abs(routed.sum(axis=2) - routed[:, :, 0] - routed[:, :, 1])))
    checks = {
        "daily_rows_exact": len(daily_product) == 1_596_200,
        "monthly_rows_exact": len(monthly_product) == 52_440,
        "reach_count_exact": daily_product.reach_id.nunique() == 230,
        "date_range_exact": daily_product.date.min() == pd.Timestamp("2006-01-01") and daily_product.date.max() == pd.Timestamp("2024-12-31"),
        "mass_error_le_1e_8_mm": float(final.maximum_mass_error_mm) <= 1.0e-8,
        "lower_store_balance_le_1e_10_mm": lower_balance_error <= 1.0e-10,
        "component_closure_le_1e_10_m3_s": component_closure <= 1.0e-10,
        "all_fluxes_and_states_nonnegative": bool(daily_product[[
            "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s",
            "routed_slow_response_m3_s", "routed_total_m3_s", "percolation_to_lower_mm_day",
            "soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm",
        ]].ge(-1.0e-12).all().all()),
        "spinup_converged": bool(spin_audit["converged"]),
        "2019_2022_observations_not_read": True,
        "four_spatial_station_observations_not_read": True,
        "TN_not_read": True,
        "output_only_component_reallocation_absent": True,
    }
    if not all(checks.values()):
        write_json(REPORTS / "blind_product_validation.json", {"status": "FAIL", "checks": checks})
        raise RuntimeError("Blind product checks failed")
    model_path = OUT / "full_development_state_consistent_model.pt"
    daily_path = OUT / "canonical_reach_daily_2006_2024.parquet"
    monthly_path = OUT / "canonical_reach_monthly_2006_2024.parquet"
    trace_path = OUT / "full_development_training_trace.parquet"
    torch.save({
        "seed": seed, "lambda_S": lambda_s, "projection_alpha": float(lock6["selected_projection_alpha"]),
        "effective_component_weight": gamma_effective, "epochs": EPOCHS,
        "model_state": model.state_dict(), "raw_parameters": physical_raw,
        "physical_parameters": physical, "spinup": spin_audit,
        "regionalized_score_path": str(STAGE8 / "outputs" / "regionalized_slow_score.parquet"),
    }, model_path)
    daily_product.to_parquet(daily_path, index=False)
    monthly_product.to_parquet(monthly_path, index=False)
    pd.DataFrame(trace).to_parquet(trace_path, index=False)
    hashes = {
        "model": sha256(model_path), "daily_2006_2024": sha256(daily_path),
        "monthly_2006_2024": sha256(monthly_path), "training_trace": sha256(trace_path),
        "forcing": sha256(FORCING), "regionalized_score": sha256(STAGE8 / "outputs" / "regionalized_slow_score.parquet"),
    }
    lock = {
        "stage": "20260828_7", "status": "STATE_CONSISTENT_230_REACH_PRODUCT_LOCKED",
        "station_count": 91, "reach_count": 230, "fit_period": "2010-2018",
        "prediction_period": "2006-2024", "selected_seed": seed, "lambda_S": lambda_s,
        "formal_test_observations_opened": False, "four_spatial_station_observations_opened": False,
        "created_before_test_evaluation": True, "hashes": hashes,
        "authorized_successor": "20260828_8",
    }
    validation = {
        "stage": "20260828_7", "status": "PASS_BLIND_PRODUCT_LOCK",
        "checks": checks,
        "metrics": {"mass_error_mm": float(final.maximum_mass_error_mm), "lower_store_balance_error_mm": lower_balance_error, "component_closure_m3_s": component_closure},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(REPORTS / "state_consistent_product_lock.json", lock)
    write_json(REPORTS / "blind_product_validation.json", validation)
    (REPORTS / "technical_report.md").write_text(
        "# 20260828_7 状态一致快慢流水文产品锁\n\n"
        "本产品不再对模型输出做快慢比例后处理。校正发生在上层快流—下渗分配处，"
        "下渗随后进入下层库存，慢流只从该库存释放。2006–2024的230 Reach日/月产品已在读取正式测试观测前锁定。\n",
        encoding="utf-8",
    )
    print(json.dumps({**lock, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
