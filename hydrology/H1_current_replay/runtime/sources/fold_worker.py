"""Zero-target-history terminal-tree fold for the conserving DYN3P program."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_26"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
sys.path[:0] = [
    str(STAGE25 / "scripts"),
    str(STAGE24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
    str(ROOT / "5_Test" / "20260825_4" / "scripts"),
]

from fast_route_autograd import route_autograd  # noqa: E402
from hydrology_core import HBVParameters, load_topology, parameters_to_raw  # noqa: E402
from run_stage4 import GlobalObjective  # noqa: E402
from run_stage25 import (  # noqa: E402
    CandidateModel,
    GRACE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PML,
    PRIOR_SIGMA,
    PRIOR_WEIGHT,
    WEIGHT_DECAY,
    antecedent_mean,
    build_support,
    composite_parts,
    correlation_loss,
    normalized_loss,
    station_metrics,
)
from dyn3p_hbv import simulate_dyn3p_hbv  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DISCHARGE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
CHANNEL = STAGE24 / "outputs" / "registered_channel_attributes.parquet"
SCALING = STAGE24 / "reports" / "dynamic_feature_scaling.json"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"

SEED = 260826
MODELS = ["DYN_FLUX", "JOINT_DYN3P"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def state_loss_function(
    dates: pd.DatetimeIndex,
    area: torch.Tensor,
    pml_path: Path,
    grace_path: Path,
    selected_years: set[int] | None = None,
):
    if selected_years is None:
        selected_years = set(range(2010, 2016))
    pml = pd.read_parquet(pml_path)
    reach_ids = np.arange(1, 231, dtype=int)
    pml_basin = (
        pml.loc[pml.year.between(min(selected_years), max(selected_years))]
        .merge(pd.DataFrame({"reach_id": reach_ids, "area": area.numpy()}), on="reach_id")
        .assign(weighted=lambda frame: frame.pml_aet_mm_month * frame.area)
        .groupby(["year", "month"], as_index=False)
        .agg(weighted=("weighted", "sum"), area=("area", "sum"))
    )
    pml_basin["pml"] = pml_basin.weighted / pml_basin.area
    grace = pd.read_parquet(grace_path).loc[:, ["year", "month", "grace_tws_anomaly_mm"]]
    target = pml_basin.merge(grace, on=["year", "month"], how="left").set_index(["year", "month"])

    def evaluate(simulation) -> tuple[torch.Tensor, torch.Tensor]:
        assert simulation.storage_mm is not None and simulation.aet_mm_day is not None
        model_aet: list[torch.Tensor] = []
        model_storage: list[torch.Tensor] = []
        target_aet: list[float] = []
        target_grace: list[float] = []
        storage_months: list[int] = []
        simulation_dates = dates[: simulation.storage_mm.shape[0]]
        periods = simulation_dates.to_period("M")
        for period in periods.unique():
            if period.year not in selected_years:
                continue
            mask_np = np.asarray(periods == period)
            model_aet.append((simulation.aet_mm_day[mask_np].sum(dim=0) * area).sum() / area.sum())
            target_aet.append(float(target.loc[(period.year, period.month), "pml"]))
            grace_value = float(target.loc[(period.year, period.month), "grace_tws_anomaly_mm"])
            if np.isfinite(grace_value):
                last = int(np.flatnonzero(mask_np)[-1])
                model_storage.append((simulation.storage_mm[last].sum(dim=1) * area).sum() / area.sum())
                target_grace.append(grace_value)
                storage_months.append(period.month)
        aet_values = torch.stack(model_aet)
        storage_values = torch.stack(model_storage)
        target_aet_tensor = torch.tensor(target_aet, dtype=torch.float64)
        target_grace_tensor = torch.tensor(target_grace, dtype=torch.float64)
        month_numbers = torch.tensor(storage_months, dtype=torch.int64)
        storage_ds = storage_values.clone()
        grace_ds = target_grace_tensor.clone()
        for month in range(1, 13):
            select = month_numbers == month
            storage_ds[select] -= storage_values[select].mean()
            grace_ds[select] -= target_grace_tensor[select].mean()
        return correlation_loss(aet_values, target_aet_tensor), correlation_loss(storage_ds, grace_ds)

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=int, required=True)
    args = parser.parse_args()
    heldout_tree = int(args.tree)
    output = RUN / "outputs" / "folds" / f"tree_{heldout_tree}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    topo = pd.read_csv(TOPOLOGY).sort_values("hydseq")
    reach_index = {reach: position for position, reach in enumerate(reach_ids.tolist())}
    order_index = torch.tensor([reach_index[int(reach)] for reach in topo.reach_id], dtype=torch.int64)
    downstream_index_np = np.full(230, -1, dtype=np.int64)
    downstream_fraction_np = np.ones(230, dtype=np.float64)
    for row in topo.itertuples():
        if not pd.isna(row.downstream_reach):
            position = reach_index[int(row.reach_id)]
            downstream_index_np[position] = reach_index[int(row.downstream_reach)]
            downstream_fraction_np[position] = float(row.frac)
    downstream_index = torch.from_numpy(downstream_index_np)
    downstream_fraction = torch.from_numpy(downstream_fraction_np)

    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    coverage = discharge.assign(
        period=np.select(
            [
                discharge.date.dt.year.between(2010, 2015),
                discharge.date.dt.year.eq(2016),
                discharge.date.dt.year.between(2017, 2018),
            ],
            ["train", "stop", "eval"],
            default="other",
        )
    ).groupby(["station_norm", "period"]).q_m3_s.count().unstack(fill_value=0)
    complete_names = coverage.index[
        coverage.get("train", 0).ge(2191)
        & coverage.get("stop", 0).ge(366)
        & coverage.get("eval", 0).ge(730)
    ]
    gauges = pd.read_parquet(GAUGES)
    eligible = (
        gauges.loc[gauges.topology_representative & gauges.four_group_check_eligible]
        .drop_duplicates("station_norm")
        .sort_values(["terminal_tree", "station_norm"])
        .reset_index(drop=True)
    )
    train_stations = eligible.loc[
        eligible.station_norm.isin(complete_names) & eligible.terminal_tree.ne(heldout_tree)
    ].reset_index(drop=True)
    target_stations = eligible.loc[eligible.terminal_tree.eq(heldout_tree)].reset_index(drop=True)
    target_counts = discharge.loc[
        discharge.station_norm.isin(target_stations.station_norm)
        & discharge.date.dt.year.between(2010, 2018)
    ].groupby("station_norm").q_m3_s.count()
    target_stations = target_stations.loc[target_stations.station_norm.map(target_counts).fillna(0).ge(180)].reset_index(drop=True)
    if train_stations.empty or target_stations.empty:
        raise RuntimeError(f"Tree {heldout_tree} lacks training or target stations")

    support_train_np = build_support(train_stations, reach_ids, order, downstream)
    support_target_np = build_support(target_stations, reach_ids, order, downstream)
    support_train = torch.from_numpy(support_train_np)
    support_target = torch.from_numpy(support_target_np)
    target_index = torch.tensor([reach_index[int(value)] for value in target_stations.reach_id], dtype=torch.int64)
    target_fraction = torch.from_numpy(target_stations.downstream_fraction_on_reach.to_numpy(np.float64).copy())
    train_tree_values = np.sort(train_stations.terminal_tree.unique())
    train_tree_groups = [
        torch.from_numpy(np.flatnonzero(train_stations.terminal_tree.to_numpy() == tree)) for tree in train_tree_values
    ]

    area_np = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64).copy()
    )
    area = torch.from_numpy(area_np)
    static = torch.from_numpy(
        pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy()
    )
    channel = pd.read_parquet(CHANNEL).sort_values("reach_id")
    geometry_numerator = torch.from_numpy(
        (channel.reach_length_m * channel.bankfull_width_m * channel.bankfull_depth_m / 86400.0)
        .to_numpy(np.float64).copy()
    )
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    dynamic_center = torch.tensor(scaling["center"], dtype=torch.float64)
    dynamic_scale = torch.tensor(scaling["scale"], dtype=torch.float64)

    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    observed_train_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=train_stations.station_norm).to_numpy(np.float64).copy()
    observed_target_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=target_stations.station_norm).to_numpy(np.float64).copy()
    observed_train = torch.from_numpy(observed_train_np)
    spin_np = np.asarray(dates.year <= 2009)
    train_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    stop_np = np.asarray(dates.year == 2016)
    train_end = int(np.flatnonzero(stop_np)[-1] + 1)
    train_mask = torch.from_numpy(train_np[:train_end])
    stop_mask = torch.from_numpy(stop_np[:train_end])
    q20 = torch.from_numpy(np.nanquantile(observed_train_np[train_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_train_np[train_np], 0.9, axis=0))

    fixed_prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(fixed_prior)
    scipy_objective = GlobalObjective(
        p_np[spin_np], pet_np[spin_np], p_np[train_np], pet_np[train_np],
        observed_train_np[train_np], support_train_np, area_np,
        train_stations.terminal_tree.to_numpy(int), prior_raw,
    )
    starts = [
        np.zeros(8),
        np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
        np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
    ]
    results = []
    for start in starts:
        results.append(minimize(
            scipy_objective, np.clip(prior_raw + start, -5.25, 5.25), method="L-BFGS-B",
            bounds=[(-5.5, 5.5)] * 8,
            options={"maxiter": 45, "maxfun": 550, "ftol": 1.0e-10, "gtol": 1.0e-5, "maxls": 20},
        ))
    best_scipy = min(results, key=lambda value: float(value.fun))
    clean_raw = torch.tensor(np.asarray(best_scipy.x, dtype=np.float64))
    clean_physical = raw_to_physical(clean_raw)
    clean_initial, clean_spin = periodic_spinup(p[spin_np], pet[spin_np], clean_physical, 1.0e-8, 500)
    if not clean_spin["converged"]:
        raise RuntimeError("Fold clean parent spin-up failed")

    with torch.no_grad():
        clean_sim = simulate_dyn3p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, clean_physical, clean_initial,
            static, dynamic_center, dynamic_scale, None, force_parent=True,
            collect_storage=True, collect_aet=True,
        )
    clean_local = clean_sim.components_mm_day * area[None, :, None] * 1000.0
    clean_site = clean_local.sum(dim=2) @ support_train.T / 86400.0
    clean_parts = composite_parts(clean_site[:train_end], observed_train[:train_end], train_mask, q20, q90, train_tree_groups, dates[:train_end])
    reference = {name: float(value) for name, value in clean_parts.items()}
    state_losses = state_loss_function(dates, area, PML, GRACE)
    clean_aet_loss, clean_grace_loss = state_losses(clean_sim)

    # Same-objective fold parent: target-tree observations are not present in any tensor above.
    torch.manual_seed(SEED)
    raw_offset = torch.nn.Parameter(0.01 * torch.randn(8, dtype=torch.float64))
    optimizer = torch.optim.AdamW([raw_offset], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_validation = float("inf")
    best_epoch = -1
    best_parent_raw = clean_raw.clone()
    no_improvement = 0
    trace_rows: list[dict[str, object]] = []
    for epoch in range(MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        raw_value = clean_raw + 1.5 * torch.tanh(raw_offset)
        physical = raw_to_physical(raw_value)
        sim = simulate_dyn3p_hbv(
            p[:train_end], pet[:train_end], api3[:train_end], api30[:train_end],
            sin_doy[:train_end], cos_doy[:train_end], physical, clean_initial,
            static, dynamic_center, dynamic_scale, None, force_parent=True,
            collect_storage=True, collect_aet=True,
        )
        site = (sim.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support_train.T / 86400.0
        train_parts = composite_parts(site, observed_train[:train_end], train_mask, q20, q90, train_tree_groups, dates[:train_end])
        stop_parts = composite_parts(site, observed_train[:train_end], stop_mask, q20, q90, train_tree_groups, dates[:train_end])
        train_loss = normalized_loss(train_parts, reference)
        stop_loss = normalized_loss(stop_parts, reference)
        aet_loss, grace_loss = state_losses(sim)
        guardrail = 0.05 * torch.relu(aet_loss / clean_aet_loss.clamp_min(1.0e-6) - 1.0) ** 2
        guardrail = guardrail + 0.05 * torch.relu(grace_loss / clean_grace_loss.clamp_min(1.0e-6) - 1.0) ** 2
        prior = PRIOR_WEIGHT * torch.mean(((raw_value - clean_raw) / PRIOR_SIGMA) ** 2)
        objective = train_loss + guardrail + prior
        stop_value = float(stop_loss.detach())
        if stop_value < best_validation - 1.0e-7:
            best_validation = stop_value
            best_epoch = epoch
            best_parent_raw = raw_value.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        trace_rows.append({
            "heldout_tree": heldout_tree, "model_id": "FOLD_PARENT", "epoch": epoch,
            "train_composite": float(train_loss.detach()), "stop_composite": stop_value,
            "state_guardrail": float(guardrail.detach()), "prior": float(prior.detach()),
        })
        if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
            break
        if epoch == MAX_EPOCHS:
            break
        objective.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_([raw_offset], GRADIENT_CLIP))
        if not np.isfinite(gradient_norm):
            raise RuntimeError("Non-finite fold parent gradient")
        optimizer.step()

    parent_physical = raw_to_physical(best_parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_np], pet[spin_np], parent_physical, 1.0e-8, 500)
    with torch.no_grad():
        parent_sim = simulate_dyn3p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, parent_physical, parent_initial,
            static, dynamic_center, dynamic_scale, None, force_parent=True,
            collect_storage=True, collect_aet=True,
        )
    parent_local = parent_sim.components_mm_day * area[None, :, None] * 1000.0
    parent_train_site = parent_local.sum(dim=2) @ support_train.T / 86400.0
    parent_target_site = parent_local.sum(dim=2) @ support_target.T / 86400.0
    parent_parts = composite_parts(parent_train_site[:train_end], observed_train[:train_end], train_mask, q20, q90, train_tree_groups, dates[:train_end])
    parent_reference = {name: float(value) for name, value in parent_parts.items()}
    parent_aet_loss, parent_grace_loss = state_losses(parent_sim)
    parent_instant = route_instantaneous(parent_local, reach_ids.tolist(), order, downstream)
    q_floor = torch.maximum(
        torch.quantile(parent_instant[train_np].sum(dim=2) / 86400.0, 0.01, dim=0),
        torch.full((230,), 1.0e-6, dtype=torch.float64),
    )

    evaluation_np = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    eval_dates = dates[evaluation_np]
    observed_eval = observed_target_np[evaluation_np]
    q20_target = np.nanquantile(observed_eval, 0.2, axis=0)
    q90_target = np.nanquantile(observed_eval, 0.9, axis=0)
    metrics_frames = [station_metrics("FOLD_PARENT", -1, observed_eval, parent_target_site[evaluation_np].numpy(), target_stations, q20_target, q90_target)]
    prediction_frames = [pd.DataFrame({
        "date": np.repeat(eval_dates.to_numpy(), len(target_stations)),
        "station_norm": np.tile(target_stations.station_norm.to_numpy(), len(eval_dates)),
        "terminal_tree": heldout_tree,
        "model_id": "FOLD_PARENT", "seed": -1,
        "observed_m3_s": observed_eval.reshape(-1),
        "predicted_m3_s": parent_target_site[evaluation_np].numpy().reshape(-1),
    })]
    run_rows = [{
        "heldout_tree": heldout_tree, "model_id": "FOLD_PARENT", "seed": -1,
        "best_epoch": best_epoch, "best_stop_composite": best_validation,
        "train_station_count": len(train_stations), "target_station_count": len(target_stations),
        "target_observations_used_in_training": 0, "spinup_converged": bool(parent_spin["converged"]),
        "land_mass_error_mm": float(parent_sim.maximum_mass_error_mm),
    }]

    for model_id in MODELS:
        torch.manual_seed(SEED)
        model = CandidateModel(model_id, SEED)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_validation = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        no_improvement = 0
        start_time = time.perf_counter()
        for epoch in range(MAX_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            candidate_raw = model.candidate_raw(best_parent_raw)
            physical = raw_to_physical(candidate_raw)
            sim = simulate_dyn3p_hbv(
                p[:train_end], pet[:train_end], api3[:train_end], api30[:train_end],
                sin_doy[:train_end], cos_doy[:train_end], physical, parent_initial,
                static, dynamic_center, dynamic_scale, model.gate, force_parent=False,
                collect_storage=True, collect_aet=True,
            )
            local = sim.components_mm_day * area[None, :, None] * 1000.0
            if model.raw_channel_strength is None:
                site = local.sum(dim=2) @ support_train.T / 86400.0
                route_strength = torch.zeros((), dtype=torch.float64)
            else:
                routed, upstream, _, _ = route_autograd(
                    local, model.raw_channel_strength, order_index, downstream_index,
                    downstream_fraction, geometry_numerator, q_floor,
                )
                train_index = torch.tensor([reach_index[int(value)] for value in train_stations.reach_id], dtype=torch.int64)
                train_fraction = torch.from_numpy(train_stations.downstream_fraction_on_reach.to_numpy(np.float64).copy())
                site = (
                    (1.0 - train_fraction[None, :, None]) * upstream[:, train_index, :]
                    + train_fraction[None, :, None] * routed[:, train_index, :]
                ).sum(dim=2) / 86400.0
                route_strength = torch.sigmoid(model.raw_channel_strength)
            train_parts = composite_parts(site, observed_train[:train_end], train_mask, q20, q90, train_tree_groups, dates[:train_end])
            stop_parts = composite_parts(site, observed_train[:train_end], stop_mask, q20, q90, train_tree_groups, dates[:train_end])
            train_loss = normalized_loss(train_parts, parent_reference)
            stop_loss = normalized_loss(stop_parts, parent_reference)
            aet_loss, grace_loss = state_losses(sim)
            guardrail = 0.05 * torch.relu(aet_loss / parent_aet_loss.clamp_min(1.0e-6) - 1.0) ** 2
            guardrail = guardrail + 0.05 * torch.relu(grace_loss / parent_grace_loss.clamp_min(1.0e-6) - 1.0) ** 2
            prior = PRIOR_WEIGHT * torch.mean(((candidate_raw - best_parent_raw) / PRIOR_SIGMA) ** 2)
            objective = train_loss + guardrail + prior
            stop_value = float(stop_loss.detach())
            if stop_value < best_validation - 1.0e-7:
                best_validation = stop_value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            trace_rows.append({
                "heldout_tree": heldout_tree, "model_id": model_id, "epoch": epoch,
                "train_composite": float(train_loss.detach()), "stop_composite": stop_value,
                "state_guardrail": float(guardrail.detach()), "prior": float(prior.detach()),
                "gate_strength": float(sim.gate_strength.detach()), "routing_strength": float(route_strength.detach()),
            })
            if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                break
            if epoch == MAX_EPOCHS:
                break
            objective.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
            if not np.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite {model_id} gradient")
            optimizer.step()

        model.load_state_dict(best_state)
        candidate_raw = model.candidate_raw(best_parent_raw).detach()
        physical = raw_to_physical(candidate_raw)
        candidate_initial, candidate_spin = periodic_spinup(p[spin_np], pet[spin_np], physical, 1.0e-8, 500)
        with torch.no_grad():
            final_sim = simulate_dyn3p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical, candidate_initial,
                static, dynamic_center, dynamic_scale, model.gate, force_parent=False,
                collect_storage=True, collect_aet=True,
            )
        local = final_sim.components_mm_day * area[None, :, None] * 1000.0
        if model.raw_channel_strength is None:
            target_site = local.sum(dim=2) @ support_target.T / 86400.0
            route_strength_value = 0.0
        else:
            routed, upstream, _, _ = route_autograd(
                local, model.raw_channel_strength.detach(), order_index, downstream_index,
                downstream_fraction, geometry_numerator, q_floor,
            )
            target_site = (
                (1.0 - target_fraction[None, :, None]) * upstream[:, target_index, :]
                + target_fraction[None, :, None] * routed[:, target_index, :]
            ).sum(dim=2) / 86400.0
            route_strength_value = float(torch.sigmoid(model.raw_channel_strength.detach()))
        predicted_eval = target_site[evaluation_np].detach().numpy()
        metrics_frames.append(station_metrics(model_id, SEED, observed_eval, predicted_eval, target_stations, q20_target, q90_target))
        prediction_frames.append(pd.DataFrame({
            "date": np.repeat(eval_dates.to_numpy(), len(target_stations)),
            "station_norm": np.tile(target_stations.station_norm.to_numpy(), len(eval_dates)),
            "terminal_tree": heldout_tree, "model_id": model_id, "seed": SEED,
            "observed_m3_s": observed_eval.reshape(-1), "predicted_m3_s": predicted_eval.reshape(-1),
        }))
        run_rows.append({
            "heldout_tree": heldout_tree, "model_id": model_id, "seed": SEED,
            "best_epoch": best_epoch, "best_stop_composite": best_validation,
            "train_station_count": len(train_stations), "target_station_count": len(target_stations),
            "target_observations_used_in_training": 0,
            "gate_strength": float(model.gate.strength().detach()),
            "routing_strength": route_strength_value,
            "spinup_converged": bool(candidate_spin["converged"]),
            "land_mass_error_mm": float(final_sim.maximum_mass_error_mm),
            "elapsed_seconds": time.perf_counter() - start_time,
        })

    pd.DataFrame(trace_rows).to_parquet(output / "training_trace.parquet", index=False)
    pd.DataFrame(run_rows).to_parquet(output / "run_summary.parquet", index=False)
    pd.concat(metrics_frames, ignore_index=True).to_parquet(output / "station_metrics.parquet", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    write_json(output / "metadata.json", {
        "heldout_tree": heldout_tree,
        "train_stations": train_stations.station_norm.tolist(),
        "target_stations": target_stations.station_norm.tolist(),
        "target_history_used_in_training": False,
        "full_development_lock_used": False,
        "fixed_prior_only": True,
        "models": MODELS,
        "seed": SEED,
    })
    print(f"TREE {heldout_tree} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
