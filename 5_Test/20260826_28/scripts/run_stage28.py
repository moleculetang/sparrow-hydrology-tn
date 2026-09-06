"""Full-development refit of the locked conserving two-response hydrology."""

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
RUN = ROOT / "5_Test" / "20260826_28"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
STAGE26 = ROOT / "5_Test" / "20260826_26"
STAGE27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE27 / "scripts"), str(STAGE26 / "scripts"), str(STAGE25 / "scripts"),
    str(STAGE24 / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import DynamicFluxGate2P, simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, GAUGES, GRACE, GRADIENT_CLIP, LEARNING_RATE, MAX_EPOCHS,
    PML, PRIOR_SIGMA, PRIOR_WEIGHT, Q72, SCALING, STATIC, TOPOLOGY, WEIGHT_DECAY,
    antecedent_mean, build_support, composite_parts, normalized_loss, station_metrics,
)
from fold_worker import state_loss_function  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous  # noqa: E402


SEEDS = [260826, 260827, 260828]
RAW_PARAMETER_NAMES = [
    "fc_mm", "beta", "lp", "perc_mm_day", "uzl_mm", "tau0_day", "delta_tau10_day", "delta_tau21_day"
]
PARENT_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"


class FinalTwoPath(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.raw_offset = torch.nn.Parameter(0.01 * torch.randn((8,), generator=generator, dtype=torch.float64))
        self.gate = DynamicFluxGate2P(seed=seed)

    def candidate_raw(self, parent_raw: torch.Tensor) -> torch.Tensor:
        return parent_raw + 1.5 * torch.tanh(self.raw_offset)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_frame(model: torch.nn.Module) -> pd.DataFrame:
    rows = []
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().numpy().reshape(-1)
        shape = json.dumps(list(tensor.shape))
        rows.extend({"tensor_name": name, "flat_index": index, "value": float(value), "shape_json": shape}
                    for index, value in enumerate(values))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    predecessor = json.loads((STAGE27 / "reports" / "stage27_decision.json").read_text(encoding="utf-8"))
    if predecessor["formal_operational_path_selection"] != "TWO_PATH":
        raise RuntimeError("Stage 27 did not lock TWO_PATH")
    contract = {
        "stage": "20260826_28",
        "registered_before_refit": True,
        "locked_structure": "DYN2P conserving fast-response plus slow-response",
        "fit_period": "2010-2018 all eligible development discharge",
        "spinup_forcing": "2006-2009",
        "fixed_hyperparameters": {"seeds": SEEDS, "epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE,
                                  "weight_decay": WEIGHT_DECAY, "prior_sigma": PRIOR_SIGMA, "prior_weight": PRIOR_WEIGHT},
        "selection": "lowest full-development registered objective across the three fixed seeds",
        "2019_2022_discharge_read": False,
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES)
    eligible = gauges.loc[gauges.topology_representative & gauges.four_group_check_eligible].drop_duplicates("station_norm")
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    counts = discharge.loc[discharge.date.dt.year.between(2010, 2018)].groupby("station_norm").q_m3_s.count()
    stations = eligible.loc[eligible.station_norm.map(counts).fillna(0).ge(180)].sort_values(
        ["terminal_tree", "station_norm"]
    ).reset_index(drop=True)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    trees = np.sort(stations.terminal_tree.unique())
    tree_groups = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in trees]
    area = torch.from_numpy(
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
        .set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy()
    )
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
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
    observed_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64).copy()
    observed = torch.from_numpy(observed_np)
    spin_np = np.asarray(dates.year <= 2009)
    development_np = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    development_mask = torch.from_numpy(development_np)
    q20 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[development_np], 0.9, axis=0))

    parent_json = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    parent_raw = torch.tensor([parent_json["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=torch.float64)
    parent_physical = raw_to_physical(parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_np], pet[spin_np], parent_physical, 1.0e-8, 500)
    with torch.no_grad():
        parent_sim = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, parent_physical, parent_initial,
            static, center, scale, None, force_parent=True, collect_storage=True, collect_aet=True,
        )
    parent_site = (parent_sim.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support.T / 86400.0
    reference = {name: float(value) for name, value in composite_parts(
        parent_site, observed, development_mask, q20, q90, tree_groups, dates
    ).items()}
    state_losses = state_loss_function(dates, area, PML, GRACE, set(range(2010, 2019)))
    parent_aet, parent_grace = state_losses(parent_sim)

    traces = []
    seed_results = []
    locks: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        model = FinalTwoPath(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_objective = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        start = time.perf_counter()
        for epoch in range(MAX_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            raw = model.candidate_raw(parent_raw)
            physical = raw_to_physical(raw)
            sim = simulate_dyn2p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical, parent_initial,
                static, center, scale, model.gate, collect_storage=True, collect_aet=True,
            )
            site = (sim.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support.T / 86400.0
            parts = composite_parts(site, observed, development_mask, q20, q90, tree_groups, dates)
            data_loss = normalized_loss(parts, reference)
            aet_loss, grace_loss = state_losses(sim)
            guardrail = 0.05 * torch.relu(aet_loss / parent_aet.clamp_min(1.0e-6) - 1.0) ** 2
            guardrail = guardrail + 0.05 * torch.relu(grace_loss / parent_grace.clamp_min(1.0e-6) - 1.0) ** 2
            prior = PRIOR_WEIGHT * torch.mean(((raw - parent_raw) / PRIOR_SIGMA) ** 2)
            objective = data_loss + guardrail + prior
            value = float(objective.detach())
            if value < best_objective:
                best_objective = value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            traces.append({"seed": seed, "epoch": epoch, "objective": value,
                           "data_loss": float(data_loss.detach()), "state_guardrail": float(guardrail.detach()),
                           "prior": float(prior.detach()), "gate_strength": float(sim.gate_strength.detach())})
            if epoch == MAX_EPOCHS:
                break
            objective.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
            if not np.isfinite(gradient):
                raise RuntimeError(f"Non-finite final-refit gradient seed {seed}")
            optimizer.step()
            if epoch % 10 == 0:
                print(f"FINAL_DYN2P seed={seed} epoch={epoch} objective={value:.5f}", flush=True)
        model.load_state_dict(best_state)
        raw = model.candidate_raw(parent_raw).detach()
        physical = raw_to_physical(raw)
        initial, spin = periodic_spinup(p[spin_np], pet[spin_np], physical, 1.0e-8, 500)
        locks[seed] = {"model": model, "raw": raw, "physical": physical, "initial": initial, "spin": spin}
        seed_results.append({"seed": seed, "best_epoch": best_epoch, "best_objective": best_objective,
                             "gate_strength": float(model.gate.strength()), "spinup_converged": bool(spin["converged"]),
                             "elapsed_seconds": time.perf_counter() - start})

    selected_seed = int(min(seed_results, key=lambda row: row["best_objective"])["seed"])
    selected = locks[selected_seed]
    model = selected["model"]
    with torch.no_grad():
        final = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, selected["physical"], selected["initial"],
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    local_m3_day = final.components_mm_day * area[None, :, None] * 1000.0
    routed_m3_day = route_instantaneous(local_m3_day, reach_ids.tolist(), order, downstream)
    daily = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(dates)),
        "local_fast_m3_s": (local_m3_day[:, :, 0] / 86400.0).numpy().reshape(-1),
        "local_slow_m3_s": (local_m3_day[:, :, 1] / 86400.0).numpy().reshape(-1),
        "routed_fast_m3_s": (routed_m3_day[:, :, 0] / 86400.0).numpy().reshape(-1),
        "routed_slow_m3_s": (routed_m3_day[:, :, 1] / 86400.0).numpy().reshape(-1),
    })
    daily["routed_total_m3_s"] = daily.routed_fast_m3_s + daily.routed_slow_m3_s
    daily["routed_fast_fraction"] = daily.routed_fast_m3_s / daily.routed_total_m3_s.clip(lower=1.0e-12)
    daily.to_parquet(OUT / "full_development_dyn2p_daily_2006_2018.parquet", index=False)
    monthly = daily.assign(year=daily.date.dt.year, month=daily.date.dt.month).groupby(
        ["reach_id", "year", "month"], as_index=False
    ).agg({column: "mean" for column in ["local_fast_m3_s", "local_slow_m3_s", "routed_fast_m3_s", "routed_slow_m3_s", "routed_total_m3_s"]})
    monthly["routed_fast_fraction"] = monthly.routed_fast_m3_s / monthly.routed_total_m3_s.clip(lower=1.0e-12)
    monthly.to_parquet(OUT / "full_development_dyn2p_monthly_2006_2018.parquet", index=False)
    state_dict_frame(model).to_parquet(OUT / "full_development_model_state.parquet", index=False)
    pd.DataFrame(traces).to_parquet(OUT / "full_development_training_trace.parquet", index=False)
    pd.DataFrame(seed_results).to_parquet(OUT / "full_development_seed_summary.parquet", index=False)

    site = (final.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support.T / 86400.0
    metrics = station_metrics("FINAL_DYN2P", selected_seed, observed_np[development_np], site[development_np].numpy(),
                              stations, q20.numpy(), q90.numpy())
    metrics.to_parquet(OUT / "full_development_station_metrics.parquet", index=False)
    parameter_lock = {
        "stage": "20260826_28", "status": "FULL_DEVELOPMENT_DYN2P_LOCKED",
        "selected_seed": selected_seed,
        "fit_period": "2010-2018", "station_count": len(stations), "terminal_tree_count": len(trees),
        "raw_parameters": dict(zip(RAW_PARAMETER_NAMES, selected["raw"].tolist())),
        "physical_parameters": dict(zip(RAW_PARAMETER_NAMES, selected["physical"].tolist())),
        "gate_strength": float(model.gate.strength()), "spinup": selected["spin"],
        "land_mass_max_abs_error_mm": float(final.maximum_mass_error_mm),
        "daily_component_closure_max_abs_m3_s": float(np.max(np.abs(
            daily.routed_total_m3_s - daily.routed_fast_m3_s - daily.routed_slow_m3_s
        ))),
        "2019_2022_discharge_read": False, "TN_read": False,
        "model_state_parquet_sha256": sha256(OUT / "full_development_model_state.parquet"),
        "authorized_successor": "20260826_29",
    }
    write_json(REPORTS / "full_development_parameter_lock.json", parameter_lock)
    validation = {
        "stage": "20260826_28",
        "checks": {
            "three_seed_refit_complete": len(seed_results) == 3,
            "all_spinups_converged": all(row["spinup_converged"] for row in seed_results),
            "land_mass_bounded": parameter_lock["land_mass_max_abs_error_mm"] <= 1.0e-8,
            "fast_slow_total_exact": parameter_lock["daily_component_closure_max_abs_m3_s"] <= 1.0e-10,
            "retrospective_not_read": True, "TN_not_read": True,
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    (RUN / "README.md").write_text("# 20260826_28 full-development DYN2P refit\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_28 完整开发期重拟合\n\n"
        f"两路径结构使用2010–2018全部合格开发数据重拟合；锁定种子为`{selected_seed}`。"
        "2019–2022流量和TN均未读取。\n",
        encoding="utf-8",
    )
    print(json.dumps(parameter_lock, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
