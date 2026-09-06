from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
S14_1 = TEST / "20260814_1"
S14_6 = TEST / "20260814_6"
S14_9 = TEST / "20260814_9"
S15_1 = TEST / "20260815_1"
S15_2 = TEST / "20260815_2"
S16_1 = TEST / "20260816_1"
S18_1 = TEST / "20260818_1"
S18_6 = TEST / "20260818_6"
S18_7 = TEST / "20260818_7"
S20_1 = TEST / "20260820_1"
S20_2 = TEST / "20260820_2"

MONTHLY_PATH = S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
EARLY_PATH = S15_2 / "outputs" / "pre1961_early_n_mean_by_reach.parquet"
OBS_PATH = S15_1 / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = S15_1 / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = S14_1 / "inputs" / "topology" / "topology_edges.csv"
FORMAL_PATH = S18_6 / "outputs" / "final_formal_model_registry.parquet"
PARENT_PRED_PATH = S18_1 / "outputs" / "readout_level_predictions.parquet"
PARENT_PARAM_PATH = S18_1 / "outputs" / "candidate_fold_readout_parameters.parquet"
PARENT_CORE_PATH = S16_1 / "scripts" / "legacy16_core.py"
GW_TEMP_PATH = S14_9 / "inputs" / "model_ready" / "static" / "groundwater_temperature_benz_2024_by_reach.parquet"
AQ_TEMP_PATH = S14_9 / "inputs" / "model_ready" / "monthly" / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"
STATIC_PATH = S14_9 / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
GEOMETRY_PATH = S14_9 / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
REACHES_PATH = S14_9 / "inputs" / "spatial" / "reaches_topology.shp"
ASRIV_PATH = ROOT / "0_reach_topology" / "data" / "raw" / "hydrology" / "channel_geometry" / "andreadis_2025_global_bankfull_width_depth" / "data" / "asriv.shp"
DEM_PATH = ROOT / "0_reach_topology" / "data" / "processed" / "dem_prb" / "dem.tif"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
Q10_VALUES = (1.0, 1.5, 2.0, 2.5)
FULL_TERMINALS = (1, 13, 16, 20, 22, 23, 26, 28, 56, 102, 163, 166, 212, 217)
OBSERVED_TERMINALS = (20, 22, 23, 26, 56, 163, 212, 217)
BOOTSTRAP_REPLICATES = 10_000
STATION_BOOTSTRAP_SEED = 20260820
TREE_BOOTSTRAP_SEED = 20260821
NONINFERIOR_MARGIN = 0.005
REFERENCE_DAYS = 30.4375
SURVIVAL_LOWER = 0.01
SURVIVAL_UPPER = 1.0
SURVIVAL_INITIAL = 0.8
SURVIVAL_RIDGE_LAMBDA = 1.0
SURVIVAL_BOUNDARY_LOW = 0.02
SURVIVAL_BOUNDARY_HIGH = 0.99
MASS_REL_TOL = 1e-12
MASS_ZERO_ABS_TOL_KG_N = 1e-9
WATER_EPS = 1e-12


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_manifest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_core() -> ModuleType:
    return load_module(PARENT_CORE_PATH, "legacy20_frozen_parent_core")


def formal_registry() -> pd.DataFrame:
    frame = pd.read_parquet(FORMAL_PATH).copy()
    frame = frame.loc[frame.formal_ensemble_member.astype(bool)].sort_values("model_id").reset_index(drop=True)
    if len(frame) != 12 or set(frame.effective_tn_delivery_mu_month.astype(int)) != set(FORMAL_MUS):
        raise RuntimeError("frozen formal registry is not the expected 12-member ensemble")
    frame["delivery_mode"] = "T1"
    frame["rho_T"] = frame.effective_tn_delivery_mu_month.astype(float) / (1.0 + frame.effective_tn_delivery_mu_month.astype(float))
    frame["T0_encoded_as_mu_zero"] = False
    return frame


def topology_operators(reach_ids: np.ndarray | None = None) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY_PATH)
    topo["reach_id"] = topo.reach_id.astype(int)
    nodes = sorted(topo.reach_id.tolist()) if reach_ids is None else sorted(map(int, reach_ids))
    node_set = set(nodes)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topo.itertuples():
        if pd.notna(row.downstream_reach):
            rid, down = int(row.reach_id), int(row.downstream_reach)
            if rid in node_set:
                if down not in node_set:
                    raise RuntimeError("topology references a reach outside the domain")
                downstream[rid] = (down, float(row.frac))
    indegree = {rid: 0 for rid in nodes}
    for down, _ in downstream.values():
        indegree[down] += 1
    queue = deque(sorted(rid for rid, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
    if len(order) != len(nodes):
        raise RuntimeError("topology is cyclic or incomplete")
    terminal: dict[int, int] = {}
    for rid in nodes:
        current, seen = rid, set()
        fraction = 1.0
        while current in downstream:
            if current in seen:
                raise RuntimeError("topology cycle")
            seen.add(current)
            down, frac = downstream[current]
            fraction *= frac
            if not (0 < fraction <= 1 + 1e-12):
                raise RuntimeError("invalid cumulative routing fraction")
            current = down
        terminal[rid] = current
    terminals = tuple(sorted(set(terminal.values())))
    if terminals != FULL_TERMINALS:
        raise RuntimeError(f"expected 14 full terminal trees, got {terminals}")
    return order, downstream, terminal


def route_arrays(local: np.ndarray, reach_ids: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    order, downstream, terminal = topology_operators(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    routed = np.asarray(local, dtype=float).copy()
    for rid in order:
        if rid in downstream:
            down, frac = downstream[rid]
            routed[:, index[down]] += frac * routed[:, index[rid]]
    return routed, terminal


def cumulative_path_fractions(reach_ids: np.ndarray) -> pd.DataFrame:
    _, downstream, terminal = topology_operators(reach_ids)
    rows: list[dict[str, object]] = []
    for source in map(int, reach_ids):
        target, fraction, steps = source, 1.0, 0
        rows.append({"source_reach_id": source, "target_reach_id": target, "cumulative_routing_fraction": fraction, "path_steps": steps})
        while target in downstream:
            target, frac = downstream[target]
            fraction *= frac
            steps += 1
            rows.append({"source_reach_id": source, "target_reach_id": target, "cumulative_routing_fraction": fraction, "path_steps": steps})
    frame = pd.DataFrame(rows)
    frame["terminal_tree_id"] = frame.source_reach_id.map(terminal).astype(int)
    return frame


def prepare_model_arrays() -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray], np.ndarray]:
    return parent_core().prepare_model_arrays()


def temperature_forcings() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(MONTHLY_PATH)[["reach_id", "year", "month"]].copy()
    reach_ids = np.sort(base.reach_id.unique().astype(int))
    gw = pd.read_parquet(GW_TEMP_PATH).copy()
    gw["reach_id"] = gw.reach_id.astype(int)
    if len(gw) != 230 or set(gw.reach_id) != set(reach_ids):
        raise RuntimeError("Benz temperature must cover all 230 reaches")
    gw["gw_temp_2000_c"] = gw.groundwater_temperature_2020_c - gw.groundwater_temperature_change_2000_2020_c
    gw_long = base.merge(gw[["reach_id", "groundwater_temperature_2020_c", "gw_temp_2000_c"]], on="reach_id", validate="many_to_one")
    year_fraction = np.clip((gw_long.year.to_numpy(float) - 2000.0) / 20.0, 0.0, 1.0)
    gw_long["GW_STATIC_2020"] = gw_long.groundwater_temperature_2020_c
    gw_long["GW_STATIC_2000"] = gw_long.gw_temp_2000_c
    gw_long["GW_ENDPOINT_LINEAR_PROXY"] = gw_long.gw_temp_2000_c + year_fraction * (gw_long.groundwater_temperature_2020_c - gw_long.gw_temp_2000_c)
    gw_out = gw_long.melt(
        id_vars=["reach_id", "year", "month"],
        value_vars=["GW_STATIC_2020", "GW_STATIC_2000", "GW_ENDPOINT_LINEAR_PROXY"],
        var_name="gw_temperature_mode", value_name="groundwater_temperature_c",
    )
    gw_out["scientific_role"] = np.where(gw_out.gw_temperature_mode.eq("GW_STATIC_2020"), "primary_spatial_thermal_state", "sensitivity_only")
    gw_out["is_observed_monthly_groundwater_temperature"] = False

    aq = pd.read_parquet(AQ_TEMP_PATH).copy()
    aq["reach_id"] = aq.reach_id.astype(int)
    if len(aq) != 46_920 or aq[["reach_id", "year", "month"]].duplicated().any():
        raise RuntimeError("LMLT temperature is not the expected 230 x 204 registry")
    clim = aq.loc[aq.year.between(2006, 2015)].groupby(["reach_id", "month"], as_index=False).lake_mix_layer_temperature_c.mean().rename(columns={"lake_mix_layer_temperature_c": "AQ_CLIM"})
    aq_long = base.merge(clim, on=["reach_id", "month"], validate="many_to_one").merge(
        aq[["reach_id", "year", "month", "lake_mix_layer_temperature_c"]],
        on=["reach_id", "year", "month"], how="left", validate="one_to_one",
    )
    aq_long["AQ_FULL"] = np.where(aq_long.year.ge(2006), aq_long.lake_mix_layer_temperature_c, aq_long.AQ_CLIM)
    if aq_long[["AQ_CLIM", "AQ_FULL"]].isna().any().any():
        raise RuntimeError("AQ forcing contains an implicit pre-2006 gap")
    aq_out = aq_long.melt(
        id_vars=["reach_id", "year", "month"], value_vars=["AQ_CLIM", "AQ_FULL"],
        var_name="aquatic_temperature_mode", value_name="aquatic_temperature_c",
    )
    aq_out["pre2006_forcing"] = "AQ_CLIM"
    aq_out["is_observed_river_temperature"] = False
    return gw_out.sort_values(["gw_temperature_mode", "year", "month", "reach_id"]), aq_out.sort_values(["aquatic_temperature_mode", "year", "month", "reach_id"])


def station_macro_rmse(frame: pd.DataFrame, prediction: str = "pred_tn_mg_l") -> float:
    values = []
    for _, group in frame.groupby("station_key"):
        values.append(np.sqrt(np.mean((np.log1p(group[prediction]) - np.log1p(group.tn_mg_l)) ** 2)))
    return float(np.mean(values))


def tree_macro_rmse(frame: pd.DataFrame, prediction: str = "pred_tn_mg_l") -> float:
    values = []
    for _, group in frame.groupby("terminal_tree_id"):
        values.append(np.sqrt(np.mean((np.log1p(group[prediction]) - np.log1p(group.tn_mg_l)) ** 2)))
    return float(np.mean(values))


def paired_block_bootstrap(parent: pd.DataFrame, candidate: pd.DataFrame, block: str, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    keys = ["station_key", "year", "month", "fold_id", "layer", "model_id"]
    keep = list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block]))
    paired = parent[keep].merge(candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_parent", "_candidate"), validate="one_to_one")
    deltas = []
    for _, group in paired.groupby(block):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        rp = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_parent) - obs) ** 2))
        rc = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - obs) ** 2))
        deltas.append(rc - rp)
    block_delta = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(block_delta), size=(BOOTSTRAP_REPLICATES, len(block_delta)))
    distribution = block_delta[indices].mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    point = station_macro_rmse(candidate) - station_macro_rmse(parent) if block == "station_key" else tree_macro_rmse(candidate) - tree_macro_rmse(parent)
    return distribution, {
        "point_delta_rmse_log1p": float(point), "ci95_lower": float(lo), "ci95_upper": float(hi),
        "direction_improved": bool(point < 0), "noninferior": bool(hi < NONINFERIOR_MARGIN),
        "predictively_improved": bool(hi < 0), "paired_within_scheme": True,
    }


def mass_balance_pass(error: np.ndarray, reference_mass: np.ndarray) -> np.ndarray:
    error = np.abs(np.asarray(error, dtype=float))
    reference_mass = np.asarray(reference_mass, dtype=float)
    return np.where(reference_mass > MASS_ZERO_ABS_TOL_KG_N, error / reference_mass <= MASS_REL_TOL, error <= MASS_ZERO_ABS_TOL_KG_N)

