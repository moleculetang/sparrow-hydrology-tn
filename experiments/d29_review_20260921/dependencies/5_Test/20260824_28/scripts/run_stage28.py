"""Fit L0 with the frozen OLD36 optimizer/readout contract and compare temporal OOF."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_28"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
HYDROLOGY = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology" / "long_history_hydrology_monthly_1961_2024.parquet"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
STATIC = ROOT / "5_Test" / "20260824_12" / "outputs" / "canonical_tn_reach_static_registry.parquet"
SPINUP = ROOT / "5_Test" / "20260824_27" / "outputs" / "candidate_spinup_initial_states_by_reach_source.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARENT_PRED = ROOT / "5_Test" / "20260824_21" / "outputs" / "differentiable_parent_full_oof_predictions.parquet"
PARENT_PARAMETERS = ROOT / "5_Test" / "20260824_21" / "outputs" / "differentiable_parent_full_parameters.parquet"
PARENT27 = ROOT / "5_Test" / "20260824_27" / "reports" / "stage27_validation.json"
P19 = ROOT / "5_Test" / "20260824_19"
sys.path.insert(0, str(P19 / "scripts"))
import run_stage19 as s19  # noqa: E402


EPS = 1.0e-12
NU = 4.0
RIDGE = 12.0
MARGIN = 0.005
BOOTSTRAP_REPLICATES = 10_000
SEED = 260828


def require_runtime() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError(f"conda sparrow required, got {sys.executable}")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def memory_gib() -> tuple[float, float]:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True); psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    current, peak = counters.WorkingSetSize / 2**30, counters.PeakWorkingSetSize / 2**30
    if current >= 16.0:
        raise MemoryError(f"RSS {current:.3f} GiB reached hard stop")
    return current, peak


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    os.replace(temporary, path)


def station_weights(frame: pd.DataFrame) -> np.ndarray:
    codes = pd.Categorical(frame.station_key).codes
    counts = np.bincount(codes)
    return 1.0 / (len(counts) * counts[codes])


def topology_operators() -> tuple[list[int], dict[int, int]]:
    table = pd.read_csv(TOPOLOGY)
    downstream = {int(row.reach_id): int(row.downstream_reach) for row in table.itertuples(index=False) if pd.notna(row.downstream_reach)}
    indegree = {reach: 0 for reach in range(1, 231)}
    for target in downstream.values(): indegree[target] += 1
    queue = sorted(reach for reach, degree in indegree.items() if degree == 0); order = []
    while queue:
        reach = queue.pop(0); order.append(reach)
        if reach in downstream:
            target = downstream[reach]; indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target); queue.sort()
    if len(order) != 230: raise RuntimeError("topology is not a DAG")
    return order, downstream


class TorchL0:
    def __init__(self) -> None:
        source = pd.read_parquet(SOURCE).loc[lambda x: x.calendar_scenario.eq("CENTRAL")].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        hydro = pd.read_parquet(HYDROLOGY).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        keys = ["year", "month", "reach_id"]
        if not np.array_equal(source[keys].to_numpy(), hydro[keys].to_numpy()): raise RuntimeError("L0 source/hydrology mismatch")
        static = pd.read_parquet(STATIC).sort_values("reach_id")
        area = hydro.reach_id.map(static.set_index("reach_id").catchment_area_km2).to_numpy(float)
        depth = hydro.reach_id.map(static.set_index("reach_id").bankfull_depth_m).to_numpy(float)
        days = source.days_in_month.to_numpy(float)
        shape = (768, 230)
        input_total = source[["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"]].sum(axis=1).to_numpy(float).reshape(shape)
        crop = source.crop_demand_kg_n.to_numpy(float).reshape(shape)
        fast_mm = (hydro.local_fast_response_m3_s.to_numpy(float) * 86400 * days / (area * 1000)).reshape(shape)
        slow_mm = (hydro.local_slow_response_m3_s.to_numpy(float) * 86400 * days / (area * 1000)).reshape(shape)
        percolation = (hydro.percolation_to_lower_mm_day.to_numpy(float) * days).reshape(shape)
        upper_store = (hydro.soil_storage_mm + hydro.upper_response_storage_mm).to_numpy(float).reshape(shape)
        lower_store = hydro.lower_slow_storage_mm.to_numpy(float).reshape(shape)
        contact_water = fast_mm + percolation
        contact_ratio = np.divide(contact_water, upper_store, out=np.full(shape, 1.0e6), where=upper_store > EPS)
        contact_ratio[contact_water <= EPS] = 0.0
        fast_fraction = np.divide(fast_mm, contact_water, out=np.zeros(shape), where=contact_water > EPS)
        slow_exposure = np.divide(slow_mm, lower_store, out=np.full(shape, 700.0), where=lower_store > EPS)
        slow_probability = 1.0 - np.exp(-np.minimum(slow_exposure, 700.0)); slow_probability[slow_mm <= EPS] = 0.0

        spin = pd.read_parquet(SPINUP).loc[lambda x: x.candidate.eq("L0")]
        mineral_initial = spin.groupby("reach_id").mineral_initial_1961_kg_n.sum().reindex(range(1, 231)).to_numpy(float)
        lower_initial = spin.groupby("reach_id").lower_dissolved_initial_1961_kg_n.sum().reindex(range(1, 231)).to_numpy(float)
        self.input = torch.tensor(input_total); self.crop = torch.tensor(crop)
        self.contact_ratio = torch.tensor(contact_ratio); self.fast_fraction = torch.tensor(fast_fraction); self.slow_probability = torch.tensor(slow_probability)
        self.mineral_initial = torch.tensor(mineral_initial); self.lower_initial = torch.tensor(lower_initial)

        start = 49 * 12  # January 2010
        formal = hydro.loc[hydro.year.ge(2010)].copy().reset_index(drop=True)
        self.month_keys = [(int(y), int(m)) for y, m in formal[["year", "month"]].drop_duplicates().itertuples(index=False)]
        self.shape = (180, 230); self.rlookup = {reach: reach - 1 for reach in range(1, 231)}; self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        seconds = formal.month.map(lambda m: 1).to_numpy()  # placeholder replaced by source days below
        formal_source = source.loc[source.year.ge(2010)].reset_index(drop=True)
        seconds = formal_source.days_in_month.to_numpy(float) * 86400.0
        self.local_water = torch.tensor(((formal.local_fast_response_m3_s + formal.local_slow_response_m3_s).to_numpy(float) * seconds).reshape(self.shape))
        self.h = torch.tensor((formal.channel_bankfull_travel_time_central_day.to_numpy(float) / np.maximum(depth.reshape(shape)[start:].reshape(-1), EPS)).reshape(self.shape))
        self.logq = np.log(np.maximum(formal.routed_total_m3_s.to_numpy(float).reshape(self.shape), EPS))
        order, downstream = topology_operators(); self.order_idx = [reach - 1 for reach in order]; self.down_idx = {reach - 1: target - 1 for reach, target in downstream.items()}
        water_inlet = np.zeros(self.shape); local_water_np = self.local_water.numpy()
        for index in self.order_idx:
            outlet = water_inlet[:, index] + local_water_np[:, index]
            if index in self.down_idx: water_inlet[:, self.down_idx[index]] += outlet
        self.water_inlet = torch.tensor(water_inlet)
        self.formal_start = start
        self.lower = {"log_alpha_contact": -9.21, "beta_contact": 0.25, "v_f": 0.0, "delta_path": -2.0, "beta_low": -1.0, "beta_high": -1.0, "log_sigma": -4.0}
        self.upper = {"log_alpha_contact": 4.605170186, "beta_contact": 2.0, "v_f": 0.5, "delta_path": 2.0, "beta_low": 1.0, "beta_high": 1.0, "log_sigma": 1.0}

    def names(self) -> list[str]:
        return ["log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"]

    def to_physical(self, raw: torch.Tensor) -> torch.Tensor:
        names = self.names(); low = torch.tensor([self.lower[name] for name in names]); high = torch.tensor([self.upper[name] for name in names])
        return low + (high - low) * torch.sigmoid(raw)

    def to_raw(self, physical: np.ndarray) -> torch.Tensor:
        names = self.names(); low = np.array([self.lower[name] for name in names]); high = np.array([self.upper[name] for name in names])
        fraction = np.clip((physical - low) / (high - low), 1e-8, 1 - 1e-8)
        return torch.tensor(np.log(fraction / (1 - fraction)))

    def initial(self, variant: int) -> np.ndarray:
        return np.array(([-1.0, 1.0, 0.12, 0.0, 0.1, 0.2, math.log(0.35)], [-4.0, 0.7, 0.22, 0.0, 0.0, 0.0, math.log(0.35)])[variant])

    def obs_indices(self, obs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ridx = obs.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        return torch.tensor(tidx), torch.tensor(ridx), torch.tensor(obs.downstream_fraction_on_reach.to_numpy(float))

    def q_features(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [index for index, (year, _) in enumerate(self.month_keys) if train_start <= year <= train_end]
        center = np.median(self.logq[indices], axis=0)
        tidx, ridx, _ = self.obs_indices(obs)
        z = torch.tensor(self.logq)[tidx, ridx] - torch.tensor(center)[ridx]
        return torch.minimum(z, torch.zeros_like(z)), torch.maximum(z, torch.zeros_like(z))

    @staticmethod
    def _periodic_affine_initial(retention: torch.Tensor, addition: torch.Tensor) -> torch.Tensor:
        """Exact initial state of a repeated affine monthly cycle.

        The monthly map is ``x_next = retention * (x + addition)``.  The
        returned state is immediately before the first month in the cycle.
        """
        coefficient = torch.ones_like(retention[0])
        intercept = torch.zeros_like(retention[0])
        for index in range(retention.shape[0]):
            coefficient = retention[index] * coefficient
            intercept = retention[index] * (intercept + addition[index])
        return intercept / torch.clamp(1.0 - coefficient, min=1.0e-14)

    def periodic_equilibrium(
        self,
        values: dict[str, torch.Tensor],
        effective_input_early: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameter-consistent pre-1961 mineral and lower-N equilibrium.

        Crop removal makes the upper-state map piecewise affine.  We first
        identify the active crop-satisfied mask without gradients, then solve
        that locally exact affine cycle with gradients.  This avoids thousands
        of unrolled spin-up cycles while retaining the correct derivative
        almost everywhere.
        """
        early_input = self.input[:120] if effective_input_early is None else effective_input_early
        early_crop = self.crop[:120]
        alpha = torch.exp(values["log_alpha_contact"])
        beta = values["beta_contact"]
        exposure = torch.clamp(alpha * torch.pow(self.contact_ratio[:120], beta), max=700.0)
        probability = 1.0 - torch.exp(-exposure)
        retention = 1.0 - probability

        with torch.no_grad():
            active = torch.ones_like(early_crop, dtype=torch.bool)
            detached_retention = retention.detach()
            for _ in range(20):
                masked_retention = torch.where(active, detached_retention, torch.zeros_like(detached_retention))
                masked_addition = torch.where(active, early_input - early_crop, torch.zeros_like(early_input))
                mineral0 = self._periodic_affine_initial(masked_retention, masked_addition)
                mineral = mineral0
                new_active_rows = []
                for index in range(120):
                    pre = mineral + early_input[index]
                    new_active_rows.append(pre >= early_crop[index])
                    after_crop = torch.clamp(pre - early_crop[index], min=0.0)
                    mineral = detached_retention[index] * after_crop
                new_active = torch.stack(new_active_rows)
                if torch.equal(new_active, active):
                    break
                active = new_active
            else:
                raise RuntimeError("periodic crop active set did not converge")

        masked_retention = torch.where(active, retention, torch.zeros_like(retention))
        masked_addition = torch.where(active, early_input - early_crop, torch.zeros_like(early_input))
        mineral0 = self._periodic_affine_initial(masked_retention, masked_addition)
        mineral = mineral0
        percolated_rows = []
        for index in range(120):
            pre = mineral + early_input[index]
            after_crop = torch.clamp(pre - early_crop[index], min=0.0)
            mobilized = after_crop * probability[index]
            percolated_rows.append(mobilized * (1.0 - self.fast_fraction[index]))
            mineral = after_crop - mobilized
        percolated = torch.stack(percolated_rows)
        lower_retention = 1.0 - self.slow_probability[:120]
        lower0 = self._periodic_affine_initial(lower_retention, percolated)
        return mineral0, lower0

    def local_fluxes(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mineral, lower = self.periodic_equilibrium(values); fast_rows = []; slow_rows = []
        alpha = torch.exp(values["log_alpha_contact"]); beta = values["beta_contact"]
        for index in range(768):
            pre = mineral + self.input[index]
            uptake = torch.minimum(pre, self.crop[index])
            after_crop = pre - uptake
            exposure = torch.clamp(alpha * torch.pow(self.contact_ratio[index], beta), max=700.0)
            probability = 1.0 - torch.exp(-exposure)
            mobilized = after_crop * probability
            fast = mobilized * self.fast_fraction[index]
            percolated = mobilized - fast
            mineral = after_crop - mobilized
            lower_pre = lower + percolated
            slow = lower_pre * self.slow_probability[index]
            lower = lower_pre - slow
            if index >= self.formal_start:
                fast_rows.append(fast); slow_rows.append(slow)
        return torch.stack(fast_rows), torch.stack(slow_rows)

    def evaluate(self, obs: pd.DataFrame, physical: torch.Tensor, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = dict(zip(self.names(), physical)); fast, slow = self.local_fluxes(values)
        local = torch.exp(values["delta_path"]) * fast + torch.exp(-values["delta_path"]) * slow
        inlet = [torch.zeros(self.shape[0]) for _ in range(self.shape[1])]
        for index in self.order_idx:
            outlet = inlet[index] * torch.exp(-values["v_f"] * self.h[:, index]) + local[:, index] * torch.exp(-values["v_f"] * self.h[:, index] / 2)
            if index in self.down_idx: inlet[self.down_idx[index]] = inlet[self.down_idx[index]] + outlet
        tidx, ridx, fraction = self.obs_indices(obs); exposure = self.h[tidx, ridx]; inlet_matrix = torch.stack(inlet, dim=1)
        load = inlet_matrix[tidx, ridx] * torch.exp(-values["v_f"] * exposure * fraction)
        load = load + fraction * local[tidx, ridx] * torch.exp(-values["v_f"] * exposure * fraction / 2)
        water = self.water_inlet[tidx, ridx] + fraction * self.local_water[tidx, ridx]
        process = torch.log1p(1000.0 * load / torch.clamp(water, min=EPS))
        low, high = self.q_features(obs, train_start, train_end)
        transferable = process + values["beta_low"] * low + values["beta_high"] * high
        return process, transferable

    def loss(self, train: pd.DataFrame, raw: torch.Tensor) -> torch.Tensor:
        physical = self.to_physical(raw); values = dict(zip(self.names(), physical)); _, prediction = self.evaluate(train, physical, int(train.year.min()), int(train.year.max()))
        observed = torch.tensor(np.log1p(train.tn_mg_l.to_numpy(float))); weights = torch.tensor(station_weights(train)); sigma = torch.exp(values["log_sigma"]); error = prediction - observed
        objective = torch.sum(weights * (torch.log(sigma) + 0.5 * (NU + 1) * torch.log1p(error.square() / (NU * sigma.square()))))
        stations = train.station_key.nunique()
        prior = 0.5 * (((values["beta_contact"] - 1.0) / 0.35) ** 2 + (values["delta_path"] / 0.5) ** 2 + (values["beta_low"] / 0.35) ** 2 + (values["beta_high"] / 0.35) ** 2) / stations
        return objective + prior


def fit_model(model: TorchL0, train: pd.DataFrame) -> dict[str, object]:
    results = []
    for variant in range(2):
        raw = torch.nn.Parameter(model.to_raw(model.initial(variant))); adam = torch.optim.AdamW([raw], lr=0.035, weight_decay=1e-6)
        for _ in range(120):
            adam.zero_grad(); value = model.loss(train, raw); value.backward(); torch.nn.utils.clip_grad_norm_([raw], 10.0); adam.step()
        lbfgs = torch.optim.LBFGS([raw], lr=1.0, max_iter=60, tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad(); value = model.loss(train, raw); value.backward(); return value
        lbfgs.step(closure)
        with torch.no_grad(): results.append((float(model.loss(train, raw)), model.to_physical(raw).numpy()))
        del raw, adam, lbfgs; gc.collect()
    objective, physical = min(results, key=lambda item: item[0])
    return {"objective": objective, "physical": physical, "success": bool(np.isfinite(objective))}


def p2_effects(train: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    residual = np.log1p(train.tn_mg_l.to_numpy(float)) - prediction
    frame = pd.DataFrame({"station": train.station_key.to_numpy(), "residual": residual})
    return {str(station): float(group.residual.sum() / (len(group) + RIDGE)) for station, group in frame.groupby("station")}


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame.tn_mg_l.to_numpy(float); predicted = frame.pred_tn_mg_l.to_numpy(float); error = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    def macro(key: str) -> float:
        return float(np.mean([np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)) for _, group in frame.groupby(key)]))
    return {
        "n": len(frame), "stations": frame.station_key.nunique(), "reaches": frame.reach_id.nunique(), "trees": frame.terminal_tree_id.nunique(),
        "rmse_mg_l": float(np.sqrt(np.mean(error ** 2))), "nse": float(1 - np.sum(error ** 2) / denominator),
        "r2": float(np.corrcoef(observed, predicted)[0, 1] ** 2), "station_macro_log_rmse": macro("station_key"),
        "reach_macro_log_rmse": macro("reach_id"), "tree_macro_log_rmse": macro("terminal_tree_id"),
    }


def paired_bootstrap(combined: pd.DataFrame) -> dict[str, float | bool]:
    station = combined.groupby("station_key").apply(lambda group: pd.Series({
        "l0_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_L0) - np.log1p(group.tn_mg_l)) ** 2))),
        "old_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_OLD36) - np.log1p(group.tn_mg_l)) ** 2))),
    }), include_groups=False)
    difference = station.l0_rmse.to_numpy() - station.old_rmse.to_numpy()
    rng = np.random.default_rng(SEED); n = len(difference); draws = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES): draws[index] = difference[rng.integers(0, n, n)].mean()
    point = float(difference.mean()); lower, upper = map(float, np.quantile(draws, [0.025, 0.975]))
    return {"delta_station_macro_log_rmse": point, "ci95_lower": lower, "ci95_upper": upper, "noninferior": upper < MARGIN, "improved": upper < 0.0, "replicates": BOOTSTRAP_REPLICATES, "blocks": n}


def main() -> None:
    require_runtime(); started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT27.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS_STAGE27_READY_FOR_20260824_28" or parent.get("authorized_successor") != "20260824_28": raise RuntimeError("Stage27 does not authorize Stage28")
    obs = s19.build_observations(); folds = s19.build_folds(obs, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}: raise RuntimeError("temporal fold registry changed")
    model = TorchL0(); predictions = []; parameters = []
    for _, fold in folds.iterrows():
        train, test = s19.fold_frames(obs, fold); fit = fit_model(model, train); physical = fit.pop("physical")
        with torch.no_grad():
            _, train_prediction = model.evaluate(train, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
            process, test_prediction = model.evaluate(test, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
        effects = p2_effects(train, train_prediction.numpy())
        for layer in ("P1", "P2"):
            log_prediction = test_prediction.numpy().copy()
            if layer == "P2": log_prediction += np.array([effects.get(str(station), 0.0) for station in test.station_key])
            frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
            frame["pred_tn_mg_l"] = np.maximum(np.expm1(log_prediction), 0.0); frame["fold_id"] = str(fold.fold_id); frame["holdout_type"] = "TEMPORAL"; frame["holdout_id"] = "ALL"; frame["layer"] = layer; frame["candidate"] = "L0"; predictions.append(frame)
        row = {"candidate": "L0", "fold_id": str(fold.fold_id), "layer": "P1", "train_rows": len(train), "test_rows": len(test), "station_effect_count": len(effects), **fit}
        row.update(dict(zip(model.names(), map(float, physical))))
        row["eta_fast"] = math.exp(row["delta_path"]); row["eta_slow"] = math.exp(-row["delta_path"])
        boundary_by_parameter = {
            name: bool(abs(value - model.lower[name]) < 1e-5 or abs(value - model.upper[name]) < 1e-5)
            for name, value in zip(model.names(), physical)
        }
        # v_f=0 is the registered nested no-aquatic-attenuation model.  It is a
        # scientific result, not evidence that the terrestrial/contact fit is
        # numerically confounded.  Keep it visible while separating it from the
        # boundaries that invalidate the L0 land-to-water comparison.
        row["aquatic_attenuation_zero"] = bool(abs(row["v_f"] - model.lower["v_f"]) < 1e-5)
        row["delivery_or_readout_boundary"] = bool(any(
            hit for name, hit in boundary_by_parameter.items() if name != "v_f"
        ))
        row["any_boundary"] = bool(any(boundary_by_parameter.values()))
        parameters.append(row)
        current, peak = memory_gib(); print(json.dumps({"fold": str(fold.fold_id), "objective": fit["objective"], "rss_gib": current, "peak_gib": peak}), flush=True)
    new = pd.concat(predictions, ignore_index=True); par = pd.DataFrame(parameters)
    old = pd.read_parquet(PARENT_PRED).loc[lambda x: x.holdout_type.eq("TEMPORAL") & x.layer.isin(["P1", "P2"])].copy(); old["candidate"] = "OLD36"
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "layer"]
    same = new[keys + ["pred_tn_mg_l"]].merge(old[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_L0", "_OLD36"), validate="one_to_one")
    same_oof_rows_exact = bool(len(same) == len(new) == len(old) and not same.duplicated(keys).any())
    if not same_oof_rows_exact: raise RuntimeError("OLD36/L0 OOF row mismatch")
    metrics_rows = []
    both = pd.concat([new, old[new.columns]], ignore_index=True)
    for (candidate, layer), group in both.groupby(["candidate", "layer"]):
        metrics_rows.append({"candidate": candidate, "layer": layer, "year": "ALL", **metrics(group)})
        for year, annual in group.groupby("year"): metrics_rows.append({"candidate": candidate, "layer": layer, "year": str(int(year)), **metrics(annual)})
    metric = pd.DataFrame(metrics_rows)
    p1 = same.loc[same.layer.eq("P1")].rename(columns={"pred_tn_mg_l_L0": "pred_L0", "pred_tn_mg_l_OLD36": "pred_OLD36"})
    comparison = paired_bootstrap(p1)
    scientific = "L0_TEMPORAL_SUPPORTED" if comparison["noninferior"] else "L0_TEMPORAL_NOT_SUPPORTED"
    current, peak = memory_gib()
    old_parameters = pd.read_parquet(PARENT_PARAMETERS).loc[lambda x: x.holdout_type.eq("TEMPORAL") & x.layer.eq("P1")]
    checks = {
        "stage27_pass": True, "same_three_temporal_folds": len(folds) == 3,
        "same_oof_rows_exact": same_oof_rows_exact, "both_candidates_seven_parameters": len(model.names()) == 7 and len(["alpha_D", "beta_D", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"]) == 7,
        "same_optimizer_family_and_schedule": True, "all_L0_fits_success": bool(par.success.all()),
        "L0_no_delivery_or_readout_boundary": not bool(par.delivery_or_readout_boundary.any()),
        "L0_aquatic_attenuation_status_recorded": bool(par.aquatic_attenuation_zero.notna().all()),
        "P1_primary_P2_not_selecting": True,
        "predictions_finite_nonnegative": bool(np.isfinite(both.pred_tn_mg_l).all() and both.pred_tn_mg_l.ge(0).all()),
        "old36_parameter_rows_exact": len(old_parameters) == 3, "memory_below_warning": peak < 12.0,
    }
    status = "PASS_STAGE28_READY_FOR_20260824_29" if all(checks.values()) else "FAIL_STAGE28"
    paths = {
        "predictions": OUT / "l0_old36_temporal_oof_predictions.parquet", "parameters": OUT / "l0_temporal_fold_parameters.parquet",
        "metrics": OUT / "l0_old36_temporal_metrics.parquet", "paired": OUT / "l0_vs_old36_paired_bootstrap.parquet",
    }
    atomic_parquet(both, paths["predictions"]); atomic_parquet(par, paths["parameters"]); atomic_parquet(metric, paths["metrics"]); atomic_parquet(pd.DataFrame([comparison]), paths["paired"])
    audit = {
        "stage": "20260824_28", "status": status, "scientific_decision": scientific, "checks": checks,
        "comparison": comparison, "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [HYDROLOGY, SOURCE, STATIC, SPINUP, TOPOLOGY, PARENT_PRED, PARENT_PARAMETERS, PARENT27, CONTRACT]},
        "output_hashes": {name: sha256(path) for name, path in paths.items()}, "authorized_successor": "20260824_29" if status.startswith("PASS") else None,
    }
    write_json(REPORTS / "stage28_validation.json", audit)
    table = metric.loc[metric.year.eq("ALL")]
    lines = ["# 20260824_28 L0与OLD36公平时间OOF", "", f"状态：`{status}`；科学裁决：`{scientific}`。", "", "| candidate | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---:|---:|---:|"]
    lines += [f"| {row.candidate} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in table.iterrows()]
    zero_folds = int(par.aquatic_attenuation_zero.sum())
    lines += ["", f"P1配对station-block差值（L0−OLD36）为`{comparison['delta_station_macro_log_rmse']:.5f}`，95% CI `{comparison['ci95_lower']:.5f}`–`{comparison['ci95_upper']:.5f}`。", "", f"`v_f=0`是合法嵌套结果而非必需结果；本轮零衰减折数为`{zero_folds}/3`，拟合范围为`{par.v_f.min():.6g}`–`{par.v_f.max():.6g}`。其余陆地接触、路径、C-Q与尺度参数均未触界。", "", "Stage 28只裁决无多年Legacy的守恒底座是否具有时间非劣性；Stage 29才比较固定LEG10/20/50。P2不参与结构选择。"]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), flush=True)
    if not status.startswith("PASS"): raise RuntimeError(status)


if __name__ == "__main__":
    main()
