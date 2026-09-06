from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_30"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
BRIDGE = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
SUPPORT = TEST / "20260823_28" / "outputs" / "gauge_support_hydrology.parquet"
PROXY = TEST / "20260823_28" / "outputs" / "monthly_hydrograph_separation_proxy.parquet"
OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
CORE = TEST / "20260823_29" / "scripts"
sys.path.insert(0, str(CORE))
from differentiable_hydrology import (  # noqa: E402
    HydroParameters,
    parameter_dict,
    physical_to_raw,
    raw_to_physical,
    route_volumes,
    simulate,
)


torch.set_default_dtype(torch.float64)
torch.manual_seed(20260823)
np.random.seed(20260823)
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("stage30_topology", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.TOPOLOGY_PATH = TOPOLOGY
    return module._upstream_matrix(reaches)


def tensor(values, dtype=torch.float64):
    return torch.as_tensor(values, dtype=dtype, device=DEVICE)


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs, pred = obs[valid], pred[valid]
    residual = pred - obs
    log_residual = np.log1p(pred) - np.log1p(obs)
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    log_denom = float(np.sum((np.log1p(obs) - np.mean(np.log1p(obs))) ** 2))
    return {
        "n": int(len(obs)),
        "NSE": float(1 - np.sum(residual**2) / denom) if denom > 0 else math.nan,
        "NSE_log": float(1 - np.sum(log_residual**2) / log_denom) if log_denom > 0 else math.nan,
        "PBIAS_pct": float(100 * np.sum(residual) / np.sum(obs)) if np.sum(obs) > 0 else math.nan,
        "RMSE_log": float(np.sqrt(np.mean(log_residual**2))),
        "MAE_log": float(np.mean(np.abs(log_residual))),
    }


def summarize_predictions(frame: pd.DataFrame, prediction: str, candidate: str, layer: str, period: str) -> tuple[dict, pd.DataFrame]:
    part = frame[frame.period.eq(period)].copy()
    pooled = metrics(part.q_m3s.to_numpy(float), part[prediction].to_numpy(float))
    station_rows = []
    for station, group in part.groupby("station_norm", sort=False):
        row = metrics(group.q_m3s.to_numpy(float), group[prediction].to_numpy(float))
        station_rows.append({"candidate": candidate, "layer": layer, "period": period, "station_norm": station, **row})
    station = pd.DataFrame(station_rows)
    pooled.update({
        "candidate": candidate,
        "layer": layer,
        "period": period,
        "station_count": int(len(station)),
        "station_mean_NSE": float(station.NSE.mean()),
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    })
    return pooled, station


def panel_arrays(frame: pd.DataFrame):
    frame = frame.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    reaches = frame.reach_id.drop_duplicates().to_numpy(int)
    n_reach = len(reaches)
    n_time = len(frame) // n_reach

    def rt(column: str) -> torch.Tensor:
        return tensor(frame[column].to_numpy(float).reshape(n_reach, n_time).T)

    first = frame.groupby("reach_id", sort=False).head(1)
    time = frame.groupby(["year", "month"], sort=False).head(1)[["year", "month", "month_seconds"]].reset_index(drop=True)
    return frame, reaches, time, rt, tensor(first.catchment_area_km2.to_numpy(float))


def periodic_initial(effective: torch.Tensor, demand: torch.Tensor, parameters: HydroParameters, cycles: int = 8):
    n_reach = effective.shape[1]
    source = torch.full((n_reach,), 0.5, device=DEVICE) * parameters.prod_capacity
    quick = torch.zeros(n_reach, device=DEVICE)
    delayed = torch.zeros(n_reach, device=DEVICE)
    for _ in range(cycles):
        spun = simulate(effective[:72], demand[:72], parameters, source, quick, delayed)
        source, quick, delayed = spun["source_end"][-1], spun["quick_end"][-1], spun["delayed_end"][-1]
    return source, quick, delayed


def postfit_spinup(effective: torch.Tensor, demand: torch.Tensor, parameters: HydroParameters, tolerance: float, max_cycles: int):
    n_reach = effective.shape[1]
    source = torch.full((n_reach,), 0.5, device=DEVICE) * parameters.prod_capacity
    quick = torch.zeros(n_reach, device=DEVICE)
    delayed = torch.zeros(n_reach, device=DEVICE)
    delta = math.inf
    for cycle in range(1, max_cycles + 1):
        previous = torch.cat([source, quick, delayed]).detach()
        with torch.no_grad():
            spun = simulate(effective[:72], demand[:72], parameters, source, quick, delayed)
            source, quick, delayed = spun["source_end"][-1], spun["quick_end"][-1], spun["delayed_end"][-1]
            delta = float(torch.max(torch.abs(torch.cat([source, quick, delayed]) - previous)).cpu())
        if delta <= tolerance:
            return source, quick, delayed, cycle, delta, True
    return source, quick, delayed, max_cycles, delta, False


class JointProblem:
    def __init__(self, contract: dict):
        bridge = pd.read_parquet(BRIDGE)
        self.bridge, self.reaches, self.time, rt, self.area = panel_arrays(bridge)
        self.reach_index = {int(reach): i for i, reach in enumerate(self.reaches)}
        self.time_index = {(int(row.year), int(row.month)): i for i, row in self.time.iterrows()}
        self.effective = rt("positive_input_mm")
        self.demand = rt("aet_storage_withdrawn_mm") + rt("aet_unmet_mm")
        self.seconds = tensor(self.time.month_seconds.to_numpy(float))
        self.upstream = tensor(upstream_matrix(self.reaches))
        support = pd.read_parquet(SUPPORT)
        station_meta = support[["station_norm", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
        station_meta = station_meta.sort_values(["reach_id", "station_norm"]).reset_index(drop=True)
        self.stations = station_meta.station_norm.astype(str).tolist()
        self.station_index = {station: i for i, station in enumerate(self.stations)}
        self.station_reach = tensor([self.reach_index[int(value)] for value in station_meta.reach_id], torch.long)
        self.station_fraction = tensor(station_meta.downstream_fraction_on_reach.to_numpy(float))
        obs = pd.read_parquet(OBS).copy()
        obs = obs[obs.station_norm.astype(str).isin(self.station_index)].copy()
        obs["station_norm"] = obs.station_norm.astype(str)
        obs["period"] = np.where(obs.year.le(2018), "development_2006_2018", "check_2019_2022")
        self.obs_frame = obs.sort_values(["station_norm", "year", "month"]).reset_index(drop=True)
        self.obs_train = self.obs_frame[self.obs_frame.year.le(2018)].copy()
        self.train_pad = self._pad_observations(self.obs_train)
        proxy = pd.read_parquet(PROXY).copy()
        proxy["station_norm"] = proxy.station_norm.astype(str)
        proxy = proxy[proxy.station_norm.isin(self.station_index) & proxy.proxy_usable & proxy.year.le(2018)].copy()
        self.proxy_frame = proxy.sort_values(["station_norm", "year", "month"]).reset_index(drop=True)
        self.proxy_pad = self._pad_proxy(self.proxy_frame, float(contract["proxy_min_fraction_sigma"]))
        self.ridge = float(contract["gauge_ridge_precision"])
        self.prior_precision = float(contract["physical_raw_prior_precision"])
        self.proxy_weight = float(contract["proxy_likelihood_weight"])
        locked = json.loads(LOCK.read_text(encoding="utf-8"))["physical_parameters"]
        self.parent_values = {name: float(locked[name]) for name in ["prod_capacity", "runoff_gamma", "quick_rho", "base_release", "base_rho"]}
        self.parent_raw = physical_to_raw(self.parent_values).to(DEVICE)
        with torch.no_grad():
            parent = raw_to_physical(self.parent_raw)
            parent_sim = self.forward(parent)
            train_support = self._extract_padded(parent_sim, self.train_pad)
            valid = self.train_pad["mask"]
            log_total = torch.log1p(train_support["total"])[valid]
            fast_fraction = train_support["fast_fraction"][valid]
            self.log_total_mean = log_total.mean()
            self.log_total_std = torch.clamp(log_total.std(unbiased=False), min=EPS)
            self.fast_fraction_mean = fast_fraction.mean()
            self.fast_fraction_std = torch.clamp(fast_fraction.std(unbiased=False), min=EPS)

    def _pad_observations(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        counts = frame.groupby("station_norm").size().reindex(self.stations, fill_value=0)
        width = int(counts.max())
        shape = (len(self.stations), width)
        t_idx = np.zeros(shape, dtype=np.int64)
        obs = np.zeros(shape, dtype=float)
        month = np.ones(shape, dtype=float)
        mask = np.zeros(shape, dtype=bool)
        for station, group in frame.groupby("station_norm", sort=False):
            s = self.station_index[station]
            n = len(group)
            t_idx[s, :n] = [self.time_index[(int(y), int(m))] for y, m in zip(group.year, group.month)]
            obs[s, :n] = group.q_m3s.to_numpy(float)
            month[s, :n] = group.month.to_numpy(float)
            mask[s, :n] = True
        return {"time": tensor(t_idx, torch.long), "obs": tensor(obs), "month": tensor(month), "mask": tensor(mask, torch.bool)}

    def _pad_proxy(self, frame: pd.DataFrame, min_sigma: float) -> dict[str, torch.Tensor]:
        counts = frame.groupby("station_norm").size().reindex(self.stations, fill_value=0)
        width = int(counts.max())
        shape = (len(self.stations), width)
        t_idx = np.zeros(shape, dtype=np.int64)
        median = np.zeros(shape, dtype=float)
        sigma = np.ones(shape, dtype=float)
        low = np.zeros(shape, dtype=float)
        high = np.zeros(shape, dtype=float)
        mask = np.zeros(shape, dtype=bool)
        methods = {name: np.zeros(shape, dtype=float) for name in ["lh", "eckhardt", "ukih"]}
        for station, group in frame.groupby("station_norm", sort=False):
            s = self.station_index[station]
            n = len(group)
            t_idx[s, :n] = [self.time_index[(int(y), int(m))] for y, m in zip(group.year, group.month)]
            median[s, :n] = group.delayed_proxy_median_fraction.to_numpy(float)
            low[s, :n] = group.delayed_proxy_lower_fraction.to_numpy(float)
            high[s, :n] = group.delayed_proxy_upper_fraction.to_numpy(float)
            sigma[s, :n] = np.maximum((high[s, :n] - low[s, :n]) / 2.0, min_sigma)
            methods["lh"][s, :n] = group.delayed_lh_fraction.to_numpy(float)
            methods["eckhardt"][s, :n] = group.delayed_eckhardt_fraction.to_numpy(float)
            methods["ukih"][s, :n] = group.delayed_ukih_fraction.to_numpy(float)
            mask[s, :n] = True
        result = {"time": tensor(t_idx, torch.long), "median": tensor(median), "sigma": tensor(sigma), "low": tensor(low), "high": tensor(high), "mask": tensor(mask, torch.bool)}
        result.update({name: tensor(value) for name, value in methods.items()})
        return result

    def forward(self, parameters: HydroParameters) -> dict[str, torch.Tensor]:
        source, quick, delayed = periodic_initial(self.effective, self.demand, parameters, cycles=8)
        result = simulate(self.effective, self.demand, parameters, source, quick, delayed)
        local_fast_volume = result["quick_release"] * self.area[None, :] * 1000.0
        local_delayed_volume = result["delayed_discharge"] * self.area[None, :] * 1000.0
        routed_fast_volume = route_volumes(result["quick_release"], self.area, self.upstream)
        routed_delayed_volume = route_volumes(result["delayed_discharge"], self.area, self.upstream)
        reach = self.station_reach
        fraction = self.station_fraction[None, :]
        support_fast_volume = routed_fast_volume[:, reach] - (1.0 - fraction) * local_fast_volume[:, reach]
        support_delayed_volume = routed_delayed_volume[:, reach] - (1.0 - fraction) * local_delayed_volume[:, reach]
        support_fast = support_fast_volume / self.seconds[:, None]
        support_delayed = support_delayed_volume / self.seconds[:, None]
        return {
            "state": result,
            "local_fast_volume": local_fast_volume,
            "local_delayed_volume": local_delayed_volume,
            "routed_fast_volume": routed_fast_volume,
            "routed_delayed_volume": routed_delayed_volume,
            "support_fast": support_fast,
            "support_delayed": support_delayed,
            "support_total": support_fast + support_delayed,
            "support_fast_fraction": support_fast / torch.clamp(support_fast + support_delayed, min=EPS),
            "support_delayed_fraction": support_delayed / torch.clamp(support_fast + support_delayed, min=EPS),
        }

    def _extract_padded(self, simulation: dict[str, torch.Tensor], pad: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        station = torch.arange(len(self.stations), device=DEVICE)[:, None].expand_as(pad["time"])
        fast = simulation["support_fast"][pad["time"], station]
        delayed = simulation["support_delayed"][pad["time"], station]
        return {
            "fast": fast,
            "delayed": delayed,
            "total": fast + delayed,
            "fast_fraction": fast / torch.clamp(fast + delayed, min=EPS),
            "delayed_fraction": delayed / torch.clamp(fast + delayed, min=EPS),
        }

    def profile_gauge(self, simulation: dict[str, torch.Tensor]):
        values = self._extract_padded(simulation, self.train_pad)
        mask = self.train_pad["mask"]
        log_total = torch.log1p(values["total"])
        x = torch.stack([
            torch.ones_like(log_total),
            (log_total - self.log_total_mean) / self.log_total_std,
            (values["fast_fraction"] - self.fast_fraction_mean) / self.fast_fraction_std,
            torch.sin(2 * math.pi * self.train_pad["month"] / 12.0),
            torch.cos(2 * math.pi * self.train_pad["month"] / 12.0),
        ], dim=2)
        xm = x * mask[:, :, None]
        y = torch.log1p(self.train_pad["obs"]) - log_total
        xtx = torch.einsum("snk,snl->skl", xm, xm)
        xty = torch.einsum("snk,sn->sk", xm, y * mask)
        identity = torch.eye(5, device=DEVICE)[None, :, :]
        beta = torch.linalg.solve(xtx + self.ridge * identity, xty[:, :, None]).squeeze(2)
        correction = torch.einsum("snk,sk->sn", x, beta)
        residual = (log_total + correction - torch.log1p(self.train_pad["obs"])) * mask
        count = torch.clamp(mask.sum(dim=1), min=1)
        station_loss = (torch.sum(residual**2, dim=1) + self.ridge * torch.sum(beta**2, dim=1)) / count
        return station_loss.mean(), beta

    def proxy_loss_value(self, simulation: dict[str, torch.Tensor]):
        values = self._extract_padded(simulation, self.proxy_pad)
        mask = self.proxy_pad["mask"]
        z = (values["delayed_fraction"] - self.proxy_pad["median"]) / self.proxy_pad["sigma"]
        count = torch.clamp(mask.sum(dim=1), min=1)
        station_loss = torch.sum((z * mask) ** 2, dim=1) / count
        valid_station = mask.any(dim=1)
        return station_loss[valid_station].mean()

    def objective(self, raw: torch.Tensor, use_proxy: bool):
        parameters = raw_to_physical(raw)
        simulation = self.forward(parameters)
        total_loss, beta = self.profile_gauge(simulation)
        prior = 0.5 * self.prior_precision * torch.sum((raw - self.parent_raw) ** 2)
        proxy_loss = self.proxy_loss_value(simulation) if use_proxy else torch.zeros((), device=DEVICE)
        objective = total_loss + prior + self.proxy_weight * proxy_loss
        return objective, total_loss, proxy_loss, prior, beta, simulation


def fit_candidate(problem: JointProblem, candidate: str, use_proxy: bool):
    perturbations = [
        torch.zeros(5, device=DEVICE),
        tensor([0.20, -0.15, 0.10, 0.12, 0.18]),
        tensor([-0.18, 0.12, -0.10, -0.08, -0.15]),
    ]
    trials = []
    best_raw = None
    for start_id, perturbation in enumerate(perturbations):
        raw = torch.nn.Parameter((problem.parent_raw + perturbation).clone())
        optimizer = torch.optim.Adam([raw], lr=0.025)
        history = []
        for iteration in range(180):
            optimizer.zero_grad()
            objective, total_loss, proxy_loss, prior, _, _ = problem.objective(raw, use_proxy)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            optimizer.step()
            if iteration % 30 == 0 or iteration == 179:
                history.append(float(objective.detach().cpu()))
        lbfgs = torch.optim.LBFGS([raw], max_iter=80, tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            objective, _, _, _, _, _ = problem.objective(raw, use_proxy)
            objective.backward()
            return objective

        lbfgs.step(closure)
        with torch.no_grad():
            objective, total_loss, proxy_loss, prior, _, _ = problem.objective(raw, use_proxy)
            values = parameter_dict(raw_to_physical(raw))
            row = {
                "candidate": candidate,
                "start_id": start_id,
                "objective": float(objective.cpu()),
                "total_loss": float(total_loss.cpu()),
                "proxy_loss": float(proxy_loss.cpu()),
                "physical_prior": float(prior.cpu()),
                "adam_history": json.dumps(history),
                **values,
            }
            trials.append(row)
            if best_raw is None or row["objective"] < min(item["objective"] for item in trials[:-1]):
                best_raw = raw.detach().clone()
        print(f"{candidate} start {start_id}: objective={row['objective']:.6f} params={values}", flush=True)
    return best_raw, pd.DataFrame(trials).sort_values("objective")


def gauge_predict_all(problem: JointProblem, simulation: dict[str, torch.Tensor], beta: torch.Tensor) -> pd.DataFrame:
    frame = problem.obs_frame.copy()
    support_total = simulation["support_total"].detach().cpu().numpy()
    support_fast_fraction = simulation["support_fast_fraction"].detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    latent, gauge = [], []
    for row in frame.itertuples():
        s = problem.station_index[str(row.station_norm)]
        t = problem.time_index[(int(row.year), int(row.month))]
        q = float(support_total[t, s])
        ff = float(support_fast_fraction[t, s])
        x = np.array([
            1.0,
            (math.log1p(q) - float(problem.log_total_mean.cpu())) / float(problem.log_total_std.cpu()),
            (ff - float(problem.fast_fraction_mean.cpu())) / float(problem.fast_fraction_std.cpu()),
            math.sin(2 * math.pi * row.month / 12),
            math.cos(2 * math.pi * row.month / 12),
        ])
        latent.append(q)
        gauge.append(max(math.expm1(math.log1p(q) + float(x @ beta_np[s])), 0.0))
    frame["latent_support_m3_s"] = latent
    frame["gauge_prediction_m3_s"] = gauge
    return frame


def path_metrics(problem: JointProblem, simulation: dict[str, torch.Tensor], candidate: str) -> tuple[dict, pd.DataFrame]:
    proxy = problem.proxy_frame.copy()
    delayed = simulation["support_delayed_fraction"].detach().cpu().numpy()
    predicted = []
    for row in proxy.itertuples():
        s = problem.station_index[str(row.station_norm)]
        t = problem.time_index[(int(row.year), int(row.month))]
        predicted.append(float(delayed[t, s]))
    proxy["candidate"] = candidate
    proxy["predicted_delayed_fraction"] = predicted
    station_rows = []
    for station, group in proxy.groupby("station_norm", sort=False):
        climatology_pred = group.groupby("month").predicted_delayed_fraction.mean()
        climatology_obs = group.groupby("month").delayed_proxy_median_fraction.mean()
        common = climatology_pred.index.intersection(climatology_obs.index)
        seasonal_corr = climatology_pred.loc[common].corr(climatology_obs.loc[common], method="pearson") if len(common) >= 3 else math.nan
        if len(common):
            peak_pred = int(climatology_pred.loc[common].idxmax())
            peak_obs = int(climatology_obs.loc[common].idxmax())
            peak_distance = min(abs(peak_pred - peak_obs), 12 - abs(peak_pred - peak_obs))
        else:
            peak_distance = math.nan
        station_rows.append({
            "candidate": candidate,
            "station_norm": station,
            "mean_predicted_delayed_fraction": float(group.predicted_delayed_fraction.mean()),
            "mean_proxy_delayed_fraction": float(group.delayed_proxy_median_fraction.mean()),
            "absolute_mean_error": float(abs(group.predicted_delayed_fraction.mean() - group.delayed_proxy_median_fraction.mean())),
            "seasonal_correlation": float(seasonal_corr) if np.isfinite(seasonal_corr) else math.nan,
            "peak_month_distance": float(peak_distance),
        })
    station = pd.DataFrame(station_rows)
    by_station = station.dropna(subset=["mean_predicted_delayed_fraction", "mean_proxy_delayed_fraction"])
    spearman = by_station.mean_predicted_delayed_fraction.corr(by_station.mean_proxy_delayed_fraction, method="spearman")
    summary = {
        "candidate": candidate,
        "station_count": int(len(station)),
        "median_BFI_absolute_error": float(station.absolute_mean_error.median()),
        "fraction_stations_error_le_0_25": float((station.absolute_mean_error <= 0.25).mean()),
        "station_spearman": float(spearman),
        "median_seasonal_correlation": float(station.seasonal_correlation.median()),
        "median_peak_month_distance": float(station.peak_month_distance.median()),
        "monthly_RMSE": float(np.sqrt(np.mean((proxy.predicted_delayed_fraction - proxy.delayed_proxy_median_fraction) ** 2))),
    }
    for method in ["lh", "eckhardt", "ukih"]:
        summary[f"bias_vs_{method}"] = float((proxy.predicted_delayed_fraction - proxy[f"delayed_{method}_fraction"]).mean())
    return summary, station


def reach_output(problem: JointProblem, simulation: dict[str, torch.Tensor], candidate: str) -> pd.DataFrame:
    state = simulation["state"]
    n_time, n_reach = state["quick_release"].shape
    base = pd.DataFrame({
        "reach_id": np.repeat(problem.reaches, n_time),
        "year": np.tile(problem.time.year.to_numpy(int), n_reach),
        "month": np.tile(problem.time.month.to_numpy(int), n_reach),
    })
    def flatten(value: torch.Tensor):
        return value.detach().cpu().numpy().T.reshape(-1)
    base["candidate"] = candidate
    base["source_store_end_mm"] = flatten(state["source_end"])
    base["fast_store_end_mm"] = flatten(state["quick_end"])
    base["delayed_store_end_mm"] = flatten(state["delayed_end"])
    base["local_fast_release_mm"] = flatten(state["quick_release"])
    base["local_delayed_discharge_mm"] = flatten(state["delayed_discharge"])
    base["local_fast_m3_s"] = flatten(simulation["local_fast_volume"] / problem.seconds[:, None])
    base["local_delayed_m3_s"] = flatten(simulation["local_delayed_volume"] / problem.seconds[:, None])
    base["routed_fast_m3_s"] = flatten(simulation["routed_fast_volume"] / problem.seconds[:, None])
    base["routed_delayed_m3_s"] = flatten(simulation["routed_delayed_volume"] / problem.seconds[:, None])
    base["routed_total_m3_s"] = base.routed_fast_m3_s + base.routed_delayed_m3_s
    base["routed_delayed_fraction"] = base.routed_delayed_m3_s / base.routed_total_m3_s.clip(lower=EPS)
    base["local_mass_balance_error_mm"] = flatten(state["mass_error"])
    return base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage 30 contract was not registered before fitting")
    problem = JointProblem(contract)
    parent_raw = problem.parent_raw.detach().clone()
    trial_path = OUT / "candidate_solver_trials.parquet"
    if trial_path.exists():
        trials = pd.read_parquet(trial_path)
        recovered = {}
        for candidate in ["H1_TOTAL_PLUS_GAUGE", "H2_TOTAL_GAUGE_PLUS_DELAYED_PROXY"]:
            row = trials[trials.candidate.eq(candidate)].sort_values("objective").iloc[0]
            recovered[candidate] = physical_to_raw({
                name: float(row[name])
                for name in ["prod_capacity", "runoff_gamma", "quick_rho", "base_release", "base_rho"]
            }).to(DEVICE)
        h1_raw = recovered["H1_TOTAL_PLUS_GAUGE"]
        h2_raw = recovered["H2_TOTAL_GAUGE_PLUS_DELAYED_PROXY"]
        print("Recovered H1/H2 from candidate_solver_trials.parquet", flush=True)
    else:
        h1_raw, h1_trials = fit_candidate(problem, "H1_TOTAL_PLUS_GAUGE", False)
        h2_raw, h2_trials = fit_candidate(problem, "H2_TOTAL_GAUGE_PLUS_DELAYED_PROXY", True)
        trials = pd.concat([h1_trials, h2_trials], ignore_index=True)
        trials.to_parquet(trial_path, index=False)
    candidate_raw = {"H0_LOCKED_PARENT": parent_raw, "H1_TOTAL_PLUS_GAUGE": h1_raw, "H2_TOTAL_GAUGE_PLUS_DELAYED_PROXY": h2_raw}
    parameter_rows, gauge_rows, prediction_parts, reach_parts = [], [], [], []
    summary_rows, station_parts, path_rows, path_station_parts, audit_rows = [], [], [], [], []
    for candidate, raw in candidate_raw.items():
        parameters = raw_to_physical(raw)
        source, quick, delayed, cycles, terminal_delta, converged = postfit_spinup(
            problem.effective,
            problem.demand,
            parameters,
            float(contract["differentiable_periodic_spinup"]["postfit_audit_tolerance_mm"]),
            int(contract["differentiable_periodic_spinup"]["postfit_max_cycles"]),
        )
        # Use the audited converged state for all permanent outputs and evaluation.
        state = simulate(problem.effective, problem.demand, parameters, source, quick, delayed)
        local_fast_volume = state["quick_release"] * problem.area[None, :] * 1000.0
        local_delayed_volume = state["delayed_discharge"] * problem.area[None, :] * 1000.0
        routed_fast_volume = route_volumes(state["quick_release"], problem.area, problem.upstream)
        routed_delayed_volume = route_volumes(state["delayed_discharge"], problem.area, problem.upstream)
        reach = problem.station_reach
        fraction = problem.station_fraction[None, :]
        support_fast = (routed_fast_volume[:, reach] - (1 - fraction) * local_fast_volume[:, reach]) / problem.seconds[:, None]
        support_delayed = (routed_delayed_volume[:, reach] - (1 - fraction) * local_delayed_volume[:, reach]) / problem.seconds[:, None]
        simulation = {
            "state": state,
            "local_fast_volume": local_fast_volume,
            "local_delayed_volume": local_delayed_volume,
            "routed_fast_volume": routed_fast_volume,
            "routed_delayed_volume": routed_delayed_volume,
            "support_fast": support_fast,
            "support_delayed": support_delayed,
            "support_total": support_fast + support_delayed,
            "support_fast_fraction": support_fast / torch.clamp(support_fast + support_delayed, min=EPS),
            "support_delayed_fraction": support_delayed / torch.clamp(support_fast + support_delayed, min=EPS),
        }
        _, beta = problem.profile_gauge(simulation)
        params = parameter_dict(parameters)
        parameter_rows.append({"candidate": candidate, **params})
        for station, values in zip(problem.stations, beta.detach().cpu().numpy()):
            gauge_rows.append({"candidate": candidate, "station_norm": station, **dict(zip(["intercept", "log_support_total", "fast_fraction", "month_sin", "month_cos"], values))})
        pred = gauge_predict_all(problem, simulation, beta)
        pred["candidate"] = candidate
        prediction_parts.append(pred)
        for layer, column in [("LATENT_SUPPORT", "latent_support_m3_s"), ("GAUGE_OBSERVATION", "gauge_prediction_m3_s")]:
            for period in ["development_2006_2018", "check_2019_2022"]:
                pooled, station_metric = summarize_predictions(pred, column, candidate, layer, period)
                summary_rows.append(pooled)
                station_parts.append(station_metric)
        path, path_station = path_metrics(problem, simulation, candidate)
        path_rows.append(path)
        path_station_parts.append(path_station)
        reach_parts.append(reach_output(problem, simulation, candidate))
        mass_error = float(torch.max(torch.abs(state["mass_error"])).cpu())
        minimum = min(float(torch.min(value).cpu()) for key, value in state.items() if key != "mass_error")
        local_total = local_fast_volume + local_delayed_volume
        routed_total = routed_fast_volume + routed_delayed_volume
        routing_direct = route_volumes(state["quick_release"] + state["delayed_discharge"], problem.area, problem.upstream)
        routing_rel = float(torch.max(torch.abs(routed_total - routing_direct) / torch.clamp(torch.abs(routing_direct), min=EPS)).cpu())
        audit_rows.append({
            "candidate": candidate,
            "spinup_cycles": cycles,
            "spinup_terminal_max_abs_delta_mm": terminal_delta,
            "spinup_converged": converged,
            "local_mass_balance_max_abs_mm": mass_error,
            "minimum_state_or_flux": minimum,
            "routing_relative_error": routing_rel,
            "local_fast_delayed_closure_max_abs_m3": float(torch.max(torch.abs(local_total - local_fast_volume - local_delayed_volume)).cpu()),
        })
        print(f"{candidate} permanent outputs complete", flush=True)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "candidate_physical_parameters.parquet", index=False)
    pd.DataFrame(gauge_rows).to_parquet(OUT / "profiled_gauge_parameters.parquet", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(OUT / "station_predictions.parquet", index=False)
    pd.concat(reach_parts, ignore_index=True).to_parquet(OUT / "full_reach_fast_delayed_hydrology.parquet", index=False)
    summary = pd.DataFrame(summary_rows)
    stations = pd.concat(station_parts, ignore_index=True)
    paths = pd.DataFrame(path_rows)
    summary.to_parquet(OUT / "total_flow_metrics.parquet", index=False)
    stations.to_parquet(OUT / "station_total_flow_metrics.parquet", index=False)
    paths.to_parquet(OUT / "path_partition_metrics.parquet", index=False)
    pd.concat(path_station_parts, ignore_index=True).to_parquet(OUT / "station_path_partition_metrics.parquet", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_parquet(OUT / "solver_spinup_mass_audit.parquet", index=False)
    gate = contract["hard_numerical_gates"]
    numerical_pass = bool(
        audit.spinup_converged.all()
        and (audit.spinup_terminal_max_abs_delta_mm <= float(gate["spinup_terminal_max_abs_delta_mm"])).all()
        and (audit.local_mass_balance_max_abs_mm <= float(gate["local_mass_balance_max_abs_mm"])).all()
        and (audit.routing_relative_error <= float(gate["routing_relative_error"])).all()
        and (audit.minimum_state_or_flux >= -1e-10).all()
    )
    decision = {
        "stage": "20260823_30",
        "status": "GLOBAL_JOINT_MAP_NUMERICAL_PASS" if numerical_pass else "GLOBAL_JOINT_MAP_NUMERICAL_FAIL",
        "device": str(DEVICE),
        "numerical_pass": numerical_pass,
        "candidate_role": {
            "H0_LOCKED_PARENT": "registered parent",
            "H1_TOTAL_PLUS_GAUGE": "joint total-flow candidate",
            "H2_TOTAL_GAUGE_PLUS_DELAYED_PROXY": "joint total-flow plus delayed-proxy candidate"
        },
        "promotion_deferred_to": "20260823_31-34",
        "gauge_correction_changes_tn_facing_flow": False,
        "physical_parameters_change_tn_facing_states_and_fluxes": True,
        "2019_2022_used_for_fitting": False,
    }
    (REPORT / "stage30_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# 20260823_30 全局联合水文校准\n\n"
        f"状态：`{decision['status']}`。五个 Gauge 系数仍是观测算子；只有五个物理参数能改变 TN 使用的快流、延迟流和储量。\n\n"
        "## 物理参数\n\n```text\n" + pd.DataFrame(parameter_rows).to_string(index=False) + "\n```\n\n"
        "## 总流量结果\n\n```text\n" + summary.to_string(index=False) + "\n```\n\n"
        "## 延迟响应代理结果\n\n```text\n" + paths.to_string(index=False) + "\n```\n\n"
        "## 边界\n\nGauge 层结果不能晋级为 TN 水量；`_31–34` 只依据潜在河段/测站支撑水量及路径证据完成时间和空间裁决。\n"
    )
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "bridge_sha256": sha256(BRIDGE),
        "support_sha256": sha256(SUPPORT),
        "proxy_sha256": sha256(PROXY),
        "observations_sha256": sha256(OBS),
        "parameters_sha256": sha256(OUT / "candidate_physical_parameters.parquet"),
        "reach_hydrology_sha256": sha256(OUT / "full_reach_fast_delayed_hydrology.parquet"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if not numerical_pass:
        raise RuntimeError(f"Stage30 numerical gate failed: {audit.to_dict(orient='records')}")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print(paths.to_string(index=False))


if __name__ == "__main__":
    main()
