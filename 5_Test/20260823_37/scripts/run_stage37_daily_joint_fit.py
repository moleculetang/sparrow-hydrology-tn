from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_37"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
FORCING = TEST / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
DAILY_Q = TEST / "20260823_28" / "outputs" / "daily_hydrograph_separation.parquet"
SUPPORT = TEST / "20260823_28" / "outputs" / "gauge_support_hydrology.parquet"
MONTHLY_PARENT = SUPPORT
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-9, 1 - 1e-9)
    return np.log(value / (1 - value))


def physical_to_raw(values: dict[str, float]) -> np.ndarray:
    unit = np.array([
        (values["capacity"] - 50.0) / 950.0,
        (values["gamma"] - 0.5) / 4.5,
        (values["quick_rho"] - 0.01) / 0.94,
        (values["base_recharge"] - 0.0001) / 0.0499,
        (values["base_rho"] - values["quick_rho"] - 0.02) / max(0.9999 - values["quick_rho"] - 0.02, 1e-9),
    ])
    return logit(unit)


def raw_to_physical(raw: np.ndarray) -> np.ndarray:
    unit = 1.0 / (1.0 + np.exp(-np.asarray(raw, float)))
    capacity = 50.0 + 950.0 * unit[0]
    gamma = 0.5 + 4.5 * unit[1]
    quick = 0.01 + 0.94 * unit[2]
    recharge = 0.0001 + 0.0499 * unit[3]
    base = quick + 0.02 + (0.9999 - quick - 0.02) * unit[4]
    return np.array([capacity, gamma, quick, recharge, base])


def parameter_dict(values: np.ndarray) -> dict[str, float]:
    return dict(zip(["capacity", "gamma", "quick_rho", "base_recharge", "base_rho"], map(float, values)))


@nb.njit(cache=True)
def simulate_daily(rain: np.ndarray, aet: np.ndarray, p: np.ndarray):
    n_time, n_reach = rain.shape
    source = np.full(n_reach, 0.45 * p[0])
    quick = np.zeros(n_reach)
    delayed = np.zeros(n_reach)
    fast_out = np.empty((n_time, n_reach))
    delayed_out = np.empty((n_time, n_reach))
    max_mass = 0.0
    minimum = 1e300
    for t in range(n_time):
        for r in range(n_reach):
            total_start = source[r] + quick[r] + delayed[r]
            precipitation = max(rain[t, r], 0.0)
            saturation = min(max(source[r] / p[0], 0.0), 1.0)
            generated = min(precipitation, precipitation * saturation ** p[1])
            source[r] += precipitation - generated
            actual_aet = min(max(aet[t, r], 0.0), source[r])
            source[r] -= actual_aet
            overflow = max(source[r] - p[0], 0.0)
            source[r] = min(source[r], p[0])
            recharge = p[3] * source[r]
            source[r] -= recharge
            quick_available = quick[r] + generated + overflow
            fast_out[t, r] = (1.0 - p[2]) * quick_available
            quick[r] = p[2] * quick_available
            delayed_available = delayed[r] + recharge
            delayed_out[t, r] = (1.0 - p[4]) * delayed_available
            delayed[r] = p[4] * delayed_available
            total_end = source[r] + quick[r] + delayed[r]
            error = total_start + precipitation - actual_aet - total_end - fast_out[t, r] - delayed_out[t, r]
            max_mass = max(max_mass, abs(error))
            minimum = min(minimum, source[r], quick[r], delayed[r], fast_out[t, r], delayed_out[t, r])
    return fast_out, delayed_out, max_mass, minimum


def upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("stage37_topology", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.TOPOLOGY_PATH = TOPOLOGY
    return module._upstream_matrix(reaches)


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return np.nan
    denominator = np.sum((obs[mask] - np.mean(obs[mask])) ** 2)
    return float(1.0 - np.sum((obs[mask] - pred[mask]) ** 2) / denominator) if denominator > 0 else np.nan


def metric_summary(obs: np.ndarray, pred: np.ndarray, stations: list[str]) -> dict[str, float]:
    station_nse = [nse(obs[:, i], pred[:, i]) for i in range(len(stations))]
    valid = np.asarray([value for value in station_nse if np.isfinite(value)])
    mask = np.isfinite(obs) & np.isfinite(pred)
    return {
        "rows": int(mask.sum()), "stations": int(len(valid)),
        "pooled_NSE": nse(obs[mask], pred[mask]),
        "station_mean_NSE": float(np.mean(valid)), "station_median_NSE": float(np.median(valid)),
        "station_q25_NSE": float(np.quantile(valid, 0.25)), "station_q75_NSE": float(np.quantile(valid, 0.75)),
        "PBIAS_percent": float(100.0 * np.sum(pred[mask] - obs[mask]) / np.sum(obs[mask])),
        "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred[mask]) - np.log1p(obs[mask])) ** 2))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage37 was not registered")
    forcing = pd.read_parquet(FORCING)
    forcing["date"] = pd.to_datetime(forcing.date)
    forcing = forcing.sort_values(["date", "reach_id"]).reset_index(drop=True)
    reaches = np.sort(forcing.reach_id.unique())
    dates = pd.DatetimeIndex(np.sort(forcing.date.unique()))
    n_time, n_reach = len(dates), len(reaches)
    rain = forcing.precipitation_daily_mm.to_numpy(float).reshape(n_time, n_reach)
    aet = forcing.aet_uniform_daily_mm.to_numpy(float).reshape(n_time, n_reach)
    area = forcing.groupby("reach_id", sort=True).catchment_area_km2.first().reindex(reaches).to_numpy(float)
    daily = pd.read_parquet(DAILY_Q)
    daily["date"] = pd.to_datetime(daily.date)
    meta = pd.read_parquet(SUPPORT)[["station_norm", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
    meta["station_norm"] = meta.station_norm.astype(str)
    daily["station_norm"] = daily.station_norm.astype(str)
    stations = sorted(set(daily.station_norm) & set(meta.station_norm))
    meta = meta.set_index("station_norm").loc[stations].reset_index()
    reach_index = {int(reach): i for i, reach in enumerate(reaches)}
    upstream = upstream_matrix(reaches)
    support_matrix = np.empty((len(stations), n_reach), dtype=float)
    for i, row in enumerate(meta.itertuples(index=False)):
        local = reach_index[int(row.reach_id)]
        support_matrix[i] = upstream[local]
        support_matrix[i, local] = float(row.downstream_fraction_on_reach)
    volume_operator = area[:, None] * 1000.0 * support_matrix.T / 86400.0
    full_index = pd.MultiIndex.from_product([dates, stations], names=["date", "station_norm"])
    obs_long = daily.set_index(["date", "station_norm"]).reindex(full_index)
    obs = obs_long.q_m3_s.to_numpy(float).reshape(n_time, len(stations))
    proxy = obs_long.delayed_proxy_median_m3_s.to_numpy(float).reshape(n_time, len(stations))
    proxy_low = obs_long.delayed_proxy_lower_m3_s.to_numpy(float).reshape(n_time, len(stations))
    proxy_high = obs_long.delayed_proxy_upper_m3_s.to_numpy(float).reshape(n_time, len(stations))
    obs_fraction = proxy / np.maximum(obs, EPS)
    sigma_fraction = np.maximum((proxy_high - proxy_low) / np.maximum(obs, EPS) / 3.92, float(contract["proxy_min_fraction_sigma"]))
    train_time = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    simulation_time = np.asarray(dates.year <= 2018)
    train_obs = obs[train_time]
    train_fraction = obs_fraction[train_time]
    train_sigma = sigma_fraction[train_time]
    locked = json.loads(LOCK.read_text(encoding="utf-8"))["physical_parameters"]
    days_per_month = 365.2425 / 12.0
    parent = {
        "capacity": float(locked["prod_capacity"]), "gamma": float(locked["runoff_gamma"]),
        "quick_rho": float(locked["quick_rho"]) ** (1.0 / days_per_month),
        "base_recharge": 1.0 - (1.0 - float(locked["base_release"])) ** (1.0 / days_per_month),
        "base_rho": float(locked["base_rho"]) ** (1.0 / days_per_month),
    }
    parent_raw = physical_to_raw(parent)
    prior_precision = float(contract["physical_raw_prior_precision"])
    proxy_weight = float(contract["proxy_likelihood_weight"])
    history = []
    def forward(raw: np.ndarray, development_only: bool = True):
        end = int(simulation_time.sum()) if development_only else n_time
        p = raw_to_physical(raw)
        fast, delayed, max_mass, minimum = simulate_daily(rain[:end], aet[:end], p)
        fast_q = fast @ volume_operator
        delayed_q = delayed @ volume_operator
        return p, fast_q, delayed_q, max_mass, minimum
    def objective(raw: np.ndarray) -> float:
        p, fast_q, delayed_q, _, _ = forward(raw, True)
        total = fast_q + delayed_q
        predicted = total[-int(train_time.sum()):]
        predicted_fraction = delayed_q[-int(train_time.sum()):] / np.maximum(predicted, EPS)
        total_residual = np.log1p(predicted) - np.log1p(train_obs)
        total_loss = float(np.nanmean(np.nanmean(total_residual ** 2, axis=0)))
        z = (predicted_fraction - train_fraction) / train_sigma
        proxy_loss = float(np.nanmean(np.nanmean(z ** 2, axis=0)))
        prior = 0.5 * prior_precision * float(np.sum((raw - parent_raw) ** 2))
        value = total_loss + proxy_weight * proxy_loss + prior
        history.append({"objective": value, "total_loss": total_loss, "proxy_loss": proxy_loss, "prior": prior, **parameter_dict(p)})
        return value
    starts = [parent_raw, parent_raw + np.array([0.25, -0.2, 0.15, 0.2, 0.2]), parent_raw + np.array([-0.25, 0.2, -0.15, -0.2, -0.2])]
    trials = []
    best_result = None
    for start_id, start in enumerate(starts):
        result = minimize(objective, start, method="L-BFGS-B", bounds=[(-6.0, 6.0)] * 5, options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-6, "maxls": 30})
        row = {"start_id": start_id, "success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "evaluations": int(result.nfev), "objective": float(result.fun), **parameter_dict(raw_to_physical(result.x))}
        trials.append(row)
        if best_result is None or result.fun < best_result.fun:
            best_result = result
        print(json.dumps(row, ensure_ascii=False), flush=True)
    assert best_result is not None
    pd.DataFrame(trials).sort_values("objective").to_parquet(OUT / "daily_joint_fit_trials.parquet", index=False)
    pd.DataFrame(history).to_parquet(OUT / "daily_joint_fit_objective_history.parquet", index=False)
    best_raw = np.asarray(best_result.x)
    best_p, fast_q, delayed_q, max_mass, minimum = forward(best_raw, True)
    predicted = fast_q[-int(train_time.sum()):] + delayed_q[-int(train_time.sum()):]
    predicted_fraction = delayed_q[-int(train_time.sum()):] / np.maximum(predicted, EPS)
    parent_monthly = pd.read_parquet(MONTHLY_PARENT)
    parent_monthly["station_norm"] = parent_monthly.station_norm.astype(str)
    parent_daily = daily[daily.year.le(2018)][["station_norm", "date", "year", "month", "q_m3_s"]].merge(
        parent_monthly[["station_norm", "year", "month", "support_total_m3_s", "support_delayed_fraction"]],
        on=["station_norm", "year", "month"], how="left", validate="many_to_one",
    )
    development_metrics = {
        "monthly_Q72_parent_repeated_daily": metric_summary(train_obs, parent_daily.pivot(index="date", columns="station_norm", values="support_total_m3_s").reindex(index=dates[train_time], columns=stations).to_numpy(float), stations),
        "daily_joint_latent": metric_summary(train_obs, predicted, stations),
        "delayed_fraction_RMSE_parent": float(np.sqrt(np.nanmean((parent_daily.pivot(index="date", columns="station_norm", values="support_delayed_fraction").reindex(index=dates[train_time], columns=stations).to_numpy(float) - train_fraction) ** 2))),
        "delayed_fraction_RMSE_daily_joint": float(np.sqrt(np.nanmean((predicted_fraction - train_fraction) ** 2))),
    }
    lock = {
        "stage": "20260823_37", "status": "DEVELOPMENT_PARAMETER_LOCKED_BEFORE_2019_2022_READ",
        "development_period": "2010-2018", "spinup_period": "2006-2009",
        "selected_start_id": int(pd.DataFrame(trials).sort_values("objective").iloc[0].start_id),
        "raw_parameters": best_raw.tolist(), "physical_parameters": parameter_dict(best_p),
        "objective": float(best_result.fun), "mass_balance_max_abs_mm": float(max_mass), "minimum_state_or_flux_mm": float(minimum),
        "development_metrics": development_metrics,
        "claim_boundary": "No station-specific Gauge correction. Delayed flow is a hydrograph-response proxy, not observed groundwater or old water."
    }
    gates = contract["hard_numerical_gates"]
    lock["numerical_gate_pass"] = bool(np.isfinite(best_result.fun) and max_mass <= gates["mass_balance_max_abs_mm"] and minimum >= -gates["nonnegative_tolerance_mm"] and len(stations) == gates["development_station_count"])
    (REPORT / "development_parameter_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_37 日流量—分割联合训练\n\n"
        f"状态：`{'PASS' if lock['numerical_gate_pass'] else 'FAIL'}`。2010–2018拟合后参数已锁；2019–2022尚未进入本脚本。\n\n"
        f"- 日模型development pooled NSE：`{development_metrics['daily_joint_latent']['pooled_NSE']:.3f}`；站点中位NSE：`{development_metrics['daily_joint_latent']['station_median_NSE']:.3f}`。\n"
        f"- 月Q72常数日基线 pooled NSE：`{development_metrics['monthly_Q72_parent_repeated_daily']['pooled_NSE']:.3f}`；站点中位NSE：`{development_metrics['monthly_Q72_parent_repeated_daily']['station_median_NSE']:.3f}`。\n"
        f"- 延迟比例RMSE：`{development_metrics['delayed_fraction_RMSE_parent']:.3f} → {development_metrics['delayed_fraction_RMSE_daily_joint']:.3f}`。\n\n"
        "本阶段仅决定并锁定五个全局日过程参数；是否可升级由 `_38` 的时间外与零目标历史空间检验决定。\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"), "forcing_sha256": sha256(FORCING),
        "daily_q_sha256": sha256(DAILY_Q), "support_sha256": sha256(SUPPORT), "lock_sha256": sha256(REPORT / "development_parameter_lock.json")
    }, indent=2), encoding="utf-8")
    if not lock["numerical_gate_pass"]:
        raise RuntimeError(lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
