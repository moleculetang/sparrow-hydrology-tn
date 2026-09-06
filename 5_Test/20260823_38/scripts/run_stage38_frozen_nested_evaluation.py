from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_38"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE37_SCRIPT = TEST / "20260823_37" / "scripts" / "run_stage37_daily_joint_fit.py"
FORCING = TEST / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
DAILY_Q = TEST / "20260823_28" / "outputs" / "daily_hydrograph_separation.parquet"
SUPPORT = TEST / "20260823_28" / "outputs" / "gauge_support_hydrology.parquet"
GROUPS = TEST / "20260823_17" / "outputs" / "station_spatial_groups.parquet"
LOCK = TEST / "20260823_37" / "reports" / "development_parameter_lock.json"
EPS = 1e-12


def load_stage37():
    spec = importlib.util.spec_from_file_location("stage37_core", STAGE37_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_month_difference(a: int, b: int) -> int:
    direct = abs(int(a) - int(b))
    return min(direct, 12 - direct)


def path_metrics(pred_fraction: np.ndarray, obs_fraction: np.ndarray, dates: pd.DatetimeIndex, stations: list[str]) -> dict[str, float]:
    rows = []
    for i, station in enumerate(stations):
        mask = np.isfinite(pred_fraction[:, i]) & np.isfinite(obs_fraction[:, i])
        rows.append({"station_norm": station, "mean_predicted": float(np.mean(pred_fraction[mask, i])), "mean_proxy": float(np.mean(obs_fraction[mask, i]))})
    station = pd.DataFrame(rows)
    pred_month = pd.DataFrame(pred_fraction).assign(month=dates.month).groupby("month").mean().mean(axis=1)
    obs_month = pd.DataFrame(obs_fraction).assign(month=dates.month).groupby("month").mean().mean(axis=1)
    pred_peak, obs_peak = int(pred_month.idxmax()), int(obs_month.idxmax())
    return {
        "delayed_fraction_RMSE": float(np.sqrt(np.nanmean((pred_fraction - obs_fraction) ** 2))),
        "station_mean_fraction_spearman": float(station.mean_predicted.corr(station.mean_proxy, method="spearman")),
        "predicted_basin_peak_month": pred_peak, "proxy_basin_peak_month": obs_peak,
        "peak_circular_difference_months": circular_month_difference(pred_peak, obs_peak),
    }


def main() -> None:
    core = load_stage37()
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "DEVELOPMENT_PARAMETER_LOCKED_BEFORE_2019_2022_READ":
        raise RuntimeError("Development parameter lock missing")
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
    daily["station_norm"] = daily.station_norm.astype(str)
    meta = pd.read_parquet(SUPPORT)[["station_norm", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
    meta["station_norm"] = meta.station_norm.astype(str)
    stations = sorted(set(daily.station_norm) & set(meta.station_norm))
    meta = meta.set_index("station_norm").loc[stations].reset_index()
    reach_index = {int(reach): i for i, reach in enumerate(reaches)}
    upstream = core.upstream_matrix(reaches)
    support_matrix = np.empty((len(stations), n_reach))
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
    sigma = np.maximum((proxy_high - proxy_low) / np.maximum(obs, EPS) / 3.92, 0.05)
    parent_monthly = pd.read_parquet(SUPPORT)
    parent_monthly["station_norm"] = parent_monthly.station_norm.astype(str)
    parent_long = obs_long.reset_index()[["date", "station_norm"]]
    parent_long["year"] = parent_long.date.dt.year
    parent_long["month"] = parent_long.date.dt.month
    parent_long = parent_long.merge(parent_monthly[["station_norm", "year", "month", "support_total_m3_s", "support_delayed_fraction"]], on=["station_norm", "year", "month"], how="left", validate="many_to_one")
    parent_total = parent_long.support_total_m3_s.to_numpy(float).reshape(n_time, len(stations))
    parent_fraction = parent_long.support_delayed_fraction.to_numpy(float).reshape(n_time, len(stations))
    best_raw = np.asarray(lock["raw_parameters"], float)
    best_p = core.raw_to_physical(best_raw)
    fast, delayed, max_mass, minimum = core.simulate_daily(rain, aet, best_p)
    fast_q, delayed_q = fast @ volume_operator, delayed @ volume_operator
    total_q = fast_q + delayed_q
    fraction_q = delayed_q / np.maximum(total_q, EPS)
    frozen = np.asarray((dates.year >= 2019) & (dates.year <= 2022))
    time_metrics = {
        "parent": core.metric_summary(obs[frozen], parent_total[frozen], stations),
        "daily": core.metric_summary(obs[frozen], total_q[frozen], stations),
        "parent_path": path_metrics(parent_fraction[frozen], obs_fraction[frozen], dates[frozen], stations),
        "daily_path": path_metrics(fraction_q[frozen], obs_fraction[frozen], dates[frozen], stations),
    }
    groups = pd.read_parquet(GROUPS)
    groups["q_site"] = groups.q_site.astype(str)
    station_tree = groups.set_index("q_site").terminal_reach.astype(int).to_dict()
    missing = sorted(set(stations) - set(station_tree))
    if missing:
        raise RuntimeError(f"Terminal-tree registry missing stations: {missing}")
    train_time = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    sim_end = int(np.asarray(dates.year <= 2018).sum())
    parent_raw = core.physical_to_raw({
        "capacity": 200.0, "gamma": 1.6,
        "quick_rho": 0.3 ** (1.0 / (365.2425 / 12.0)),
        "base_recharge": 1.0 - (1.0 - 0.14) ** (1.0 / (365.2425 / 12.0)),
        "base_rho": 0.95 ** (1.0 / (365.2425 / 12.0)),
    })
    starts = [parent_raw, parent_raw + np.array([0.25, -0.2, 0.15, 0.2, 0.2]), parent_raw + np.array([-0.25, 0.2, -0.15, -0.2, -0.2])]
    prior_precision = 0.02
    proxy_weight = 0.1
    nested_pred = np.full((train_time.sum(), len(stations)), np.nan)
    nested_fraction = np.full_like(nested_pred, np.nan)
    fit_rows = []
    cache_path = OUT / "terminal_tree_daily_refit_parameters.parquet"
    cached_fits = pd.read_parquet(cache_path).set_index("terminal_tree") if cache_path.exists() else pd.DataFrame()
    for tree in sorted(set(station_tree.values())):
        target_idx = np.array([i for i, station in enumerate(stations) if station_tree[station] == tree], dtype=int)
        train_idx = np.array([i for i in range(len(stations)) if i not in set(target_idx)], dtype=int)
        def objective(raw: np.ndarray) -> float:
            p = core.raw_to_physical(raw)
            f, d, _, _ = core.simulate_daily(rain[:sim_end], aet[:sim_end], p)
            fq, dq = f @ volume_operator[:, train_idx], d @ volume_operator[:, train_idx]
            prediction = (fq + dq)[-int(train_time.sum()):]
            fraction = dq[-int(train_time.sum()):] / np.maximum(prediction, EPS)
            residual = np.log1p(prediction) - np.log1p(obs[train_time][:, train_idx])
            total_loss = float(np.nanmean(np.nanmean(residual ** 2, axis=0)))
            z = (fraction - obs_fraction[train_time][:, train_idx]) / sigma[train_time][:, train_idx]
            path_loss = float(np.nanmean(np.nanmean(z ** 2, axis=0)))
            prior = 0.5 * prior_precision * float(np.sum((raw - parent_raw) ** 2))
            return total_loss + proxy_weight * path_loss + prior
        if not cached_fits.empty and int(tree) in cached_fits.index:
            cached = cached_fits.loc[int(tree)]
            p = np.asarray([cached.capacity, cached.gamma, cached.quick_rho, cached.base_recharge, cached.base_rho], float)
            selected_raw = core.physical_to_raw(core.parameter_dict(p))
            selected_fun = float(cached.objective)
            selected_success = bool(cached.success)
            selected_nfev = int(cached.evaluations)
        else:
            results = [minimize(objective, start, method="L-BFGS-B", bounds=[(-6.0, 6.0)] * 5, options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-6, "maxls": 30}) for start in starts]
            selected = min(results, key=lambda item: item.fun)
            selected_raw = np.asarray(selected.x)
            p = core.raw_to_physical(selected_raw)
            selected_fun = float(selected.fun)
            selected_success = bool(selected.success)
            selected_nfev = int(selected.nfev)
        f, d, tree_mass, tree_minimum = core.simulate_daily(rain[:sim_end], aet[:sim_end], p)
        fq, dq = f @ volume_operator[:, target_idx], d @ volume_operator[:, target_idx]
        nested_pred[:, target_idx] = (fq + dq)[-int(train_time.sum()):]
        nested_fraction[:, target_idx] = dq[-int(train_time.sum()):] / np.maximum(nested_pred[:, target_idx], EPS)
        fit_rows.append({
            "terminal_tree": int(tree), "training_stations": len(train_idx), "target_stations": len(target_idx),
            "objective": selected_fun, "success": selected_success, "evaluations": selected_nfev,
            "max_abs_raw": float(np.max(np.abs(selected_raw))), "mass_balance_max_abs_mm": float(tree_mass),
            **core.parameter_dict(p),
        })
        print(json.dumps(fit_rows[-1], ensure_ascii=False), flush=True)
    nested_metrics = {
        "parent": core.metric_summary(obs[train_time], parent_total[train_time], stations),
        "daily": core.metric_summary(obs[train_time], nested_pred, stations),
        "parent_path": path_metrics(parent_fraction[train_time], obs_fraction[train_time], dates[train_time], stations),
        "daily_path": path_metrics(nested_fraction, obs_fraction[train_time], dates[train_time], stations),
    }
    pd.DataFrame(fit_rows).to_parquet(OUT / "terminal_tree_daily_refit_parameters.parquet", index=False)
    train_dates = dates[train_time]
    observed_train = obs[train_time]
    proxy_fraction_train = obs_fraction[train_time]
    parent_total_train = parent_total[train_time]
    parent_fraction_train = parent_fraction[train_time]
    prediction_frame = pd.DataFrame({
        "date": np.repeat(train_dates.to_numpy(), len(stations)),
        "station_norm": np.tile(np.asarray(stations, dtype=object), len(train_dates)),
        "terminal_tree": np.tile(np.asarray([station_tree[station] for station in stations], dtype=int), len(train_dates)),
        "observed_m3_s": observed_train.reshape(-1),
        "parent_m3_s": parent_total_train.reshape(-1),
        "daily_nested_m3_s": nested_pred.reshape(-1),
        "proxy_delayed_fraction": proxy_fraction_train.reshape(-1),
        "parent_delayed_fraction": parent_fraction_train.reshape(-1),
        "daily_nested_delayed_fraction": nested_fraction.reshape(-1),
    })
    prediction_frame.to_parquet(OUT / "terminal_tree_zero_history_daily_predictions.parquet", index=False)
    gates = contract["gates"]
    checks = {
        "time_pooled_NSE_noninferior": time_metrics["daily"]["pooled_NSE"] >= time_metrics["parent"]["pooled_NSE"] - gates["time_pooled_NSE_noninferiority_margin"],
        "time_station_median_NSE_noninferior": time_metrics["daily"]["station_median_NSE"] >= time_metrics["parent"]["station_median_NSE"] - gates["time_station_median_NSE_noninferiority_margin"],
        "nested_pooled_NSE_noninferior": nested_metrics["daily"]["pooled_NSE"] >= nested_metrics["parent"]["pooled_NSE"] - gates["nested_space_pooled_NSE_noninferiority_margin"],
        "nested_station_median_NSE_noninferior": nested_metrics["daily"]["station_median_NSE"] >= nested_metrics["parent"]["station_median_NSE"] - gates["nested_space_station_median_NSE_noninferiority_margin"],
        "time_delayed_fraction_RMSE_improved": time_metrics["daily_path"]["delayed_fraction_RMSE"] < time_metrics["parent_path"]["delayed_fraction_RMSE"],
        "nested_delayed_fraction_RMSE_improved": nested_metrics["daily_path"]["delayed_fraction_RMSE"] < nested_metrics["parent_path"]["delayed_fraction_RMSE"],
        "nested_fraction_spatial_ranking": nested_metrics["daily_path"]["station_mean_fraction_spearman"] >= gates["nested_station_mean_fraction_spearman_min"],
        "nested_fraction_phase": nested_metrics["daily_path"]["peak_circular_difference_months"] <= gates["basin_monthly_peak_circular_difference_max_months"],
        "full_fit_not_boundary_confounded": max(abs(value) for value in lock["raw_parameters"]) <= gates["raw_parameter_boundary_abs_max"],
    }
    decision = "DAILY_FAST_DELAYED_HYDROLOGY_SUPPORTED" if all(checks.values()) else "DAILY_FAST_DELAYED_HYDROLOGY_NOT_SUPPORTED"
    audit = {
        "stage": "20260823_38", "decision": decision, "checks": checks,
        "frozen_2019_2022": time_metrics, "nested_terminal_tree_2010_2018": nested_metrics,
        "terminal_tree_count": len(set(station_tree.values())), "full_fit_mass_balance_max_abs_mm": float(max_mass),
        "full_fit_minimum_state_or_flux_mm": float(minimum),
    }
    (REPORT / "stage38_daily_decision.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_38 日模型时间与空间检验\n\n"
        f"最终门禁状态：`{decision}`。\n\n"
        f"- 2019–2022总体NSE：月Q72日基线 `{time_metrics['parent']['pooled_NSE']:.3f}`，日模型 `{time_metrics['daily']['pooled_NSE']:.3f}`；\n"
        f"- 2019–2022站点中位NSE：`{time_metrics['parent']['station_median_NSE']:.3f} → {time_metrics['daily']['station_median_NSE']:.3f}`；\n"
        f"- 整河树零历史总体NSE：`{nested_metrics['parent']['pooled_NSE']:.3f} → {nested_metrics['daily']['pooled_NSE']:.3f}`；\n"
        f"- 整河树零历史延迟比例RMSE：`{nested_metrics['parent_path']['delayed_fraction_RMSE']:.3f} → {nested_metrics['daily_path']['delayed_fraction_RMSE']:.3f}`；\n"
        f"- 日模型延迟比例空间Spearman：`{nested_metrics['daily_path']['station_mean_fraction_spearman']:.3f}`；峰值月差 `{nested_metrics['daily_path']['peak_circular_difference_months']}`个月。\n\n"
        "所有目标河树的流量与分割数据均从对应重训中删除；没有Gauge末端校正。\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"), "development_lock_sha256": sha256(LOCK),
        "forcing_sha256": sha256(FORCING), "daily_q_sha256": sha256(DAILY_Q), "groups_sha256": sha256(GROUPS),
        "decision_sha256": sha256(REPORT / "stage38_daily_decision.json")
    }, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
