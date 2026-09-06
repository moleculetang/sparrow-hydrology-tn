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
RUN = TEST / "20260823_31"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
CACHE = OUT / "fit_cache"
STAGE30 = TEST / "20260823_30" / "scripts"
ATTRIBUTES = TEST / "20260823_16" / "outputs" / "reach_regionalization_attributes.parquet"
GROUPS = TEST / "20260823_17" / "outputs" / "station_spatial_groups.parquet"
REGIONALIZATION = TEST / "20260823_17" / "scripts" / "regionalization.py"
sys.path.insert(0, str(STAGE30))
from run_stage30_global_joint_map import (  # noqa: E402
    DEVICE,
    EPS,
    JointProblem,
    gauge_predict_all,
    metrics,
    parameter_dict,
    path_metrics,
    physical_to_raw,
    raw_to_physical,
    reach_output,
    route_volumes,
    simulate,
    summarize_predictions,
    tensor,
)


torch.set_default_dtype(torch.float64)
torch.manual_seed(20260823)
np.random.seed(20260823)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_attributes(problem: JointProblem) -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("stage31_regionalization", REGIONALIZATION)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    attributes = pd.read_parquet(ATTRIBUTES)
    attributes = attributes.set_index("comid").loc[problem.reaches].reset_index()
    array, names = module.base_attribute_matrix(attributes)
    return tensor(array), names, attributes


def parameter_field(base_raw: torch.Tensor, gamma: torch.Tensor | None, attributes: torch.Tensor):
    if gamma is None:
        raw = base_raw
    else:
        raw = base_raw[None, :] + attributes @ gamma
    return raw_to_physical(raw)


def profile_training_gauge(problem: JointProblem, simulation: dict[str, torch.Tensor], station_mask: torch.Tensor):
    values = problem._extract_padded(simulation, problem.train_pad)
    mask = problem.train_pad["mask"]
    log_total = torch.log1p(values["total"])
    x = torch.stack([
        torch.ones_like(log_total),
        (log_total - problem.log_total_mean) / problem.log_total_std,
        (values["fast_fraction"] - problem.fast_fraction_mean) / problem.fast_fraction_std,
        torch.sin(2 * math.pi * problem.train_pad["month"] / 12.0),
        torch.cos(2 * math.pi * problem.train_pad["month"] / 12.0),
    ], dim=2)
    xm = x * mask[:, :, None]
    y = torch.log1p(problem.train_pad["obs"]) - log_total
    xtx = torch.einsum("snk,snl->skl", xm, xm)
    xty = torch.einsum("snk,sn->sk", xm, y * mask)
    identity = torch.eye(5, device=DEVICE)[None, :, :]
    beta = torch.linalg.solve(xtx + problem.ridge * identity, xty[:, :, None]).squeeze(2)
    correction = torch.einsum("snk,sk->sn", x, beta)
    residual = (log_total + correction - torch.log1p(problem.train_pad["obs"])) * mask
    count = torch.clamp(mask.sum(dim=1), min=1)
    station_loss = (torch.sum(residual**2, dim=1) + problem.ridge * torch.sum(beta**2, dim=1)) / count
    return station_loss[station_mask].mean(), beta


def training_proxy_loss(problem: JointProblem, simulation: dict[str, torch.Tensor], station_mask: torch.Tensor):
    values = problem._extract_padded(simulation, problem.proxy_pad)
    mask = problem.proxy_pad["mask"]
    z = (values["delayed_fraction"] - problem.proxy_pad["median"]) / problem.proxy_pad["sigma"]
    count = torch.clamp(mask.sum(dim=1), min=1)
    station_loss = torch.sum((z * mask) ** 2, dim=1) / count
    valid = station_mask & mask.any(dim=1)
    return station_loss[valid].mean()


def objective(problem: JointProblem, attributes: torch.Tensor, base_raw: torch.Tensor, gamma: torch.Tensor | None, station_mask: torch.Tensor, contract: dict):
    parameters = parameter_field(base_raw, gamma, attributes)
    simulation = problem.forward(parameters)
    total_loss, beta = profile_training_gauge(problem, simulation, station_mask)
    proxy_loss = training_proxy_loss(problem, simulation, station_mask)
    global_prior = 0.5 * float(contract["global_raw_prior_precision"]) * torch.sum((base_raw - problem.parent_raw) ** 2)
    attribute_prior = torch.zeros((), device=DEVICE) if gamma is None else 0.5 * float(contract["attribute_raw_effect_precision"]) * torch.mean(gamma**2)
    value = total_loss + problem.proxy_weight * proxy_loss + global_prior + attribute_prior
    return value, total_loss, proxy_loss, global_prior, attribute_prior, beta, simulation


def fit(problem: JointProblem, attributes: torch.Tensor, station_mask: torch.Tensor, candidate: str, contract: dict):
    base = torch.nn.Parameter(problem.parent_raw.detach().clone())
    gamma = None if candidate == "R0_GLOBAL_REFIT" else torch.nn.Parameter(torch.zeros((attributes.shape[1], 5), device=DEVICE))
    parameters = [base] + ([] if gamma is None else [gamma])
    optimizer = torch.optim.Adam(parameters, lr=float(contract["optimization"]["adam_learning_rate"]))
    history = []
    for iteration in range(int(contract["optimization"]["adam_iterations"])):
        optimizer.zero_grad()
        values = objective(problem, attributes, base, gamma, station_mask, contract)
        values[0].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if iteration % 15 == 0 or iteration == int(contract["optimization"]["adam_iterations"]) - 1:
            history.append(float(values[0].detach().cpu()))
    lbfgs = torch.optim.LBFGS(parameters, max_iter=int(contract["optimization"]["lbfgs_max_iterations"]), tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        values = objective(problem, attributes, base, gamma, station_mask, contract)
        values[0].backward()
        return values[0]

    lbfgs.step(closure)
    with torch.no_grad():
        values = objective(problem, attributes, base, gamma, station_mask, contract)
    audit = {
        "candidate": candidate,
        "objective": float(values[0].cpu()),
        "total_loss": float(values[1].cpu()),
        "proxy_loss": float(values[2].cpu()),
        "global_prior": float(values[3].cpu()),
        "attribute_prior": float(values[4].cpu()),
        "history": json.dumps(history),
        **parameter_dict(raw_to_physical(base)),
    }
    return base.detach(), None if gamma is None else gamma.detach(), audit


def strict_target_predictions(problem: JointProblem, simulation: dict[str, torch.Tensor], target_stations: list[str], candidate: str, fold: int) -> pd.DataFrame:
    frame = problem.obs_frame[problem.obs_frame.station_norm.astype(str).isin(target_stations)].copy()
    support = simulation["support_total"].detach().cpu().numpy()
    delayed = simulation["support_delayed_fraction"].detach().cpu().numpy()
    predicted, delayed_fraction = [], []
    for row in frame.itertuples():
        s = problem.station_index[str(row.station_norm)]
        t = problem.time_index[(int(row.year), int(row.month))]
        predicted.append(float(support[t, s]))
        delayed_fraction.append(float(delayed[t, s]))
    frame["candidate"] = candidate
    frame["spatial_group"] = fold
    frame["zero_target_history_prediction_m3_s"] = predicted
    frame["predicted_delayed_fraction"] = delayed_fraction
    return frame


def strict_target_proxy(problem: JointProblem, simulation: dict[str, torch.Tensor], target_stations: list[str], candidate: str, fold: int) -> pd.DataFrame:
    frame = problem.proxy_frame[problem.proxy_frame.station_norm.astype(str).isin(target_stations)].copy()
    delayed = simulation["support_delayed_fraction"].detach().cpu().numpy()
    values = []
    for row in frame.itertuples():
        s = problem.station_index[str(row.station_norm)]
        t = problem.time_index[(int(row.year), int(row.month))]
        values.append(float(delayed[t, s]))
    frame["candidate"] = candidate
    frame["spatial_group"] = fold
    frame["predicted_delayed_fraction"] = values
    return frame


def strict_path_summary(proxy: pd.DataFrame, candidate: str) -> tuple[dict, pd.DataFrame]:
    part = proxy[proxy.candidate.eq(candidate)].copy()
    rows = []
    for station, group in part.groupby("station_norm", sort=False):
        pred_clim = group.groupby("month").predicted_delayed_fraction.mean()
        obs_clim = group.groupby("month").delayed_proxy_median_fraction.mean()
        common = pred_clim.index.intersection(obs_clim.index)
        corr = pred_clim.loc[common].corr(obs_clim.loc[common]) if len(common) >= 3 else math.nan
        if len(common):
            pred_peak, obs_peak = int(pred_clim.loc[common].idxmax()), int(obs_clim.loc[common].idxmax())
            distance = min(abs(pred_peak - obs_peak), 12 - abs(pred_peak - obs_peak))
        else:
            distance = math.nan
        rows.append({
            "candidate": candidate,
            "station_norm": station,
            "mean_predicted": float(group.predicted_delayed_fraction.mean()),
            "mean_proxy": float(group.delayed_proxy_median_fraction.mean()),
            "absolute_mean_error": float(abs(group.predicted_delayed_fraction.mean() - group.delayed_proxy_median_fraction.mean())),
            "seasonal_correlation": float(corr) if np.isfinite(corr) else math.nan,
            "peak_month_distance": float(distance),
        })
    station = pd.DataFrame(rows)
    spearman = station.mean_predicted.corr(station.mean_proxy, method="spearman")
    summary = {
        "candidate": candidate,
        "station_count": int(len(station)),
        "median_BFI_absolute_error": float(station.absolute_mean_error.median()),
        "fraction_stations_error_le_0_25": float((station.absolute_mean_error <= 0.25).mean()),
        "station_spearman": float(spearman),
        "median_seasonal_correlation": float(station.seasonal_correlation.median()),
        "median_peak_month_distance": float(station.peak_month_distance.median()),
        "monthly_RMSE": float(np.sqrt(np.mean((part.predicted_delayed_fraction - part.delayed_proxy_median_fraction) ** 2))),
        "bias_vs_lh": float((part.predicted_delayed_fraction - part.delayed_lh_fraction).mean()),
        "bias_vs_eckhardt": float((part.predicted_delayed_fraction - part.delayed_eckhardt_fraction).mean()),
        "bias_vs_ukih": float((part.predicted_delayed_fraction - part.delayed_ukih_fraction).mean()),
    }
    return summary, station


def flow_summary(frame: pd.DataFrame, candidate: str, period: str) -> tuple[dict, pd.DataFrame]:
    part = frame[(frame.candidate.eq(candidate)) & (frame.period.eq(period))].copy()
    pooled = metrics(part.q_m3s.to_numpy(float), part.zero_target_history_prediction_m3_s.to_numpy(float))
    rows = []
    for station, group in part.groupby("station_norm", sort=False):
        rows.append({"candidate": candidate, "period": period, "station_norm": station, **metrics(group.q_m3s, group.zero_target_history_prediction_m3_s)})
    stations = pd.DataFrame(rows)
    pooled.update({
        "candidate": candidate,
        "period": period,
        "station_count": int(len(stations)),
        "station_mean_NSE": float(stations.NSE.mean()),
        "station_median_NSE": float(stations.NSE.median()),
        "station_mean_RMSE_log": float(stations.RMSE_log.mean()),
        "station_median_RMSE_log": float(stations.RMSE_log.median()),
    })
    return pooled, stations


def bootstrap_paired(values: np.ndarray, seed: int = 20260823) -> dict[str, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), size=(10000, len(values)))], axis=1)
    return {"mean": float(values.mean()), "ci95_lower": float(np.quantile(draws, 0.025)), "ci95_upper": float(np.quantile(draws, 0.975))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage31 contract was not registered before fitting")
    problem = JointProblem(json.loads((TEST / "20260823_30" / "experiment_contract.json").read_text(encoding="utf-8")))
    attributes, attribute_names, attribute_frame = load_attributes(problem)
    groups = pd.read_parquet(GROUPS)
    groups["q_site"] = groups.q_site.astype(str)
    station_group = groups.set_index("q_site").group.to_dict()
    if set(problem.stations) != set(station_group):
        missing = sorted(set(problem.stations) - set(station_group))
        raise RuntimeError(f"Spatial groups do not cover authoritative stations: {missing}")
    fit_rows, coefficient_rows, strict_prediction_parts, strict_proxy_parts = [], [], [], []
    candidates = ["R0_GLOBAL_REFIT", "R1_ATTRIBUTE10"]
    for fold in range(10):
        target = [station for station in problem.stations if int(station_group[station]) == fold]
        station_mask = tensor([station not in target for station in problem.stations], torch.bool)
        for candidate in candidates:
            stem = f"fold_{fold:02d}_{candidate}"
            npz_path, json_path = CACHE / f"{stem}.npz", CACHE / f"{stem}.json"
            if npz_path.exists() and json_path.exists():
                saved = np.load(npz_path)
                base = tensor(saved["base"])
                gamma = None if "gamma" not in saved.files else tensor(saved["gamma"])
                audit = json.loads(json_path.read_text(encoding="utf-8"))
                print(f"recovered {stem}", flush=True)
            else:
                base, gamma, audit = fit(problem, attributes, station_mask, candidate, contract)
                audit.update({"spatial_group": fold, "training_station_count": int(station_mask.sum().cpu()), "target_station_count": len(target)})
                arrays = {"base": base.cpu().numpy()}
                if gamma is not None:
                    arrays["gamma"] = gamma.cpu().numpy()
                np.savez_compressed(npz_path, **arrays)
                json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            fit_rows.append(audit)
            parameters = parameter_field(base, gamma, attributes)
            simulation = problem.forward(parameters)
            strict_prediction_parts.append(strict_target_predictions(problem, simulation, target, candidate, fold))
            strict_proxy_parts.append(strict_target_proxy(problem, simulation, target, candidate, fold))
            if gamma is not None:
                gamma_np = gamma.cpu().numpy()
                for i, name in enumerate(attribute_names):
                    for j, physical in enumerate(["prod_capacity", "runoff_gamma", "quick_rho", "base_release", "base_rho"]):
                        coefficient_rows.append({"scope": "spatial_fold", "spatial_group": fold, "candidate": candidate, "attribute": name, "physical_parameter": physical, "raw_coefficient": float(gamma_np[i, j])})
            print(f"fold {fold} {candidate} complete objective={audit['objective']:.6f}", flush=True)
    strict_predictions = pd.concat(strict_prediction_parts, ignore_index=True)
    strict_proxy = pd.concat(strict_proxy_parts, ignore_index=True)
    strict_predictions.to_parquet(OUT / "zero_target_history_predictions.parquet", index=False)
    strict_proxy.to_parquet(OUT / "zero_target_history_path_predictions.parquet", index=False)
    pd.DataFrame(fit_rows).to_parquet(OUT / "spatial_fold_solver_parameters.parquet", index=False)
    flow_rows, flow_station_parts, path_rows, path_station_parts = [], [], [], []
    for candidate in candidates:
        for period in ["development_2006_2018", "check_2019_2022"]:
            row, station = flow_summary(strict_predictions, candidate, period)
            flow_rows.append(row)
            flow_station_parts.append(station)
        path, path_station = strict_path_summary(strict_proxy, candidate)
        path_rows.append(path)
        path_station_parts.append(path_station)
    flow = pd.DataFrame(flow_rows)
    flow_station = pd.concat(flow_station_parts, ignore_index=True)
    paths = pd.DataFrame(path_rows)
    path_station = pd.concat(path_station_parts, ignore_index=True)
    flow.to_parquet(OUT / "zero_target_history_flow_metrics.parquet", index=False)
    flow_station.to_parquet(OUT / "zero_target_history_station_flow_metrics.parquet", index=False)
    paths.to_parquet(OUT / "zero_target_history_path_metrics.parquet", index=False)
    path_station.to_parquet(OUT / "zero_target_history_station_path_metrics.parquet", index=False)
    # Full-development fit is independent of the held-out tests and creates the candidate field for temporal auditing.
    full_mask = torch.ones(len(problem.stations), dtype=torch.bool, device=DEVICE)
    full_rows, full_reach_parts, full_prediction_parts = [], [], []
    for candidate in candidates:
        stem = f"full_{candidate}"
        npz_path, json_path = CACHE / f"{stem}.npz", CACHE / f"{stem}.json"
        if npz_path.exists() and json_path.exists():
            saved = np.load(npz_path)
            base = tensor(saved["base"])
            gamma = None if "gamma" not in saved.files else tensor(saved["gamma"])
            audit = json.loads(json_path.read_text(encoding="utf-8"))
            print(f"recovered {stem}", flush=True)
        else:
            base, gamma, audit = fit(problem, attributes, full_mask, candidate, contract)
            audit.update({"spatial_group": -1, "training_station_count": len(problem.stations), "target_station_count": 0})
            arrays = {"base": base.cpu().numpy()}
            if gamma is not None:
                arrays["gamma"] = gamma.cpu().numpy()
            np.savez_compressed(npz_path, **arrays)
            json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        full_rows.append(audit)
        parameters = parameter_field(base, gamma, attributes)
        simulation = problem.forward(parameters)
        _, beta = profile_training_gauge(problem, simulation, full_mask)
        predictions = gauge_predict_all(problem, simulation, beta)
        predictions["candidate"] = candidate
        full_prediction_parts.append(predictions)
        reach = reach_output(problem, simulation, candidate)
        p = parameters
        for name in ["prod_capacity", "runoff_gamma", "quick_rho", "base_release", "base_rho"]:
            value = getattr(p, name).detach().cpu().numpy()
            if np.ndim(value) == 0:
                value = np.full(len(problem.reaches), float(value))
            reach[name] = np.repeat(value, len(problem.time))
        full_reach_parts.append(reach)
        if gamma is not None:
            gamma_np = gamma.cpu().numpy()
            for i, name in enumerate(attribute_names):
                for j, physical in enumerate(["prod_capacity", "runoff_gamma", "quick_rho", "base_release", "base_rho"]):
                    coefficient_rows.append({"scope": "full_development", "spatial_group": -1, "candidate": candidate, "attribute": name, "physical_parameter": physical, "raw_coefficient": float(gamma_np[i, j])})
        print(f"full {candidate} complete objective={audit['objective']:.6f}", flush=True)
    pd.DataFrame(full_rows).to_parquet(OUT / "full_development_solver_parameters.parquet", index=False)
    pd.DataFrame(coefficient_rows).to_parquet(OUT / "attribute_physical_coefficients.parquet", index=False)
    pd.concat(full_reach_parts, ignore_index=True).to_parquet(OUT / "full_development_reach_hydrology.parquet", index=False)
    full_predictions = pd.concat(full_prediction_parts, ignore_index=True)
    full_predictions.to_parquet(OUT / "full_development_station_predictions.parquet", index=False)
    temporal_rows = []
    for candidate in candidates:
        part = full_predictions[full_predictions.candidate.eq(candidate)]
        for layer, column in [("LATENT_SUPPORT", "latent_support_m3_s"), ("GAUGE_OBSERVATION", "gauge_prediction_m3_s")]:
            for period in ["development_2006_2018", "check_2019_2022"]:
                pooled, _ = summarize_predictions(part, column, candidate, layer, period)
                temporal_rows.append(pooled)
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_parquet(OUT / "full_development_temporal_metrics.parquet", index=False)
    # Registered spatial gates compare the transferable attribute field to an independently refit global field.
    dev = flow[flow.period.eq("development_2006_2018")].set_index("candidate")
    r0, r1 = dev.loc["R0_GLOBAL_REFIT"], dev.loc["R1_ATTRIBUTE10"]
    gate = contract["hard_gates"]
    total_gates = {
        "pooled_NSE_noninferior": bool(r0.NSE - r1.NSE <= float(gate["total_pooled_NSE_decrease_max"])),
        "station_median_NSE_noninferior": bool(r0.station_median_NSE - r1.station_median_NSE <= float(gate["total_station_median_NSE_decrease_max"])),
        "station_macro_log_RMSE_noninferior": bool(r1.station_mean_RMSE_log - r0.station_mean_RMSE_log <= float(gate["total_station_macro_log_RMSE_increase_max"])),
        "PBIAS_noninferior": bool(abs(r1.PBIAS_pct) - abs(r0.PBIAS_pct) <= float(gate["absolute_PBIAS_deterioration_max_percentage_points"])),
    }
    p1 = paths.set_index("candidate").loc["R1_ATTRIBUTE10"]
    path_gates = {
        "median_BFI_absolute_error": bool(p1.median_BFI_absolute_error <= float(gate["median_BFI_absolute_error_max"])),
        "fraction_station_error": bool(p1.fraction_stations_error_le_0_25 >= float(gate["fraction_station_BFI_error_le_0_25_min"])),
        "station_BFI_spearman": bool(p1.station_spearman >= float(gate["station_BFI_spearman_min"])),
        "median_seasonal_correlation": bool(p1.median_seasonal_correlation >= float(gate["median_seasonal_correlation_min"])),
        "median_peak_month_distance": bool(p1.median_peak_month_distance <= float(gate["median_peak_month_distance_max"])),
    }
    paired = path_station.pivot(index="station_norm", columns="candidate", values="absolute_mean_error").dropna()
    improvement = bootstrap_paired(paired.R1_ATTRIBUTE10.to_numpy() - paired.R0_GLOBAL_REFIT.to_numpy())
    path_gates["paired_BFI_absolute_error_improved"] = bool(improvement["ci95_upper"] < 0)
    methods = [p1.bias_vs_lh, p1.bias_vs_eckhardt, p1.bias_vs_ukih]
    path_gates["separation_methods_direction_consistent"] = bool(not (min(methods) < 0 < max(methods)))
    numerical_pass = bool(np.isfinite(pd.DataFrame(fit_rows)[["objective", "total_loss", "proxy_loss"]].to_numpy()).all())
    spatial_pass = bool(numerical_pass and all(total_gates.values()) and all(path_gates.values()))
    decision = {
        "stage": "20260823_31",
        "status": "ATTRIBUTE_REGIONALIZATION_PASS" if spatial_pass else "ATTRIBUTE_REGIONALIZATION_SPATIAL_OR_PATH_FAIL",
        "numerical_pass": numerical_pass,
        "total_flow_gates": total_gates,
        "path_gates": path_gates,
        "paired_path_improvement": improvement,
        "target_monthly_history_used_in_its_fold": False,
        "target_daily_proxy_used_in_its_fold": False,
        "free_station_physical_parameters": 0,
        "attribute_count": len(attribute_names),
        "spatial_pass": spatial_pass,
        "stage32_trigger_eligibility": "requires spatial failure plus graph-residual Moran p<0.05",
    }
    (REPORT / "stage31_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# 20260823_31 属性区域化物理水文\n\n"
        f"状态：`{decision['status']}`。每一空间组的月流量和日分割代理均从自身训练中完全删除。\n\n"
        "## 零目标历史总流量\n\n```text\n" + flow.to_string(index=False) + "\n```\n\n"
        "## 零目标历史路径划分\n\n```text\n" + paths.to_string(index=False) + "\n```\n\n"
        "## 门禁\n\n```json\n" + json.dumps({"total": total_gates, "path": path_gates}, ensure_ascii=False, indent=2) + "\n```\n"
    )
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "attributes_sha256": sha256(ATTRIBUTES),
        "groups_sha256": sha256(GROUPS),
        "predictions_sha256": sha256(OUT / "zero_target_history_predictions.parquet"),
        "reach_hydrology_sha256": sha256(OUT / "full_development_reach_hydrology.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(flow.to_string(index=False))
    print(paths.to_string(index=False))


if __name__ == "__main__":
    main()
