from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
STAGE9_PATH = TEST / "20260823_9" / "scripts" / "run_candidate_implementation.py"
STAGE9_AUDIT = TEST / "20260823_9" / "reports" / "candidate_implementation_audit.json"
STAGE9_PARAMS = TEST / "20260823_9" / "outputs" / "candidate_parameters.parquet"
S111 = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
ARCHIVE_AUDIT = TEST / "20260823_8" / "reports" / "discharge_archive_independence_audit.json"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EPS = 1.0e-12
SEED = 20260823
NBOOT = 10000
FOLDS = [
    {"fold_id": "F1", "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "F2", "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "F3", "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]
MODELS = [
    "P0_Q72_PROCESS", "P1_GLOBAL_COMPACT", "P2_COMPACT_ZERO_HISTORY_COLDSTART",
    "REGIONALIZED_ROUTED_CORRECTION", "P2_36_MONTH_GAUGED_ADAPTATION",
]


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


S9 = load_file("stage10_stage9", STAGE9_PATH)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def kge2012(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) <= EPS or np.mean(obs) <= EPS or np.mean(pred) <= EPS:
        return np.nan
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float((np.std(pred, ddof=1) / np.mean(pred)) / (np.std(obs, ddof=1) / np.mean(obs)))
    beta = float(np.mean(pred) / np.mean(obs))
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def one_series_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    lo, lp = np.log1p(obs.clip(min=0)), np.log1p(pred.clip(min=0))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    lsst = float(np.sum((lo - lo.mean()) ** 2))
    return {
        "NSE": float(1 - np.sum((pred - obs) ** 2) / sst) if sst > EPS else np.nan,
        "log_NSE": float(1 - np.sum((lp - lo) ** 2) / lsst) if lsst > EPS else np.nan,
        "KGE_2012": kge2012(obs, pred),
        "RMSE_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "log_RMSE": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "PBIAS_pct": float(100 * np.sum(pred - obs) / np.sum(obs)) if np.sum(obs) > EPS else np.nan,
    }


def metric_rows(frame: pd.DataFrame, population: str = "S000") -> list[dict[str, object]]:
    rows = []
    for (mode, model), part in frame.groupby(["spatial_mode", "model_id"]):
        pooled = one_series_metrics(part["actual_cfs"].to_numpy(float), part["predict_cfs"].to_numpy(float))
        station = []
        for site, group in part.groupby("q_site"):
            values = one_series_metrics(group["actual_cfs"].to_numpy(float), group["predict_cfs"].to_numpy(float))
            values["q_site"] = str(site)
            station.append(values)
        station_frame = pd.DataFrame(station)
        row = {"population": population, "spatial_mode": mode, "model_id": model, "rows": len(part), "stations": part["q_site"].nunique(), "trees": part["terminal_tree"].nunique()}
        row.update({f"pooled_{key}": value for key, value in pooled.items()})
        for key in ["NSE", "log_NSE", "KGE_2012", "RMSE_cfs", "log_RMSE", "PBIAS_pct"]:
            row[f"station_mean_{key}"] = float(station_frame[key].mean())
            row[f"station_median_{key}"] = float(station_frame[key].median())
        lo = np.log1p(part["actual_cfs"].clip(lower=0)) - np.log1p(part["predict_cfs"].clip(lower=0))
        annual = pd.DataFrame({"year": part["year"].to_numpy(), "residual": lo}).groupby("year")["residual"].mean()
        row["J_year"] = float(np.sqrt(np.mean((annual - annual.mean()) ** 2)))
        rows.append(row)
    return rows


def fit_global_and_cold_p2(work: pd.DataFrame, train_mask: np.ndarray, eval_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_all = S9.compact_global_basis(work)
    x = x_all[train_mask]
    y = np.log1p(work.loc[train_mask, "Q_obsv_cfs"].to_numpy(float))
    gpen = np.r_[0.0, np.full(x.shape[1] - 1, 1 / 1.5)]
    ag = x.T @ x + np.diag(gpen**2)
    bg = x.T @ y
    beta_p1 = np.linalg.solve(ag, bg)

    # Exact Schur-complement solution of the same Gaussian-prior MAP model.
    schur_a = ag.copy()
    schur_b = bg.copy()
    train = work.loc[train_mask, ["q_site", "month", "routed_quick_cfs", "routed_base_cfs", "antecedent_wetness"]].copy()
    local_all = np.column_stack([
        np.ones(len(train)),
        np.sin(2 * np.pi * train["month"].to_numpy(float) / 12.0),
        np.cos(2 * np.pi * train["month"].to_numpy(float) / 12.0),
        np.log1p((train["routed_quick_cfs"] + train["routed_base_cfs"]).clip(lower=0).to_numpy(float)),
        train["antecedent_wetness"].to_numpy(float),
    ])
    names = train["q_site"].astype(str).to_numpy()
    for station in np.unique(names):
        idx = np.flatnonzero(names == station)
        xs, zs, ys = x[idx], local_all[idx], y[idx]
        c = zs.T @ zs + np.eye(5) * (1 / 0.5) ** 2
        ci = np.linalg.inv(c)
        d = xs.T @ zs
        schur_a -= d @ ci @ d.T
        schur_b -= d @ ci @ (zs.T @ ys)
    beta_p2_global = np.linalg.solve(schur_a, schur_b)
    return x_all[eval_mask] @ beta_p1, x_all[eval_mask] @ beta_p2_global


class RegionalFitter:
    def __init__(self, work: pd.DataFrame, basis: np.ndarray, local_q: np.ndarray, local_s: np.ndarray, aggregate: np.ndarray):
        self.work = work
        self.basis = basis
        self.local_q = local_q
        self.local_s = local_s
        self.aggregate = aggregate
        mask = work["Q_obsv_cfs"].gt(0).to_numpy()
        positions = np.flatnonzero(mask)
        self.obs_rows = positions
        self.ti = positions // 230
        self.ri = positions % 230
        self.obs = work.loc[mask, "Q_obsv_cfs"].to_numpy(float)
        self.years = work.loc[mask, "year"].to_numpy(int)
        self.stations = work.loc[mask, "q_site"].astype(str).to_numpy()
        self.trees = work.loc[mask, "terminal_tree"].astype(int).to_numpy()
        self.prior_mean = np.zeros(16, float)
        self.prior_mean[[0, 8]] = S9.BASE_V
        self.prior_sd = np.full(16, 0.30, float)
        self.prior_sd[[0, 8]] = 0.50

    def residual_jacobian(self, theta: np.ndarray, selected: np.ndarray, need_jac: bool):
        ti, ri, obs = self.ti[selected], self.ri[selected], self.obs[selected]
        zq = np.einsum("trk,k->tr", self.basis, theta[:8])
        zs = np.einsum("trk,k->tr", self.basis, theta[8:])
        sq, ss = expit(zq), expit(zs)
        cq, cs = 0.25 + 3.75 * sq, 0.25 + 3.75 * ss
        dcq, dcs = 3.75 * sq * (1 - sq), 3.75 * ss * (1 - ss)
        residual = np.empty(len(selected), float)
        jac = np.empty((len(selected), 16), float) if need_jac else None
        for t in np.unique(ti):
            out_pos = np.flatnonzero(ti == t)
            upstream = self.aggregate[ri[out_pos]]
            uq = upstream * self.local_q[t][None, :]
            us = upstream * self.local_s[t][None, :]
            pred = uq @ cq[t] + us @ cs[t]
            residual[out_pos] = np.log1p(pred.clip(min=0)) - np.log1p(obs[out_pos])
            if need_jac:
                denominator = (1.0 + pred)[:, None]
                jac[out_pos, :8] = ((uq * dcq[t][None, :]) @ self.basis[t]) / denominator
                jac[out_pos, 8:] = ((us * dcs[t][None, :]) @ self.basis[t]) / denominator
        prior = (theta - self.prior_mean) / self.prior_sd / 4.0
        residual = np.r_[residual, prior]
        if need_jac:
            jac = np.vstack([jac, np.diag(1.0 / self.prior_sd / 4.0)])
            return residual, jac
        return residual

    def fit(self, selected: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        result = least_squares(
            lambda theta: self.residual_jacobian(theta, selected, False), start,
            jac=lambda theta: self.residual_jacobian(theta, selected, True)[1],
            method="trf", max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        theta = result.x
        cq = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", self.basis, theta[:8]))
        cs = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", self.basis, theta[8:]))
        near = float((np.mean((cq < 0.275) | (cq > 3.9)) + np.mean((cs < 0.275) | (cs > 3.9))) / 2)
        return theta, {"success": bool(result.success), "cost": float(result.cost), "nfev": int(result.nfev), "optimality": float(result.optimality), "near_boundary_fraction": near}

    def predict(self, theta: np.ndarray, selected: np.ndarray) -> np.ndarray:
        ti, ri = self.ti[selected], self.ri[selected]
        cq = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", self.basis, theta[:8]))
        cs = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", self.basis, theta[8:]))
        pred = np.empty(len(selected), float)
        for t in np.unique(ti):
            pos = np.flatnonzero(ti == t)
            upstream = self.aggregate[ri[pos]]
            pred[pos] = (upstream * self.local_q[t][None, :]) @ cq[t] + (upstream * self.local_s[t][None, :]) @ cs[t]
        return pred


def gauged_36_month_prediction(work: pd.DataFrame, fold: dict[str, int], eval_rows: np.ndarray) -> np.ndarray:
    result = np.empty(len(eval_rows), float)
    eval_frame = work.iloc[eval_rows]
    start_ordinal = int(fold["eval_start"]) * 12 + 1
    lower = start_ordinal - 36
    base_total = (work["routed_quick_cfs"] + work["routed_base_cfs"]).clip(lower=0)
    ordinal = work["year"].astype(int) * 12 + work["month"].astype(int)
    for station, positions in eval_frame.groupby(eval_frame["q_site"].astype(str)).indices.items():
        positions = np.asarray(positions, int)
        target_rows = eval_rows[positions]
        train = work["q_site"].astype(str).eq(station) & work["Q_obsv_cfs"].gt(0) & ordinal.ge(lower) & ordinal.lt(start_ordinal)
        if train.sum() < 6:
            result[positions] = base_total.iloc[target_rows].to_numpy(float)
            continue
        month = work.loc[train, "month"].to_numpy(float)
        q0 = base_total.loc[train].to_numpy(float)
        wet = work.loc[train, "antecedent_wetness"].to_numpy(float)
        center = float(np.mean(np.log1p(q0)))
        x = np.column_stack([np.ones(train.sum()), np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), np.log1p(q0)-center, wet])
        y = np.log1p(work.loc[train, "Q_obsv_cfs"].to_numpy(float)) - np.log1p(q0)
        penalty = np.diag([0.5**2, 2.0**2, 2.0**2, 2.0**2, 2.0**2])
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        em = work.iloc[target_rows]["month"].to_numpy(float)
        eq = base_total.iloc[target_rows].to_numpy(float)
        ew = work.iloc[target_rows]["antecedent_wetness"].to_numpy(float)
        xe = np.column_stack([np.ones(len(target_rows)), np.sin(2*np.pi*em/12), np.cos(2*np.pi*em/12), np.log1p(eq)-center, ew])
        result[positions] = np.expm1(np.log1p(eq) + xe @ beta).clip(min=0)
    return result


def add_regime(predictions: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    thresholds = []
    q0 = (work["routed_quick_cfs"] + work["routed_base_cfs"]).clip(lower=0)
    temp = work[["q_site", "year"]].copy()
    temp["q0"] = q0
    for fold in FOLDS:
        subset = temp[temp["year"].le(fold["train_end"]) & temp["q_site"].notna()]
        quant = subset.groupby(subset["q_site"].astype(str))["q0"].quantile([0.2, 0.8]).unstack()
        for station, row in quant.iterrows():
            thresholds.append({"fold_id": fold["fold_id"], "q_site": str(station), "q20": row.get(0.2, np.nan), "q80": row.get(0.8, np.nan)})
    out = predictions.merge(pd.DataFrame(thresholds), on=["fold_id", "q_site"], how="left", validate="many_to_one")
    parent = out[out["model_id"].eq("P0_Q72_PROCESS")][["spatial_mode", "fold_id", "target", "reach_id", "q_site", "year", "month", "predict_cfs"]].rename(columns={"predict_cfs": "q72_for_regime"})
    out = out.merge(parent, on=["spatial_mode", "fold_id", "target", "reach_id", "q_site", "year", "month"], how="left", validate="many_to_one")
    out["flow_regime"] = np.where(out["q72_for_regime"].le(out["q20"]), "low", np.where(out["q72_for_regime"].ge(out["q80"]), "high", "middle"))
    return out


def paired_bootstrap(frame: pd.DataFrame, candidate: str, reference: str, mode: str, regime: str = "all") -> dict[str, object]:
    keys = ["fold_id", "target", "reach_id", "q_site", "terminal_tree", "year", "month", "actual_cfs"]
    c = frame[(frame["spatial_mode"] == mode) & (frame["model_id"] == candidate)]
    r = frame[(frame["spatial_mode"] == mode) & (frame["model_id"] == reference)]
    if regime != "all":
        c, r = c[c["flow_regime"] == regime], r[r["flow_regime"] == regime]
    joined = c[keys + ["predict_cfs"]].rename(columns={"predict_cfs": "candidate"}).merge(
        r[keys + ["predict_cfs"]].rename(columns={"predict_cfs": "reference"}), on=keys, validate="one_to_one")
    joined["c_loss"] = (np.log1p(joined["candidate"].clip(lower=0)) - np.log1p(joined["actual_cfs"].clip(lower=0))) ** 2
    joined["r_loss"] = (np.log1p(joined["reference"].clip(lower=0)) - np.log1p(joined["actual_cfs"].clip(lower=0))) ** 2
    cluster = "q_site" if mode == "LOSO" else "terminal_tree"
    loss = joined.groupby(cluster).agg(c_loss=("c_loss", "mean"), r_loss=("r_loss", "mean"), rows=("actual_cfs", "size")).reset_index()
    if len(loss) == 0:
        return {"candidate": candidate, "reference": reference, "spatial_mode": mode, "regime": regime, "clusters": 0, "delta_log_RMSE": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan}
    cv, rv = loss["c_loss"].to_numpy(float), loss["r_loss"].to_numpy(float)
    rng = np.random.default_rng(SEED + sum(map(ord, candidate + reference + mode + regime)))
    idx = rng.integers(0, len(loss), size=(NBOOT, len(loss)))
    delta = np.sqrt(cv[idx].mean(axis=1)) - np.sqrt(rv[idx].mean(axis=1))
    return {
        "candidate": candidate, "reference": reference, "spatial_mode": mode, "regime": regime,
        "clusters": int(len(loss)), "rows": int(len(joined)),
        "delta_log_RMSE": float(np.sqrt(cv.mean()) - np.sqrt(rv.mean())),
        "ci95_lower": float(np.quantile(delta, 0.025)), "ci95_upper": float(np.quantile(delta, 0.975)),
        "improved": bool(np.quantile(delta, 0.975) < 0),
        "noninferior_0p005": bool(np.quantile(delta, 0.975) < 0.005),
    }


def run_nested() -> tuple[pd.DataFrame, pd.DataFrame]:
    states, aggregate, _, _ = S9.build_states()
    pcs, _ = S9.static_pcs(states, training_end=2018)
    work, basis, local_q, local_s = S9.prepare_arrays(states, aggregate, pcs)
    fitter = RegionalFitter(work, basis, local_q, local_s, aggregate)
    parent_theta = np.zeros(16, float)
    parent_theta[[0, 8]] = S9.BASE_V
    rows, fit_rows = [], []
    for fold in FOLDS:
        train_base = fitter.years <= fold["train_end"]
        eval_base = (fitter.years >= fold["eval_start"]) & (fitter.years <= fold["eval_end"])
        full_idx = np.flatnonzero(train_base)
        fold_theta, fold_diag = fitter.fit(full_idx, parent_theta)
        fit_rows.append({"fold_id": fold["fold_id"], "spatial_mode": "WARM_START", "target": "all", **fold_diag, **{f"theta_{i}": float(v) for i, v in enumerate(fold_theta)}})
        for mode in ["LOSO", "LOTO"]:
            targets = sorted(np.unique(fitter.stations[eval_base]).tolist()) if mode == "LOSO" else sorted(np.unique(fitter.trees[eval_base]).tolist())
            for number, target in enumerate(targets, 1):
                if mode == "LOSO":
                    train_obs = np.flatnonzero(train_base & (fitter.stations != str(target)))
                    eval_obs = np.flatnonzero(eval_base & (fitter.stations == str(target)))
                else:
                    train_obs = np.flatnonzero(train_base & (fitter.trees != int(target)))
                    eval_obs = np.flatnonzero(eval_base & (fitter.trees == int(target)))
                train_rows_mask = work["Q_obsv_cfs"].gt(0).to_numpy() & work["year"].le(fold["train_end"]).to_numpy()
                if mode == "LOSO":
                    train_rows_mask &= work["q_site"].astype(str).ne(str(target)).to_numpy()
                else:
                    train_rows_mask &= work["terminal_tree"].ne(int(target)).to_numpy()
                eval_rows = fitter.obs_rows[eval_obs]
                eval_rows_mask = np.zeros(len(work), bool)
                eval_rows_mask[eval_rows] = True
                p1_log, p2_log = fit_global_and_cold_p2(work, train_rows_mask, eval_rows_mask)
                theta, diag = fitter.fit(train_obs, fold_theta)
                reg = fitter.predict(theta, eval_obs)
                adapt = gauged_36_month_prediction(work, fold, eval_rows)
                base = work.iloc[eval_rows]
                actual = base["Q_obsv_cfs"].to_numpy(float)
                predictions = {
                    "P0_Q72_PROCESS": (base["routed_quick_cfs"] + base["routed_base_cfs"]).to_numpy(float),
                    "P1_GLOBAL_COMPACT": np.expm1(np.clip(p1_log, -20, 20)).clip(min=0),
                    "P2_COMPACT_ZERO_HISTORY_COLDSTART": np.expm1(np.clip(p2_log, -20, 20)).clip(min=0),
                    "REGIONALIZED_ROUTED_CORRECTION": reg.clip(min=0),
                    "P2_36_MONTH_GAUGED_ADAPTATION": adapt.clip(min=0),
                }
                for model, pred in predictions.items():
                    for i, (_, source) in enumerate(base.iterrows()):
                        rows.append({
                            "spatial_mode": mode, "fold_id": fold["fold_id"], "target": str(target), "model_id": model,
                            "reach_id": int(source["comid"]), "q_site": str(source["q_site"]), "terminal_tree": int(source["terminal_tree"]),
                            "year": int(source["year"]), "month": int(source["month"]), "actual_cfs": float(actual[i]), "predict_cfs": float(pred[i]),
                            "target_history_used": bool(model == "P2_36_MONTH_GAUGED_ADAPTATION"),
                        })
                fit_rows.append({"fold_id": fold["fold_id"], "spatial_mode": mode, "target": str(target), **diag, **{f"theta_{i}": float(v) for i, v in enumerate(theta)}})
                if number % 10 == 0 or number == len(targets):
                    print(f"[{fold['fold_id']}] {mode} {number}/{len(targets)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(fit_rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions, fits = run_nested()
    predictions = add_regime(predictions, pd.read_parquet(TEST / "20260823_9" / "outputs" / "hydrologic_candidate_panel_230_reaches_2006_2022.parquet").rename(columns={"reach_id": "comid", "observed_cfs": "Q_obsv_cfs", "q72_quick_cfs": "routed_quick_cfs", "q72_slow_cfs": "routed_base_cfs"}))
    predictions.to_parquet(OUTPUTS / "nested_loso_loto_predictions.parquet", index=False)
    fits.to_parquet(OUTPUTS / "nested_regionalized_parameters.parquet", index=False)

    metrics = pd.DataFrame(metric_rows(predictions, "S000"))
    s111_keys = pd.read_parquet(S111).loc[lambda x: x["Q_obsv_cfs"].gt(0), ["comid", "year", "month"]].drop_duplicates().rename(columns={"comid": "reach_id"})
    s111_predictions = predictions.merge(s111_keys.assign(in_s111=True), on=["reach_id", "year", "month"], how="inner")
    metrics = pd.concat([metrics, pd.DataFrame(metric_rows(s111_predictions, "S111_subset_sensitivity"))], ignore_index=True)
    metrics.to_parquet(OUTPUTS / "nested_model_metrics.parquet", index=False)

    boot = []
    for mode in ["LOSO", "LOTO"]:
        for candidate in ["P1_GLOBAL_COMPACT", "P2_COMPACT_ZERO_HISTORY_COLDSTART", "REGIONALIZED_ROUTED_CORRECTION", "P2_36_MONTH_GAUGED_ADAPTATION"]:
            for reference in ["P0_Q72_PROCESS", "P1_GLOBAL_COMPACT"]:
                if candidate == reference:
                    continue
                for regime in ["all", "low", "high"]:
                    boot.append(paired_bootstrap(predictions, candidate, reference, mode, regime))
    bootstrap = pd.DataFrame(boot)
    bootstrap.to_parquet(OUTPUTS / "paired_spatial_bootstrap.parquet", index=False)

    fit_valid = fits[fits["spatial_mode"].isin(["LOSO", "LOTO"])]
    stage9 = json.loads(STAGE9_AUDIT.read_text(encoding="utf-8"))
    reg_gate_rows = bootstrap[(bootstrap["candidate"] == "REGIONALIZED_ROUTED_CORRECTION")]
    primary = reg_gate_rows[(reg_gate_rows["regime"] == "all") & reg_gate_rows["reference"].isin(["P0_Q72_PROCESS", "P1_GLOBAL_COMPACT"])]
    regimes = reg_gate_rows[(reg_gate_rows["regime"].isin(["low", "high"])) & reg_gate_rows["reference"].isin(["P0_Q72_PROCESS", "P1_GLOBAL_COMPACT"])]
    loso_metric = metrics[(metrics["population"] == "S000") & (metrics["spatial_mode"] == "LOSO")].set_index("model_id")
    pooled_nse_gate = bool(loso_metric.loc["REGIONALIZED_ROUTED_CORRECTION", "pooled_NSE"] >= loso_metric.loc["P0_Q72_PROCESS", "pooled_NSE"] - 0.005)
    median_nse_gate = bool(loso_metric.loc["REGIONALIZED_ROUTED_CORRECTION", "station_median_NSE"] > loso_metric.loc["P0_Q72_PROCESS", "station_median_NSE"])
    jyear_gate = bool(loso_metric.loc["REGIONALIZED_ROUTED_CORRECTION", "J_year"] < loso_metric.loc["P0_Q72_PROCESS", "J_year"])
    gates = {
        "nested_overall_improves_vs_P0_and_P1": bool(primary["improved"].all()),
        "pooled_raw_NSE_drop_le_0p005": pooled_nse_gate,
        "station_median_NSE_above_P0": median_nse_gate,
        "low_high_flow_noninferior": bool(regimes["noninferior_0p005"].all()),
        "J_year_below_P0": jyear_gate,
        "routing_and_parent_closure_le_1e-9": bool(stage9["parent_endpoint_relative_error"] <= 1e-9 and stage9["production_mass_balance_max_abs_mm"] <= 1e-9),
        "nested_near_boundary_fraction_le_0p05": bool(fit_valid["near_boundary_fraction"].max() <= 0.05),
        "solver_not_confounded": bool(fit_valid["success"].all() and not stage9["solver"]["optimizer_confounded"]),
    }
    regional_supported = bool(all(gates.values()))

    cold = bootstrap[(bootstrap["candidate"] == "P2_COMPACT_ZERO_HISTORY_COLDSTART") & (bootstrap["regime"] == "all")]
    p2_spatial = {
        "is_statistical_simulation_or_fit": True,
        "model_class": "station-conditioned Gaussian-prior MAP with target station terms forced to zero",
        "eligible_for_mainline": False,
        "nested_improves_vs_P0_all": bool(cold[cold["reference"] == "P0_Q72_PROCESS"]["improved"].all()),
        "nested_noninferior_vs_P1_all": bool(cold[cold["reference"] == "P1_GLOBAL_COMPACT"]["noninferior_0p005"].all()),
        "interpretation": "cold-start performance measures the transferable global part only; the large same-station gain belongs to gauged-site adaptation",
    }
    adaptation = bootstrap[(bootstrap["candidate"] == "P2_36_MONTH_GAUGED_ADAPTATION") & (bootstrap["regime"] == "all")]
    history = {
        "role": "gauged-site adaptation upper bound only",
        "uses_target_history": True,
        "eligible_for_230_reach_product": False,
        "improves_vs_P0_in_both_modes": bool(adaptation[adaptation["reference"] == "P0_Q72_PROCESS"]["improved"].all()),
        "metric_rows": adaptation.to_dict(orient="records"),
    }
    dump(REPORTS / "gauged_history_adaptation_diagnostic.json", history)

    tree163 = predictions[(predictions["spatial_mode"] == "LOTO") & (predictions["terminal_tree"] == 163)]
    stage9_panel = pd.read_parquet(TEST / "20260823_9" / "outputs" / "hydrologic_candidate_panel_230_reaches_2006_2022.parquet")
    tree163_source = stage9_panel[(stage9_panel["terminal_tree"] == 163) & stage9_panel["observed_cfs"].gt(0)]
    tree163_payload = {
        "tree": 163, "rows_per_model": int(len(tree163) / max(tree163["model_id"].nunique(), 1)),
        "source_observation_rows": int(len(tree163_source)),
        "source_observation_years": sorted(tree163_source["year"].astype(int).unique().tolist()),
        "source_station_count": int(tree163_source["q_site"].nunique()),
        "formal_2012_2018_oof_evaluable": bool(len(tree163) > 0),
        "metrics": metric_rows(tree163, "S000_tree_163") if len(tree163) else [],
        "interpretation": "Tree 163 has observations only in 2006-2007, so it has no 2012-2018 held-out rows and cannot enter the registered nested spatial gate.",
    }
    dump(REPORTS / "tree_163_audit.json", tree163_payload)

    archive = json.loads(ARCHIVE_AUDIT.read_text(encoding="utf-8"))
    replication = {
        "usable_independent_unseen_reaches": archive["usable_unseen_reaches"],
        "independent_external_spatial_validation_possible": bool(len(archive["usable_unseen_reaches"]) > 0),
        "same_reach_records_role": "replication_or_data_quality_only",
        "status": "NO_QUALIFYING_UNSEEN_REACH; nested S000 LOSO/LOTO remains the controlling spatial test",
    }
    dump(REPORTS / "discharge_archive_replication_audit.json", replication)

    decision = {
        "stage": "20260823_10", "status": "NESTED_EVALUATION_COMPLETE",
        "regionalized_gates": gates,
        "regionalized_routed_correction_supported": regional_supported,
        "eligible_stage11_product": "REGIONALIZED_ROUTED_CORRECTION" if regional_supported else "P0_Q72_PROCESS",
        "p2_coldstart_spatial_diagnostic": p2_spatial,
        "p2_36_month_role": "diagnostic_only_gauged_site_adaptation_upper_bound",
        "S111_role": "subset sensitivity only; no station deletion changes S000 primary decision",
        "next_authorized_stage": "20260823_11",
    }
    dump(REPORTS / "nested_spatial_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
