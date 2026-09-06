from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_8"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
H_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
F1_PATH = TEST / "20260824_4" / "scripts" / "run_f1_readout.py"
OLD_RETRO_PATH = TEST / "20260820_19" / "scripts" / "run_stage5_2022_retrospective.py"
Q72 = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
SNAPSHOT = OUT / "locked_2022_observations_snapshot.parquet"
RECEIPT = LOCKS / "TN_2022_source_read_receipt.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_locked_observations(h) -> pd.DataFrame:
    if RECEIPT.exists():
        if not SNAPSHOT.exists():
            raise RuntimeError("STOP_RECEIPT_WITHOUT_SNAPSHOT")
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if receipt["source_file_read_count"] != 1 or sha256(SNAPSHOT) != receipt["snapshot_sha256"]:
            raise RuntimeError("STOP_2022_SNAPSHOT_INTEGRITY")
        return pd.read_parquet(SNAPSHOT)

    observations = pd.read_parquet(OBS, filters=[("year", "==", 2022)]).copy()
    if len(observations) == 0 or set(observations.year.unique()) != {2022}:
        raise RuntimeError("STOP_2022_TN_BOUNDARY")
    domain = pd.read_parquet(h.DOMAIN, columns=["station_key", "primary_river_domain"]).drop_duplicates("station_key")
    observations = observations.merge(domain, on="station_key", how="left", validate="many_to_one")
    if observations.primary_river_domain.isna().any():
        raise RuntimeError("STOP_2022_DOMAIN_MISSING")
    observations.to_parquet(SNAPSHOT, index=False)
    receipt = {
        "event": "first_and_only_source_read_of_2022_TN_after_both_locks",
        "source_file": str(OBS),
        "source_file_sha256": sha256(OBS),
        "source_file_read_count": 1,
        "rows": len(observations),
        "stations": int(observations.station_key.nunique()),
        "snapshot": str(SNAPSHOT),
        "snapshot_sha256": sha256(SNAPSHOT),
        "development_mechanism_lock_sha256": sha256(LOCKS / "development_mechanism_lock.json"),
        "full_development_parameter_lock_sha256": sha256(LOCKS / "full_development_parameter_lock.json"),
        "TN_2022_values_read": True,
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return observations


def add_locked_q_feature(frame: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_parquet(
        Q72,
        filters=[("year", ">=", 2016), ("year", "<=", 2022)],
        columns=["reach_id", "year", "month", "q72_outlet_discharge_m3_s"],
    )
    q["log_q72"] = np.log(q.q72_outlet_discharge_m3_s.clip(lower=1e-12))
    median = q.loc[q.year.between(2016, 2021)].groupby("reach_id", observed=True).log_q72.median().rename("training_all_month_log_q_median")
    target = q.loc[q.year.eq(2022)]
    out = frame.merge(target, on=["reach_id", "year", "month"], validate="many_to_one")
    out = out.merge(median, on="reach_id", validate="many_to_one")
    if len(out) != len(frame) or out[["log_q72", "training_all_month_log_q_median"]].isna().any().any():
        raise RuntimeError("STOP_2022_Q_FEATURE")
    out["cq_z"] = out.log_q72 - out.training_all_month_log_q_median
    out["cq_low"] = np.minimum(out.cq_z, 0.0)
    out["cq_high"] = np.maximum(out.cq_z, 0.0)
    out["flow_state"] = np.select([out.cq_z < 0, out.cq_z > 0], ["below_training_median", "above_training_median"], default="at_training_median")
    return out


def basic_metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    error = pred - obs
    denom = float(np.sum(np.square(obs - obs.mean())))
    corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else math.nan
    return {
        "n": len(frame),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(error)))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse_mg_l": 1.0 - float(np.sum(np.square(error))) / denom if denom > 0 else math.nan,
        "pearson_r2_mg_l": corr * corr,
        "pbias_percent": 100.0 * float(np.sum(error)) / float(np.sum(obs)),
    }


def station_macro(frame: pd.DataFrame, anomaly: bool) -> float:
    values = []
    for _, group in frame.groupby("station_key", observed=True):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        pred = np.log1p(group.pred_tn_mg_l.to_numpy(float))
        if anomaly:
            obs = obs - obs.mean()
            pred = pred - pred.mean()
        values.append(float(np.sqrt(np.mean(np.square(pred - obs)))))
    return float(np.mean(values))


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    mechanism = json.loads((LOCKS / "development_mechanism_lock.json").read_text(encoding="utf-8"))
    parameter_lock = json.loads((LOCKS / "full_development_parameter_lock.json").read_text(encoding="utf-8"))
    if not mechanism["written_before_2022_TN_values_read"] or not parameter_lock["written_before_2022_TN_values_read"]:
        raise RuntimeError("STOP_LOCKS_NOT_WRITTEN")
    if mechanism["TN_2022_values_read"] or parameter_lock["TN_2022_values_read"]:
        raise RuntimeError("STOP_LOCK_DECLARATION_INVALID")
    for path, expected in parameter_lock["files"].items():
        if not Path(path).exists() or sha256(Path(path)) != expected:
            raise RuntimeError(f"STOP_PARAMETER_LOCK_HASH {path}")

    h = load_module(H_PATH, "hierarchical19_shared")
    f1 = load_module(F1_PATH, "final8_f1_retro")
    old_retro = load_module(OLD_RETRO_PATH, "final8_old_retro")
    shared = h.parent_shared()

    # Frozen Q72 + geometry work is completed before the only source-TN read.
    exposure = old_retro.build_2022_exposure()
    exposure.to_parquet(OUT / "reach_month_H1_exposure_2022.parquet", index=False)
    observations = read_locked_observations(h)
    parameters = pd.read_parquet(OUT / "full_development_parameters.parquet")
    h1_parameters = pd.read_parquet(OUT / "full_development_H1_parameters.parquet").set_index("model_id")
    effects = pd.read_parquet(OUT / "full_development_station_effects.parquet")

    pieces = []
    for model_id in h.FORMAL_MODELS:
        router = old_retro.build_2022_router(model_id, exposure, shared)
        vf = float(h1_parameters.loc[model_id, "v_f_m_per_day"])
        frame = add_locked_q_feature(router.frame(observations, np.full(len(router.reach_ids), vf)))
        for layer in ("P1", "P2"):
            for arm in ("GAUSSIAN_PROCESS_PARENT", "GAUSSIAN_CQ_HINGE"):
                row = parameters.loc[
                    parameters.model_id.eq(model_id) & parameters.layer.eq(layer) & parameters.arm.eq(arm)
                ]
                if len(row) != 1:
                    raise RuntimeError(f"STOP_LOCKED_PARAMETER_KEY {model_id} {layer} {arm}")
                row = row.iloc[0]
                effect_map = (
                    effects.loc[
                        effects.model_id.eq(model_id) & effects.layer.eq(layer) & effects.arm.eq(arm)
                    ].set_index("station_key").station_effect.to_dict()
                    if layer == "P2" else {}
                )
                fit = {
                    "eta": np.array([row.eta_quick, row.eta_gw], dtype=float),
                    "beta": np.array([row.beta_low, row.beta_high], dtype=float),
                    "effects": effect_map,
                }
                pred = f1.predict(frame, fit, layer)
                pred["model_id"] = model_id
                pred["layer"] = layer
                pred["arm"] = arm
                pred["fold_id"] = "R2022_LOCKED"
                pred["v_f_m_per_day"] = vf
                pred["retrospective_role"] = np.where(pred.primary_river_domain, "primary_river", "diagnostic_only")
                pieces.append(pred)
    predictions = pd.concat(pieces, ignore_index=True)
    predictions.to_parquet(OUT / "locked_2022_retrospective_predictions.parquet", index=False)
    primary = predictions.loc[predictions.primary_river_domain].copy()

    metric_rows = []
    for (model_id, layer, arm), frame in primary.groupby(["model_id", "layer", "arm"], observed=True):
        metric_rows.append({
            "model_id": model_id, "layer": layer, "arm": arm,
            **basic_metrics(frame),
            "station_macro_rmse_log1p": station_macro(frame, False),
            "station_macro_anomaly_rmse_log1p": station_macro(frame, True),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(OUT / "locked_2022_retrospective_metrics.parquet", index=False)

    residual_rows = []
    primary["signed_log_residual_obs_minus_pred"] = np.log1p(primary.tn_mg_l) - np.log1p(primary.pred_tn_mg_l)
    for keys, frame in primary.groupby(["model_id", "layer", "arm", "month", "flow_state"], observed=True):
        model_id, layer, arm, month, flow_state = keys
        residual_rows.append({
            "model_id": model_id, "layer": layer, "arm": arm, "month": int(month), "flow_state": flow_state,
            "n": len(frame),
            "mean_signed_log_residual_obs_minus_pred": float(frame.signed_log_residual_obs_minus_pred.mean()),
            "rmse_log1p": float(np.sqrt(np.mean(np.square(frame.signed_log_residual_obs_minus_pred)))),
        })
    residuals = pd.DataFrame(residual_rows)
    residuals.to_parquet(OUT / "locked_2022_month_flow_residuals.parquet", index=False)

    gate_rows = []
    seed = 202608240800
    for model_id in h.FORMAL_MODELS:
        for layer in ("P1", "P2"):
            subset = primary.loc[primary.model_id.eq(model_id) & primary.layer.eq(layer)]
            parent = subset.loc[subset.arm.eq("GAUSSIAN_PROCESS_PARENT")]
            candidate = subset.loc[subset.arm.eq("GAUSSIAN_CQ_HINGE")]
            for column, name in (("station_key", "station"), ("terminal_tree_id", "tree")):
                seed += 1
                gate_rows.append({
                    "model_id": model_id, "layer": layer, "block": name,
                    **h.paired_bootstrap(parent, candidate, column, seed),
                })
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(OUT / "locked_2022_retrospective_gates.parquet", index=False)
    tree163 = predictions.loc[predictions.terminal_tree_id.eq(163)].copy()
    tree163.to_parquet(OUT / "tree_163_locked_retrospective_diagnostic.parquet", index=False)

    mean_metrics = metrics.groupby(["layer", "arm"], as_index=False).mean(numeric_only=True)
    gate_counts = {
        f"{layer}_{block}_{field}": int(gates.loc[gates.layer.eq(layer) & gates.block.eq(block), field].sum())
        for layer in ("P1", "P2") for block in ("station", "tree")
        for field in ("noninferior", "predictively_improved")
    }
    report = {
        "status": "LOCKED_2022_RETROSPECTIVE_COMPLETE",
        "program_terminal_state": "MONITORED_STATION_PREDICTION_UPGRADE_ONLY",
        "architecture_changed_by_2022": False,
        "parameters_changed_by_2022": False,
        "TN_2022_source_file_read_count": 1,
        "TN_rows": int(len(observations)),
        "primary_river_rows": int(observations.primary_river_domain.sum()),
        "diagnostic_only_rows": int((~observations.primary_river_domain).sum()),
        "primary_stations": int(observations.loc[observations.primary_river_domain, "station_key"].nunique()),
        "tree_163_rows": int(len(tree163)),
        "gate_counts": gate_counts,
        "mean_metrics": mean_metrics.to_dict("records"),
        "interpretation": "2022 is a locked post-development retrospective check and cannot alter the development decision",
        "spatial_transfer_status": "NOT_SUPPORTED_FROM_DEVELOPMENT_NESTED_LOTO",
        "temperature_used": False,
        "TN_2022_values_read": True,
    }
    (REPORTS / "locked_2022_retrospective_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "final_program_decision.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 20260824月尺度TN结构程序最终报告", "",
        "最终状态：`MONITORED_STATION_PREDICTION_UPGRADE_ONLY`。", "",
        "锁定架构为 `Q72 → F00 → S0/S1-12 × 六个μ → fixed T1 → H1_GLOBAL → Gaussian Q72 C–Q hinge`。F1在2018–2021时间OOF通过，但nested LOTO和绝对空间skill未通过，因此升级只适用于已有监测站的月尺度时间预测。", "",
        "## 锁定2022回顾效果（12成员均值）", "",
        "| Layer | Arm | RMSE | MAE | NSE | R² | PBIAS % | station-macro log-RMSE | anomaly log-RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in ("P1", "P2"):
        table = mean_metrics.loc[mean_metrics.layer.eq(layer)].set_index("arm")
        for arm in ("GAUSSIAN_PROCESS_PARENT", "GAUSSIAN_CQ_HINGE"):
            row = table.loc[arm]
            lines.append(
                f"| {layer} | {arm} | {row.rmse_mg_l:.4f} | {row.mae_mg_l:.4f} | {row.nse_mg_l:.4f} | "
                f"{row.pearson_r2_mg_l:.4f} | {row.pbias_percent:.2f} | {row.station_macro_rmse_log1p:.4f} | "
                f"{row.station_macro_anomaly_rmse_log1p:.4f} |"
            )
    lines.extend(["", "## 证据边界", ""])
    lines.extend([
        "- 2018–2021 OOF支持C–Q hinge修正高流量响应。",
        "- 2016–2021全开发期重新拟合在2022 TN读取前完成并锁定。",
        "- 2022只作锁定回顾检查，不用于改结构、重估参数或重写门禁。",
        "- F2农田缓释接口候选和F3河道旅行时间候选均未获支持；温度未开启。",
        "- tree 163是抚仙湖心开放水体诊断域，不进入河流主门禁。",
        "- 未监测河段空间外推仍未解决；不能把P2站点修正当作230 Reach预测能力。",
        "",
        "## 下一阶段含义", "",
        "当前最清楚的剩余问题是空间异质性与source identity/availability的可识别表达，而不是继续增加地下水年龄或河道反应自由度。任何新结构必须重新注册，并以nested空间评价为硬约束。",
    ])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
