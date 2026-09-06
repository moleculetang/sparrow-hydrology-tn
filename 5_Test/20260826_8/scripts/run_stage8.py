from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_8"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(RUN / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from nested_worker import fit_fold  # noqa: E402
from run_stage5 import station_metrics  # noqa: E402

PARENT_OOF = ROOT / "5_Test" / "20260825_5" / "outputs" / "complete_tree_oof_predictions.parquet"
TREES = [1, 20, 22, 26, 56, 166, 212, 217]
MODELS = ["RAVEN_SACSMA3", "MTRS3"]


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def nse(o, p): return float(1 - np.sum((p-o)**2)/np.sum((o-o.mean())**2))


def metrics(model: str, frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    pivot_o = frame.pivot(index="date", columns="station_norm", values="q_observed_m3_s"); pivot_p = frame.pivot(index="date", columns="station_norm", values="q_candidate_m3_s").reindex_like(pivot_o); names = pivot_o.columns
    station_meta = frame.drop_duplicates("station_norm").set_index("station_norm").reindex(names).reset_index(); station_meta["terminal_tree"] = station_meta.heldout_terminal_tree
    per = station_metrics(model, pivot_o.to_numpy(float), pivot_p.to_numpy(float), station_meta); valid = np.isfinite(pivot_o) & np.isfinite(pivot_p); o, p = pivot_o.to_numpy()[valid], pivot_p.to_numpy()[valid]
    return {"model_id": model, "pooled_NSE": nse(o,p), "station_median_NSE": float(per.NSE.median()), "station_mean_NSE": float(per.NSE.mean()), "pooled_log_RMSE": float(np.sqrt(np.mean((np.log1p(p)-np.log1p(o))**2))), "station_median_absolute_PBIAS_pct": float(per.PBIAS_pct.abs().median())}, per


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    tasks = [(model, tree) for model in MODELS for tree in TREES]; results = []; pending = []
    for model, tree in tasks:
        prediction_path = OUT / "fold_predictions" / f"{model.lower()}_tree_{tree}.parquet"
        lock_path = OUT / "fold_locks" / f"{model.lower()}_tree_{tree}.json"
        if prediction_path.is_file() and lock_path.is_file():
            existing = pd.read_parquet(prediction_path)
            if len(existing) > 0:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                results.append({"model_id":model,"heldout_terminal_tree":tree,"stations":int(existing.station_norm.nunique()),"global_nfev":int(lock["global_optimizer_nfev"]),"mapping_nfev":int(lock["mapping_optimizer_nfev"]),"mapping_success":bool(lock["mapping_optimizer_success"]),"data_objective":float(lock["data_objective"]),"mass_error":float(lock["mass_error"]),"parameter_boundary_fraction":float(lock["parameter_boundary_fraction"]),"prediction_path":str(prediction_path)})
                continue
        pending.append((model, tree))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fit_fold, task): task for task in pending}
        for future in as_completed(futures):
            result = future.result(); results.append(result); print(json.dumps({"completed": futures[future], "data_objective": result["data_objective"]}, ensure_ascii=False), flush=True)
    audit = pd.DataFrame(results).sort_values(["model_id", "heldout_terminal_tree"]); audit.to_parquet(OUT / "nested_fold_audit.parquet", index=False)
    candidate_frames = {model: pd.concat([pd.read_parquet(path) for path in audit.loc[audit.model_id == model, "prediction_path"]], ignore_index=True) for model in MODELS}
    parent = pd.read_parquet(PARENT_OOF, columns=["date", "station_norm", "reach_id", "heldout_terminal_tree", "q_observed_m3_s", "q_fold_global_m3_s", "global_q0_m3_s", "global_q1_m3_s", "global_q2_m3_s"]); parent.date = pd.to_datetime(parent.date); parent = parent.rename(columns={"q_fold_global_m3_s":"q_candidate_m3_s", "global_q0_m3_s":"fast_response_m3_s", "global_q1_m3_s":"intermediate_response_m3_s", "global_q2_m3_s":"slow_response_m3_s"}); candidate_frames["HBV3_FOLD_PARENT"] = parent
    combined = pd.concat([frame.assign(model_id=model) for model, frame in candidate_frames.items()], ignore_index=True); combined.to_parquet(OUT / "complete_tree_nested_predictions.parquet", index=False)
    performance_rows = []; station_rows = []
    for model, frame in candidate_frames.items():
        perf, per = metrics(model, frame); performance_rows.append(perf); per["model_id"] = model; station_rows.append(per)
    performance = pd.DataFrame(performance_rows); station = pd.concat(station_rows, ignore_index=True); performance.to_parquet(OUT / "nested_spatial_performance.parquet", index=False); station.to_parquet(OUT / "nested_spatial_station_performance.parquet", index=False)
    parent_station = station.loc[station.model_id == "HBV3_FOLD_PARENT", ["station_norm", "terminal_tree", "log_RMSE"]].rename(columns={"log_RMSE":"parent_log_RMSE"}); comparison_rows = []; rng = np.random.default_rng(260826)
    gates = {}
    for model in MODELS:
        cand = station.loc[station.model_id == model, ["station_norm", "terminal_tree", "log_RMSE"]].rename(columns={"log_RMSE":"candidate_log_RMSE"}); paired = cand.merge(parent_station, on=["station_norm", "terminal_tree"], validate="one_to_one"); paired["delta_log_RMSE"] = paired.candidate_log_RMSE - paired.parent_log_RMSE; paired["model_id"] = model; comparison_rows.append(paired)
        tree_delta = paired.groupby("terminal_tree").delta_log_RMSE.mean().reindex(TREES).to_numpy(); boot = np.asarray([np.mean(rng.choice(tree_delta, size=len(tree_delta), replace=True)) for _ in range(10000)]); upper = float(np.quantile(boot,.975)); point = float(tree_delta.mean()); gates[model] = {"point_tree_mean_delta_log_RMSE": point, "tree_block_ci95_lower": float(np.quantile(boot,.025)), "tree_block_ci95_upper": upper, "noninferior_margin": .01, "spatially_noninferior": upper < .01}
    paired_all = pd.concat(comparison_rows, ignore_index=True); paired_all.to_parquet(OUT / "paired_spatial_log_rmse.parquet", index=False)
    selected = [model for model in MODELS if gates[model]["spatially_noninferior"] and bool((audit.loc[audit.model_id == model, "parameter_boundary_fraction"] <= .10).all())]
    decision = {"stage":"20260826_8", "status":"PASS_NESTED_SPATIAL_EVALUATION_COMPLETED", "spatial_gates":gates, "models_passing_spatial_gate":selected, "target_tree_discharge_used_in_fit":False, "target_tree_q_signatures_used_in_fit":False, "retrospective_discharge_used":False, "TN_used":False, "authorized_successor":"20260826_9"}; write_json(REPORT / "stage8_decision.json", decision)
    (REPORT / "technical_report.md").write_text("# 20260826_8 完整河树零历史空间评价\n\n状态：`%s`。每个目标河树的全部流量在全局参数和空间映射拟合前均被删除；PML AET作为无测站也可获得的独立遥感软约束保留。\n\n通过空间非劣门的结构：`%s`。\n\n%s\n\n空间门：\n\n```json\n%s\n```\n" % (decision["status"], selected, performance.to_markdown(index=False), json.dumps(gates, ensure_ascii=False, indent=2)), encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p):sha256(p) for p in [RUN/"experiment_contract.json", OUT/"nested_fold_audit.parquet", OUT/"complete_tree_nested_predictions.parquet", OUT/"nested_spatial_performance.parquet", REPORT/"stage8_decision.json", REPORT/"technical_report.md"]}); print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
