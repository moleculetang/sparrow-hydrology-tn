from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_34"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
CACHE = OUT / "tree_fit_cache"
STAGE31_SCRIPT = TEST / "20260823_31" / "scripts"
STAGE31_CONTRACT = TEST / "20260823_31" / "experiment_contract.json"
GROUPS = TEST / "20260823_17" / "outputs" / "station_spatial_groups.parquet"
PRED30 = TEST / "20260823_30" / "outputs" / "station_predictions.parquet"
PATH30 = TEST / "20260823_30" / "outputs" / "path_partition_metrics.parquet"
SPATIAL31 = TEST / "20260823_31" / "outputs" / "zero_target_history_flow_metrics.parquet"
SPATIAL_PATH31 = TEST / "20260823_31" / "outputs" / "zero_target_history_path_metrics.parquet"
EXTERNAL_PRED = TEST / "20260823_26" / "outputs" / "four_station_locked_retrospective_predictions.parquet"
EXTERNAL_METRICS = TEST / "20260823_26" / "outputs" / "four_station_locked_retrospective_metrics.parquet"
AUDIT27 = TEST / "20260823_27" / "reports" / "stage27_bridge_audit.json"
DECISION31 = TEST / "20260823_31" / "reports" / "stage31_decision.json"
DECISION32 = TEST / "20260823_32" / "reports" / "stage32_decision.json"
DECISION33 = TEST / "20260823_33" / "reports" / "stage33_decision.json"
sys.path.insert(0, str(STAGE31_SCRIPT))
from run_stage31_attribute_physical import (  # noqa: E402
    DEVICE,
    JointProblem,
    fit,
    load_attributes,
    parameter_field,
    strict_path_summary,
    strict_target_predictions,
    strict_target_proxy,
    tensor,
)
from run_stage30_global_joint_map import metrics  # noqa: E402


torch.set_default_dtype(torch.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def station_summary(frame: pd.DataFrame, prediction: str, period: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    for station, group in frame.groupby("station_norm", sort=False):
        rows.append({"period": period, "station_norm": station, **metrics(group.q_m3s, group[prediction])})
    station = pd.DataFrame(rows)
    pooled = metrics(frame.q_m3s, frame[prediction])
    pooled.update({
        "period": period,
        "station_count": len(station),
        "station_mean_NSE": float(station.NSE.mean()),
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    })
    return pooled, station


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    contract31 = json.loads(STAGE31_CONTRACT.read_text(encoding="utf-8"))
    problem = JointProblem(json.loads((TEST / "20260823_30" / "experiment_contract.json").read_text(encoding="utf-8")))
    attributes, attribute_names, attribute_frame = load_attributes(problem)
    groups = pd.read_parquet(GROUPS)
    groups["q_site"] = groups.q_site.astype(str)
    station_tree = groups.set_index("q_site").terminal_reach.astype(int).to_dict()
    trees = sorted(set(station_tree.values()))
    tree_prediction_parts, tree_proxy_parts, fit_rows = [], [], []
    for tree in trees:
        target = [station for station in problem.stations if int(station_tree[station]) == tree]
        mask = tensor([station not in target for station in problem.stations], torch.bool)
        npz_path = CACHE / f"tree_{tree}_R0_GLOBAL_REFIT.npz"
        json_path = CACHE / f"tree_{tree}_R0_GLOBAL_REFIT.json"
        if npz_path.exists() and json_path.exists():
            saved = np.load(npz_path)
            base = tensor(saved["base"])
            audit = json.loads(json_path.read_text(encoding="utf-8"))
            print(f"recovered tree {tree}", flush=True)
        else:
            base, gamma, audit = fit(problem, attributes, mask, "R0_GLOBAL_REFIT", contract31)
            audit.update({"terminal_tree": int(tree), "training_station_count": int(mask.sum().cpu()), "target_station_count": len(target)})
            np.savez_compressed(npz_path, base=base.cpu().numpy())
            json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        fit_rows.append(audit)
        parameters = parameter_field(base, None, attributes)
        simulation = problem.forward(parameters)
        tree_prediction_parts.append(strict_target_predictions(problem, simulation, target, "TREE_ZERO_HISTORY_R0", int(tree)))
        tree_proxy_parts.append(strict_target_proxy(problem, simulation, target, "TREE_ZERO_HISTORY_R0", int(tree)))
        print(f"terminal tree {tree} complete", flush=True)
    tree_predictions = pd.concat(tree_prediction_parts, ignore_index=True)
    tree_proxy = pd.concat(tree_proxy_parts, ignore_index=True)
    tree_predictions.to_parquet(OUT / "terminal_tree_zero_history_predictions.parquet", index=False)
    tree_proxy.to_parquet(OUT / "terminal_tree_zero_history_path_predictions.parquet", index=False)
    pd.DataFrame(fit_rows).to_parquet(OUT / "terminal_tree_refit_parameters.parquet", index=False)
    tree_flow_rows, tree_station_parts = [], []
    for period in ["development_2006_2018", "check_2019_2022"]:
        part = tree_predictions[tree_predictions.period.eq(period)]
        pooled, station = station_summary(part, "zero_target_history_prediction_m3_s", period)
        tree_flow_rows.append(pooled)
        tree_station_parts.append(station)
    tree_flow = pd.DataFrame(tree_flow_rows)
    tree_station = pd.concat(tree_station_parts, ignore_index=True)
    tree_flow.to_parquet(OUT / "terminal_tree_zero_history_flow_metrics.parquet", index=False)
    tree_station.to_parquet(OUT / "terminal_tree_zero_history_station_metrics.parquet", index=False)
    tree_path, tree_path_station = strict_path_summary(tree_proxy.assign(candidate="TREE_ZERO_HISTORY_R0"), "TREE_ZERO_HISTORY_R0")
    pd.DataFrame([tree_path]).to_parquet(OUT / "terminal_tree_zero_history_path_metrics.parquet", index=False)
    tree_path_station.to_parquet(OUT / "terminal_tree_zero_history_station_path_metrics.parquet", index=False)
    # Year-resolved locked-parent audit. These are internal diagnostics, not refitted temporal OOF folds.
    parent = pd.read_parquet(PRED30)
    parent = parent[parent.candidate.eq("H0_LOCKED_PARENT")].copy()
    year_rows = []
    for year, part in parent[parent.year.le(2018)].groupby("year", sort=True):
        row, _ = station_summary(part, "latent_support_m3_s", f"internal_year_{int(year)}")
        row["year"] = int(year)
        row["evaluation_role"] = "INTERNAL_YEAR_RESOLVED_DIAGNOSTIC_NOT_REFIT_OOF"
        year_rows.append(row)
    internal = pd.DataFrame(year_rows)
    internal.to_parquet(OUT / "accepted_parent_internal_year_metrics.parquet", index=False)
    check = parent[parent.year.ge(2019)]
    no_update, no_update_station = station_summary(check, "latent_support_m3_s", "frozen_no_update_2019_2022")
    pd.DataFrame([no_update]).to_parquet(OUT / "accepted_parent_frozen_check_metrics.parquet", index=False)
    no_update_station.to_parquet(OUT / "accepted_parent_frozen_check_station_metrics.parquet", index=False)
    # Preserve the previously locked four-station retrospective exactly; no model change or rereading for selection.
    external_predictions = pd.read_parquet(EXTERNAL_PRED)
    external_metrics = pd.read_parquet(EXTERNAL_METRICS)
    external_predictions.to_parquet(OUT / "four_station_locked_retrospective_predictions.parquet", index=False)
    external_metrics.to_parquet(OUT / "four_station_locked_retrospective_metrics.parquet", index=False)
    external_rows = []
    for station, part in external_predictions.groupby("station_norm", sort=False):
        obs = part.q_m3s.to_numpy(float)
        pred = part.Q_EXTERNAL_LOCKED_cfs.to_numpy(float) / 35.31466672148859
        external_rows.append({"station_norm": station, "reach_id": int(part.reach_id.iloc[0]), **metrics(obs, pred)})
    external_station = pd.DataFrame(external_rows)
    external_station.to_parquet(OUT / "four_station_locked_retrospective_station_metrics.parquet", index=False)
    d31 = json.loads(DECISION31.read_text(encoding="utf-8"))
    d32 = json.loads(DECISION32.read_text(encoding="utf-8"))
    d33 = json.loads(DECISION33.read_text(encoding="utf-8"))
    decision = {
        "stage": "20260823_34",
        "status": "FROZEN_EVALUATION_COMPLETE_PARENT_RETAINED",
        "accepted_open_loop": "H0_LOCKED_PARENT_20260823_27",
        "attribute_regionalization_promoted": False,
        "river_network_residual_status": d32["status"],
        "state_analysis_promoted": False,
        "state_analysis_diagnostic_status": d33["status"],
        "terminal_tree_count": len(trees),
        "terminal_tree_target_history_used": False,
        "terminal_tree_target_proxy_used": False,
        "accepted_parent_frozen_2019_2022": no_update,
        "terminal_tree_zero_history_development": tree_flow.iloc[0].to_dict(),
        "terminal_tree_zero_history_path": tree_path,
        "four_station_locked_retrospective": external_metrics.iloc[0].to_dict(),
        "unresolved_limitations": [
            "delayed-fraction station spatial ranking below 0.30",
            "median delayed-fraction peak-month distance is 3 months",
            "latent support flow has substantial station-scale bias",
        ],
        "model_changes_after_evaluation": False,
        "successor": "20260823_35 final 1961-2022 TN interface lock",
    }
    (REPORT / "stage34_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    report = (
        "# 20260823_34 冻结水文评价\n\n"
        "## 结论\n\n属性区域化和状态同化均未获准升级，最终保留守恒开放环路父模型。\n\n"
        "## 2019–2022 无更新检查\n\n```text\n" + pd.DataFrame([no_update]).to_string(index=False) + "\n```\n\n"
        "## 整棵河树零历史检查\n\n```text\n" + tree_flow.to_string(index=False) + "\n```\n\n"
        "## 整棵河树路径检查\n\n```text\n" + pd.DataFrame([tree_path]).to_string(index=False) + "\n```\n\n"
        "## 四站锁定回顾\n\n```text\n" + external_station.to_string(index=False) + "\n```\n\n"
        "四站结果沿用既有锁定预测，不参与任何模型修改，名称保持 retrospective check。\n"
    )
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "parent_predictions_sha256": sha256(PRED30),
        "tree_predictions_sha256": sha256(OUT / "terminal_tree_zero_history_predictions.parquet"),
        "external_source_sha256": sha256(EXTERNAL_PRED),
        "external_copy_sha256": sha256(OUT / "four_station_locked_retrospective_predictions.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=float))
    print(tree_flow.to_string(index=False))
    print(pd.DataFrame([tree_path]).to_string(index=False))
    print(external_station.to_string(index=False))


if __name__ == "__main__":
    main()
