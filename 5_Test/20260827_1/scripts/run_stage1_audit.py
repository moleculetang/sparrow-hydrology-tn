"""Evidence-backed audit of the 20260826 DYN2P program before clean refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_1"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"

DISCHARGE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
STATIC_REGISTRY = ROOT / "5_Test" / "20260826_15" / "reports" / "feature_registry.json"
FINAL_DECISION = ROOT / "5_Test" / "20260826_30" / "reports" / "final_program_decision.json"
FINAL_CONTRACT = ROOT / "5_Test" / "20260826_30" / "reports" / "tn_hydrology_interface_contract.json"
FINAL_QA = ROOT / "5_Test" / "20260826_30" / "reports" / "tn_hydrology_bridge_qa.json"
STAGE27_CONTRACT = ROOT / "5_Test" / "20260826_27" / "experiment_contract.json"
STAGE27_DECISION = ROOT / "5_Test" / "20260826_27" / "reports" / "stage27_decision.json"
STAGE32_LOCK = ROOT / "5_Test" / "20260826_32" / "reports" / "repair_parameter_lock.json"
STAGE32_CONTRACT = ROOT / "5_Test" / "20260826_32" / "experiment_contract.json"
STAGE28_SCRIPT = ROOT / "5_Test" / "20260826_28" / "scripts" / "run_stage28.py"
STAGE30_SCRIPT = ROOT / "5_Test" / "20260826_30" / "scripts" / "run_stage30.py"
STAGE25_SCRIPT = ROOT / "5_Test" / "20260826_25" / "scripts" / "run_stage25.py"
STAGE26_SCRIPT = ROOT / "5_Test" / "20260826_26" / "scripts" / "run_stage26.py"
SPATIAL_WORKER = ROOT / "5_Test" / "20260826_27" / "scripts" / "spatial_2p_worker.py"
STAGE27_SPATIAL_METRICS = ROOT / "5_Test" / "20260826_27" / "outputs" / "spatial_station_metrics.parquet"
STAGE28_LOCK = ROOT / "5_Test" / "20260826_28" / "reports" / "full_development_parameter_lock.json"
STAGE29_SUMMARY = ROOT / "5_Test" / "20260826_29" / "outputs" / "locked_retrospective_performance_summary.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finding(code: str, severity: str, confidence: str, failed: bool, evidence: object, risk: str, repair: str) -> dict:
    return {
        "code": code,
        "severity": severity,
        "confidence": confidence,
        "failed": failed,
        "evidence": evidence,
        "risk": risk,
        "repair": repair,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    inputs = [
        DISCHARGE, GAUGES, FORCING, STATIC, STATIC_REGISTRY, FINAL_DECISION,
        FINAL_CONTRACT, FINAL_QA, STAGE27_CONTRACT, STAGE27_DECISION,
        STAGE32_LOCK, STAGE32_CONTRACT, STAGE28_SCRIPT, STAGE30_SCRIPT,
        STAGE25_SCRIPT, STAGE26_SCRIPT, SPATIAL_WORKER, STAGE27_SPATIAL_METRICS,
        STAGE28_LOCK, STAGE29_SUMMARY,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    write_json(REPORTS / "input_hash_registry.json", {
        "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs]
    })

    discharge = pd.read_parquet(DISCHARGE)
    discharge["date"] = pd.to_datetime(discharge.date)
    gauges = pd.read_parquet(GAUGES)
    eligible = gauges.loc[gauges.topology_representative & gauges.four_group_check_eligible].drop_duplicates("station_norm")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    static = pd.read_parquet(STATIC)

    counts = discharge.loc[discharge.date.dt.year.between(2010, 2018)].groupby("station_norm").q_m3_s.count()
    final_stations = eligible.loc[eligible.station_norm.map(counts).fillna(0).ge(180)].copy()
    final_stations["valid_daily_observations_2010_2018"] = final_stations.station_norm.map(counts).fillna(0).astype(int)
    final_stations.to_parquet(OUT / "final_refit_station_registry.parquet", index=False)

    coverage = discharge.assign(period=np.select(
        [
            discharge.date.dt.year.between(2010, 2015),
            discharge.date.dt.year.eq(2016),
            discharge.date.dt.year.between(2017, 2018),
        ],
        ["train", "stop", "eval"],
        default="other",
    )).groupby(["station_norm", "period"]).q_m3_s.count().unstack(fill_value=0)
    complete_names = coverage.index[
        coverage.get("train", 0).ge(2191)
        & coverage.get("stop", 0).ge(366)
        & coverage.get("eval", 0).ge(730)
    ]
    temporal_stations = eligible.loc[eligible.station_norm.isin(complete_names)]

    profile_rows = [
        {"dataset": "development_discharge", "metric": "rows", "value": len(discharge)},
        {"dataset": "development_discharge", "metric": "stations", "value": discharge.station_norm.nunique()},
        {"dataset": "development_discharge", "metric": "duplicate_station_date", "value": discharge.duplicated(["station_norm", "date"]).sum()},
        {"dataset": "development_discharge", "metric": "null_q", "value": discharge.q_m3_s.isna().sum()},
        {"dataset": "development_discharge", "metric": "negative_q", "value": (discharge.q_m3_s < 0).sum()},
        {"dataset": "eligible_gauges", "metric": "stations", "value": eligible.station_norm.nunique()},
        {"dataset": "eligible_gauges", "metric": "reaches", "value": eligible.reach_id.nunique()},
        {"dataset": "eligible_gauges", "metric": "duplicate_station", "value": eligible.duplicated("station_norm").sum()},
        {"dataset": "eligible_gauges", "metric": "duplicate_reach", "value": eligible.duplicated("reach_id").sum()},
        {"dataset": "temporal_complete_cohort", "metric": "stations", "value": len(temporal_stations)},
        {"dataset": "full_refit_cohort", "metric": "stations", "value": len(final_stations)},
        {"dataset": "full_refit_cohort", "metric": "stations_below_3000_valid_days", "value": (final_stations.valid_daily_observations_2010_2018 < 3000).sum()},
        {"dataset": "forcing", "metric": "rows", "value": len(forcing)},
        {"dataset": "forcing", "metric": "duplicate_reach_date", "value": forcing.duplicated(["reach_id", "date"]).sum()},
        {"dataset": "forcing", "metric": "reaches", "value": forcing.reach_id.nunique()},
        {"dataset": "forcing", "metric": "null_cells", "value": forcing.isna().sum().sum()},
        {"dataset": "static_features", "metric": "rows", "value": len(static)},
        {"dataset": "static_features", "metric": "duplicate_reach", "value": static.duplicated("reach_id").sum()},
        {"dataset": "static_features", "metric": "null_cells", "value": static.isna().sum().sum()},
    ]
    profile = pd.DataFrame(profile_rows)
    profile["value"] = profile.value.astype(float)
    profile.to_parquet(OUT / "data_quality_profile.parquet", index=False)

    stage27 = read_json(STAGE27_CONTRACT)
    stage27_decision = read_json(STAGE27_DECISION)
    stage32_lock = read_json(STAGE32_LOCK)
    stage32_contract = read_json(STAGE32_CONTRACT)
    static_registry = read_json(STATIC_REGISTRY)
    final_qa = read_json(FINAL_QA)
    final_contract = read_json(FINAL_CONTRACT)
    stage28_source = STAGE28_SCRIPT.read_text(encoding="utf-8")
    stage30_source = STAGE30_SCRIPT.read_text(encoding="utf-8")
    stage25_source = STAGE25_SCRIPT.read_text(encoding="utf-8")
    stage26_source = STAGE26_SCRIPT.read_text(encoding="utf-8")
    spatial_worker_source = SPATIAL_WORKER.read_text(encoding="utf-8")
    stage28_lock = read_json(STAGE28_LOCK)
    spatial_metrics = pd.read_parquet(STAGE27_SPATIAL_METRICS)
    spatial_dyn2p = spatial_metrics.loc[spatial_metrics.model_id.eq("DYN2P")].copy()
    retrospective_summary = pd.read_parquet(STAGE29_SUMMARY)
    retrospective_monthly = retrospective_summary.loc[
        retrospective_summary.temporal_scale.eq("monthly")
        & retrospective_summary.candidate.eq("DYN2P")
    ].set_index("scope")

    findings = []
    findings.append(finding(
        "POST_RETROSPECTIVE_ALPHA_SELECTION", "high", "high", True,
        {
            "lock_status": stage32_lock.get("status"),
            "retrospective_failure_trigger_disclosed": stage32_lock.get("retrospective_failure_trigger_disclosed"),
            "independent_validation_claim": stage32_lock.get("independent_validation_claim"),
        },
        "The final engineering parameterization was decided after the 2019-2022 result was known, so that period cannot validate the promoted baseline independently.",
        "Fix DYN2P and alpha=0.5 before the 20260827 rerun; use fold-contained training and label 2019-2022 only as known-period replication.",
    ))
    findings.append(finding(
        "REUSED_SPATIAL_FOLDS_FOR_STRUCTURE_DEVELOPMENT", "high", "high", True,
        {
            "heldout_trees": stage27.get("heldout_terminal_trees"),
            "predecessor_structure_selection": "The same eight trees informed the earlier DYN_FLUX/JOINT and later DYN2P decisions.",
        },
        "Direct target discharge was excluded within each fold, but repeated structural decisions on the same eight held-out trees make the reported spatial result post-selection rather than untouched external evidence.",
        "Treat DYN2P as fixed in 20260827, rerun fold-specific parents/candidates, and add the four user-approved supplemental stations as a separate zero-training-history check.",
    ))
    findings.append(finding(
        "FINAL_ALGORITHM_NOT_SPATIALLY_REPLAYED", "high", "high", True,
        {
            "spatial_worker_seed": 260826 if "260826" in spatial_worker_source else None,
            "full_development_selected_seed": stage28_lock.get("selected_seed"),
            "final_alpha": stage32_lock.get("selected_alpha", 0.5),
        },
        "The spatial folds evaluated a different seed and unshrunk parameterization from the final deployed seed/alpha combination, so they do not validate the exact final algorithm.",
        "Replay the exact frozen training algorithm and all registered seeds without reading target discharge; evaluate only after prediction hashes are locked.",
    ))
    spatial_tree_median = spatial_dyn2p.groupby("terminal_tree").NSE.median()
    findings.append(finding(
        "SPATIAL_ABSOLUTE_SKILL_WEAK_TAIL", "high", "high", True,
        {
            "station_count": int(len(spatial_dyn2p)),
            "station_median_NSE": float(spatial_dyn2p.NSE.median()),
            "station_mean_NSE": float(spatial_dyn2p.NSE.mean()),
            "station_P10_NSE": float(spatial_dyn2p.NSE.quantile(0.10)),
            "station_P25_NSE": float(spatial_dyn2p.NSE.quantile(0.25)),
            "negative_NSE_count": int(spatial_dyn2p.NSE.lt(0).sum()),
            "negative_NSE_fraction": float(spatial_dyn2p.NSE.lt(0).mean()),
            "worst_tree_median_NSE": float(spatial_tree_median.min()),
            "worst_tree": int(spatial_tree_median.idxmin()),
        },
        "Relative noninferiority hides a weak lower tail; the model cannot yet be claimed reliable on every ungauged Reach.",
        "Add absolute spatial-skill reporting and reserve the four user-approved zero-history stations for the post-lock test only.",
    ))
    findings.append(finding(
        "FAST_SLOW_PROXY_IDENTIFICATION_WEAK", "high", "medium", True,
        {
            "development_period": "2010-2018",
            "station_count": 91,
            "model_slow_fraction_vs_three_method_median_BFI_spearman": 0.118,
            "p_value": 0.266,
            "RMSE": 0.179,
            "model_slow_fraction_median": 0.312,
            "proxy_BFI_median": 0.453,
            "interpretation": "BFI is a soft proxy, not component truth",
        },
        "The components are numerically nondegenerate but have not yet been shown to match independent fast/slow hydrologic behavior.",
        "Run pre-registered BFI, recession, event-response, memory and stability diagnostics before authorizing component use by TN.",
    ))
    findings.append(finding(
        "TARGET_TREE_REMOTE_STATE_CONSTRAINT", "medium", "high", False,
        {
            "target_discharge_used": stage27_decision.get("target_tree_history_used_in_training"),
            "PML_GRACE_role": "basin-wide remotely sensed state guardrails remain available in spatial folds",
        },
        "This is not target-discharge leakage if remote sensing is considered globally available, but the spatial claim must be named zero-target-discharge rather than zero-target-observation.",
        "Declare PML/GRACE availability explicitly and keep their time slices inside the training/stopping periods.",
    ))
    findings.append(finding(
        "FULL_REFIT_COHORT_THRESHOLD", "medium", "high", bool((final_stations.valid_daily_observations_2010_2018 < 3000).any()),
        {
            "station_count": len(final_stations),
            "below_3000_days": int((final_stations.valid_daily_observations_2010_2018 < 3000).sum()),
            "minimum_days": int(final_stations.valid_daily_observations_2010_2018.min()),
        },
        "The full refit uses a >=180-day threshold although the temporal experiment uses complete records. A few short records receive station-level weight comparable to full records.",
        "Register one cohort rule before refit; report sensitivity with and without stations below 3000 valid days.",
    ))
    findings.append(finding(
        "CENTERED_EVENT_WINDOW_BOUNDARY_LEAKAGE", "medium", "high", True,
        {
            "centered_padding_present": "padding = 3" in stage25_source or "padding=3" in stage25_source,
            "maximum_affected_station_days": 195,
            "known_high_flow_loss_rows": 3,
            "boundary": "2015 training to 2016 stopping",
        },
        "Centered seven-day event windows allow a small number of 2016 observations to enter the 2015 training objective.",
        "Use a strictly trailing seven-day window and require every contributing day to belong to the same partition.",
    ))
    findings.append(finding(
        "TRAIN_DEPLOY_SPINUP_OPERATOR_MISMATCH", "medium", "high", True,
        {
            "parent_initial_reused_during_candidate_training": "parent_initial" in stage25_source,
            "candidate_spinup_after_training": "periodic_spinup" in stage25_source,
        },
        "Training and deployment do not evaluate the candidate from the same initialized state, which makes the fitted objective differ from the deployed operator.",
        "Use candidate-consistent initialization during training and verify prediction-scale equivalence to a second registered initialization.",
    ))
    findings.append(finding(
        "EARLY_OUTPUT_PERIOD_NONCAUSAL_SPINUP", "medium", "high", True,
        {
            "period": "2006-2009",
            "mechanism": "2006-2009 forcing is cycled to equilibrium before the same years are emitted",
        },
        "The first four output years depend on later forcing within that spin-up cycle and are not strict causal simulations.",
        "Mark 2006-2009 as is_spinup_period=true and exclude them from formal performance claims.",
    ))
    findings.append(finding(
        "CACHE_AND_HASH_CONTRACT_INCOMPLETE", "medium", "high", True,
        {
            "parent_cache_station_count_only": "station_count" in stage25_source,
            "spatial_skip_on_metadata": "metadata.json" in stage26_source,
            "missing_hash_classes": ["parent parameter lock", "Q72 area/bridge inputs", "external code modules", "Shapefile sidecars"],
        },
        "A stale result can be silently reused after code or inputs change while station counts or directory names remain unchanged.",
        "Make cache keys depend on input, code, station list, time partition, seed and hyperparameter SHA-256 values; otherwise recompute.",
    ))
    findings.append(finding(
        "INSTANTANEOUS_NETWORK_ROUTING_WITH_SEPARATE_TRAVEL_TIME", "medium", "high", True,
        {
            "final_flow_route": "route_instantaneous" in stage28_source,
            "travel_time_formula_present": "channel_bankfull_travel_time" in stage30_source,
            "travel_time_role": final_contract.get("authorized_hydraulic_covariates"),
        },
        "The final daily/monthly flow is accumulated instantaneously, while channel travel time is only a later TN exposure covariate. It must not be described as calibrated hydraulic routing.",
        "In 20260827 quantify the monthly boundary effect of conservative channel storage; retain instantaneous routing only if the pre-registered effect is negligible or the routed candidate fails clean validation.",
    ))
    findings.append(finding(
        "FAST_SLOW_NOT_DIRECTLY_OBSERVED", "medium", "high", False,
        {
            "formal_paths": read_json(FINAL_DECISION).get("formal_operational_paths"),
            "component_semantics": final_contract.get("component_semantics"),
        },
        "Total discharge, AET and storage constrain the model, but no tracer or baseflow observation directly labels the two components.",
        "Keep the terms operational fast/slow response and require seed/fold stability plus exact closure; forbid groundwater/new-water/old-water claims.",
    ))
    findings.append(finding(
        "STATIC_FEATURE_FUTURE_OR_TARGET_DISCHARGE", "critical", "high", False,
        {
            "hydroclimate_definition": static_registry.get("hydroclimate_definition"),
            "discharge_read": False,
            "feature_reaches": int(static.reach_id.nunique()),
        },
        "No evidence that target discharge or 2016-2022 forcing entered the static features.",
        "Preserve the existing static-feature hash and provenance.",
    ))
    findings.append(finding(
        "CORE_DATA_GRAIN", "critical", "high",
        bool(discharge.duplicated(["station_norm", "date"]).any() or eligible.duplicated("reach_id").any() or forcing.duplicated(["reach_id", "date"]).any()),
        {
            "discharge_duplicate_keys": int(discharge.duplicated(["station_norm", "date"]).sum()),
            "eligible_duplicate_reaches": int(eligible.duplicated("reach_id").sum()),
            "forcing_duplicate_keys": int(forcing.duplicated(["reach_id", "date"]).sum()),
            "negative_discharge": int((discharge.q_m3_s < 0).sum()),
        },
        "Duplicate station-days, duplicate representative Reaches or invalid flows would distort calibration.",
        "Keep these as executable hard gates in every successor.",
    ))
    findings.append(finding(
        "FINAL_INTERFACE_NUMERICAL_INTEGRITY", "critical", "high", not bool(final_qa.get("all_checks_pass")),
        final_qa,
        "A broken component closure or incomplete Reach-time grid would make the TN interface unusable.",
        "Retain exact closure, complete-grid and hash tests in the rebuilt baseline.",
    ))

    audit = {
        "stage": "20260827_1",
        "status": "AUDIT_COMPLETE_REBUILD_REQUIRED",
        "finding_counts": {
            severity: int(sum(row["failed"] and row["severity"] == severity for row in findings))
            for severity in ["critical", "high", "medium", "low"]
        },
        "findings": findings,
        "bottom_line": {
            "direct_future_observation_in_training_tensor_found": False,
            "evaluation_informed_model_decision_found": True,
            "old_independent_validation_claim_allowed": False,
            "engineering_interface_numerically_valid": bool(final_qa.get("all_checks_pass")),
            "20260827_clean_refit_required": True,
            "total_flow_provisionally_valid": True,
            "fast_slow_components_identified": False,
        },
        "authorized_successor": "20260827_2",
        "TN_read": False,
    }
    write_json(REPORTS / "leakage_and_structure_audit.json", audit)

    validation = {
        "stage": "20260827_1",
        "checks": {
            "all_inputs_hashed": True,
            "discharge_key_unique": not discharge.duplicated(["station_norm", "date"]).any(),
            "eligible_station_unique": not eligible.duplicated("station_norm").any(),
            "eligible_reach_unique": not eligible.duplicated("reach_id").any(),
            "forcing_key_unique": not forcing.duplicated(["reach_id", "date"]).any(),
            "static_reach_unique": not static.duplicated("reach_id").any(),
            "post_retrospective_issue_recorded": any(row["code"] == "POST_RETROSPECTIVE_ALPHA_SELECTION" and row["failed"] for row in findings),
            "spatial_reuse_issue_recorded": any(row["code"] == "REUSED_SPATIAL_FOLDS_FOR_STRUCTURE_DEVELOPMENT" and row["failed"] for row in findings),
            "final_algorithm_spatial_mismatch_recorded": any(row["code"] == "FINAL_ALGORITHM_NOT_SPATIALLY_REPLAYED" and row["failed"] for row in findings),
            "event_window_leakage_recorded": any(row["code"] == "CENTERED_EVENT_WINDOW_BOUNDARY_LEAKAGE" and row["failed"] for row in findings),
            "component_identification_limit_recorded": any(row["code"] == "FAST_SLOW_PROXY_IDENTIFICATION_WEAK" and row["failed"] for row in findings),
            "TN_not_read": True,
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)

    report = f"""# 20260827_1 DYN2P独立审计

## 结论

未发现2019–2022实测流量被直接放入2010–2018训练张量；核心流量、站点和静态属性主键也通过。旧接口在数值上守恒且完整。

但旧结果不能作为无泄露的新基线，原因有两项：

1. `alpha=0.5`是在首次看到2019–2022偏差门失败后才固定，属于评价结果影响最终模型决策；
2. DYN2P结构是在同一组八个空间河树被多轮查看后形成，旧空间结果是post-selection evidence，不是未接触外部检验。

因此状态为`AUDIT_COMPLETE_REBUILD_REQUIRED`。20260827必须把DYN2P与alpha=0.5预先冻结，所有父模型、候选参数和停止轮次在各自训练折内重估；2019–2022只称已知期复现。

独立子agent还确认：旧空间折只运行seed 260826，而最终模型采用另一seed并再做alpha收缩，最终部署算法没有被原样做空间重跑；旧空间检验91站NSE中位0.559、平均0.163，16站为负，最弱河树中位为-0.233。快慢分量与三种BFI代理中位的Spearman仅0.118（BFI不是分量真值，因此这是风险信号而非否决证据）。

## 数据质量

- 开发流量：{len(discharge):,}行，{discharge.station_norm.nunique()}站，重复站日={int(discharge.duplicated(['station_norm', 'date']).sum())}，负流量={int((discharge.q_m3_s < 0).sum())}；
- 正式拓扑代表：{len(eligible)}站、{eligible.reach_id.nunique()}个唯一Reach；
- 严格完整时间队列：{len(temporal_stations)}站；完整开发期最终重拟合队列：{len(final_stations)}站，其中{int((final_stations.valid_daily_observations_2010_2018 < 3000).sum())}站少于3000个有效日；
- 静态属性为230个唯一Reach，未使用流量，水文气候态仅用2006–2015 forcing。

## 结构边界

旧最终流量采用瞬时河网累计；Andreadis停留时间只作为后续TN水力暴露协变量，并未进入已校准流量路由。20260827需量化守恒河道储存对月边界的影响后再锁定这一选择。

快、慢仍是操作性模型响应分量，没有独立示踪剂或基流观测直接标注，禁止解释成真实地表/地下、新/老水或水龄。

另有四项必须在干净重跑中修复：中心7日窗跨2015/2016边界、候选训练与部署初态不一致、2006–2009周期spin-up输出非严格因果、缓存与哈希不足。正式测试防火墙固定为：训练和230 Reach导出进程不得打开2019–2022实测流量或珠坑、昭平、瓦村（二）、盘江桥（三）的观测/映射文件；先锁模型与预测哈希，再由独立评价进程读取。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260827_1 leakage and structure audit\n\nSee `reports/technical_report.md` and `reports/leakage_and_structure_audit.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["bottom_line"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
