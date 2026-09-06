"""Register the differentiable, conserving hydrology program before fitting.

This stage is deliberately read-only with respect to hydrologic observations and
model fitting.  It freezes the parent, inputs, claims, evaluation splits, gates,
allowed candidates and stopping rules for stages 20260826_14--22.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_13"
REPORTS = RUN / "reports"

PARENT_RUN = ROOT / "5_Test" / "20260825_7"
PRIOR_PROGRAM = ROOT / "5_Test" / "20260826_10"

INPUTS = {
    "daily_forcing": ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet",
    "development_discharge": ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet",
    "retrospective_discharge": ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet",
    "gauge_audit": ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet",
    "topology": ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv",
    "static_attributes": ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet",
    "bankfull_geometry": ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet",
    "q72_reach_area_bridge": ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet",
    "soil_hydroclimate_attributes": ROOT / "5_Test" / "20260825_5" / "outputs" / "mpr_attributes_by_reach.parquet",
    "dem_recomputed_attributes": ROOT / "5_Test" / "20260826_2" / "outputs" / "reach_static_attributes_dem_recomputed.parquet",
    "parent_core": ROOT / "5_Test" / "20260825_3" / "scripts" / "hydrology_core.py",
    "parent_regional_core": ROOT / "5_Test" / "20260825_5" / "scripts" / "regional_hbv_core.py",
    "parent_parameter_lock": PARENT_RUN / "reports" / "full_development_parameter_lock.json",
    "parent_daily_bridge": PARENT_RUN / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet",
    "parent_monthly_bridge": PARENT_RUN / "outputs" / "tn_hydrology_bridge_monthly_2006_2022.parquet",
    "prior_program_decision": PRIOR_PROGRAM / "reports" / "final_program_decision.json",
}

TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_registry() -> dict[str, object]:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing registered inputs: {missing}")
    rows = []
    for role, path in INPUTS.items():
        stat = path.stat()
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )
    return {"created_at": now(), "algorithm": "sha256", "files": rows}


def literature_registry() -> dict[str, object]:
    # DOI metadata and design-level claims were checked in the preceding
    # multi-source literature audit.  Claims below are intentionally narrower
    # than the papers' full conclusions.
    entries = [
        {
            "key": "Tsai_2021_dPL",
            "doi": "10.1038/s41467-021-26107-z",
            "url": "https://doi.org/10.1038/s41467-021-26107-z",
            "design_use": "Supports learning spatial parameter mappings jointly across many basins while retaining process equations.",
        },
        {
            "key": "Feng_2023_regional_differentiable_models",
            "doi": "10.5194/hess-27-2357-2023",
            "url": "https://doi.org/10.5194/hess-27-2357-2023",
            "design_use": "Supports differentiable regional process models and strict evaluation in ungauged basins.",
        },
        {
            "key": "Song_2025_multiscale_differentiable_hydrology",
            "doi": "10.1029/2024WR038928",
            "url": "https://doi.org/10.1029/2024WR038928",
            "design_use": "Supports representing local and upstream multiscale attributes in differentiable hydrologic parameter learning.",
        },
        {
            "key": "Acuna_Espinoza_2024_dynamic_parameterization",
            "doi": "10.5194/hess-28-2705-2024",
            "url": "https://doi.org/10.5194/hess-28-2705-2024",
            "design_use": "Motivates treating time-varying neural parameterization as a conditional diagnostic because interpretability and identifiability can weaken.",
        },
        {
            "key": "Frame_2023_mass_conservation_forcing_error",
            "doi": "10.1002/hyp.14847",
            "url": "https://doi.org/10.1002/hyp.14847",
            "design_use": "Motivates preserving explicit conservation while auditing forcing error rather than silently absorbing it with unrestricted residual correction.",
        },
        {
            "key": "Dembele_2020_multisatellite_constraints",
            "doi": "10.1029/2019WR026085",
            "url": "https://doi.org/10.1029/2019WR026085",
            "design_use": "Supports using multiple independent remotely sensed hydrologic states or fluxes to constrain internal model behavior.",
        },
        {
            "key": "Kratzert_2024_large_sample_neural_hydrology",
            "doi": "10.5194/hess-28-4187-2024",
            "url": "https://doi.org/10.5194/hess-28-4187-2024",
            "design_use": "Supports large-sample regional neural hydrology rather than separately fitting a high-capacity network to each basin.",
        },
        {
            "key": "Hoedt_2021_MC_LSTM",
            "doi": "10.48550/arXiv.2101.05186",
            "url": "https://doi.org/10.48550/arXiv.2101.05186",
            "design_use": "Provides a mass-conserving neural architecture precedent; in this program it is a bounded diagnostic, not the first-line replacement for HBV.",
        },
    ]
    return {
        "created_at": now(),
        "workflow": "nature-academic-search/multi-source-search",
        "evidence_scope": "design justification only; no paper is treated as proof that PRB fast/slow components are identified",
        "entries": entries,
    }


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    previous = json.loads(INPUTS["prior_program_decision"].read_text(encoding="utf-8"))
    parent_decision = json.loads((PARENT_RUN / "reports" / "stage7_decision.json").read_text(encoding="utf-8"))
    parent_validation = json.loads((PARENT_RUN / "reports" / "final_validation.json").read_text(encoding="utf-8"))
    parent_performance = pd.read_parquet(PARENT_RUN / "outputs" / "locked_retrospective_performance_summary.parquet")

    if previous["status"] != "PROGRAM_COMPLETE_PRIMARY_HBV_TOTAL_FLOW_RESPONSE_ENSEMBLE_DIAGNOSTIC":
        raise RuntimeError("Prior program is not in its expected sealed state")
    if parent_decision["total_flow_status"] != "GLOBAL_HBV_R0_TOTAL_FLOW_PROMOTED_FOR_TN_HYDROLOGY":
        raise RuntimeError("Authoritative parent status changed")
    if previous["component_identifiability_status"] != "TOTAL_FLOW_SUPPORTED_FAST_INTERMEDIATE_SLOW_NOT_IDENTIFIED":
        raise RuntimeError("Parent component claim boundary changed")
    if not parent_validation["validation_pass"]:
        raise RuntimeError("Frozen parent validation no longer passes")

    hashes = file_registry()
    write_json(REPORTS / "input_hash_registry.json", hashes)
    write_json(REPORTS / "literature_evidence_registry.json", literature_registry())

    gates = {
        "primary_metric": "paired tree-block delta log-RMSE",
        "noninferiority": "CI95 upper < 0.01",
        "improvement": "CI95 upper < 0",
        "station_median_NSE_max_drop": 0.02,
        "median_absolute_PBIAS_max_worsening_percentage_points": 2.0,
        "flow_regimes": "high-flow and low-flow must each be noninferior",
        "seed_consistency": "at least 2 of 3 registered seeds agree in direction",
        "promotion": "all temporal and full-tree gates pass, and temporal or spatial CI95 upper < 0",
    }
    manifest = {
        "program": "20260826 differentiable conserving hydrology with independent state constraints",
        "registered_at": now(),
        "status": "running",
        "authoritative_parent": "20260825_7 GLOBAL_HBV_R0",
        "parent_TN_authorized_field": "routed_total_m3_s",
        "parent_component_status": "TOTAL_FLOW_SUPPORTED_FAST_INTERMEDIATE_SLOW_NOT_IDENTIFIED",
        "folder_boundary": "20260826_13_to_22",
        "prior_program_repair_slots": "20260826_11_to_12 remain reserved for integrity repairs to 20260826_1_to_10",
        "stage_status": {
            "20260826_13": "PASS_PROGRAM_REGISTERED",
            "20260826_14": "authorized_next_differentiable_parent_reproduction",
            "20260826_15": "registered_multiscale_attribute_construction",
            "20260826_16": "registered_Q_only_static_DPL_HBV",
            "20260826_17": "conditional_full_tree_spatial_evaluation",
            "20260826_18": "registered_independent_state_data_QA",
            "20260826_19": "conditional_state_constrained_DPL_and_neural_flux_gate",
            "20260826_20": "conditional_forcing_sensitivity",
            "20260826_21": "conditional_final_refit_retrospective_and_TN_bridge",
            "20260826_22": "integrity_repair_only",
        },
        "hard_stop": "No automatic creation of 20260826_23 or later folders",
        "authorized_successor": "20260826_14",
    }
    write_json(RUN / "program_manifest.json", manifest)

    contract = {
        "stage": "20260826_13",
        "status": "registered_before_candidate_training",
        "hypothesis": "A low-capacity differentiable mapping from multiscale static catchment attributes to bounded HBV parameters can improve transferable total-flow prediction while retaining exact HBV mass conservation; independent state products may subsequently constrain internal response states.",
        "parent": "20260825_7 GLOBAL_HBV_R0",
        "registered_candidates": {
            "GLOBAL_HBV_PARENT": "frozen eight global raw parameters",
            "DPL_HBV_LOCAL_STATIC": "seven local static attributes -> 16 hidden units -> eight bounded raw-parameter offsets",
            "DPL_HBV_MULTISCALE_STATIC": "local attributes + upstream mean/std + area (22 inputs) -> 16 hidden units -> eight bounded raw-parameter offsets",
            "DPL_HBV_MULTISCALE_STATE": "only after independent-state QA and static DPL gate",
            "MC_FLUX_GATE_DIAGNOSTIC": "only bounded mass-conserving infiltration and fast/slow transfer gates after static DPL evaluation",
        },
        "parameter_mapping_contract": {
            "output": "eight Reach-specific raw HBV parameters",
            "final_layer_initialization": "all zeros so epoch 0 exactly reproduces the frozen parent",
            "raw_offset_bound": "parent raw value +/- 1.5 for every Reach and parameter",
            "first_round_time_dependence": False,
            "first_round_station_identity": False,
        },
        "time_contract": {
            "periodic_spinup": "2006-2009",
            "optimization": "2010-2015",
            "early_stopping": "2016",
            "locked_development_temporal_evaluation": "2017-2018",
            "locked_retrospective": "2019-2022 after all development decisions and full-development parameter lock",
        },
        "spatial_contract": {
            "terminal_trees": [1, 20, 22, 26, 56, 166, 212, 217],
            "target_tree_Q_2010_2018": "completely excluded from mapping-network optimization, early stopping and model selection",
            "target_station_history": "zero target-tree discharge history may enter prediction",
            "loss_weighting": "equal terminal-tree weight, then equal station weight within tree",
        },
        "gates": gates,
        "independent_state_contract": {
            "qa_candidates": ["ESA CCI soil moisture", "SMAP soil moisture", "GRACE terrestrial water storage", "PML evapotranspiration"],
            "minimum_support_for_H2": "ESA CCI, GRACE and PML: at least two improve; SMAP must not clearly worsen",
            "maximum_claim": "H2 components state-consistent sensitivity",
            "tracer_required_for_H3": True,
        },
        "authorization_levels": {
            "H0": "total flow unsupported",
            "H1": "total flow supported; response components internal only",
            "H2": "response components compatible with independent states and usable for structured sensitivity",
            "H3": "components independently identified by tracer or conductivity evidence",
        },
        "forbidden": [
            "TN observations in hydrologic fitting, early stopping or model selection",
            "Gauge/P2 station-history correction in the 230-Reach product",
            "target-tree or target-station discharge history in spatial evaluation",
            "station identity, station random intercepts or terminal-tree random intercepts",
            "unrestricted residual LSTM or neural total-flow correction as the TN interface",
            "daily neural rewriting of all HBV parameters in the first candidate round",
            "reservoir or dam-operation equations",
            "Andreadis reference discharge or upstream area; only width and depth may be used",
            "unverified nationwide 1-km groundwater-level raster",
            "temperature in this hydrology program",
            "claims of observed surface water, groundwater, new water, old water or water age without H3 evidence",
            "automatic folders beyond 20260826_22",
        ],
        "software_preflight": "Use a dedicated PyTorch environment; do not bypass OpenMP conflicts with unsafe duplicate-library environment flags.",
        "authorized_successor": "20260826_14",
    }
    write_json(RUN / "experiment_contract.json", contract)

    failure_audit = {
        "created_at": now(),
        "prior_program": "20260826_1_to_10",
        "conclusion": "The prior program did not falsify differentiable hydrology. It falsified promotion of the tested low-capacity parameter-spatialization and alternative structures under the registered complete-tree spatial gate.",
        "observed_failures": [
            "20260825_5 assigned only one preselected attribute to each HBV parameter.",
            "20260826_6 compressed seven attributes to two indices with fixed parameter loadings.",
            "20260826_6 Powell optimization had a limited evaluation budget.",
            "20260826_8 complete-tree global search used popsize=3 and maxiter=3, too small to establish a broad structural optimum.",
            "Discharge-only fitting did not independently identify fast/intermediate/slow response components.",
        ],
        "retained_evidence": {
            "parent_total_flow": "supported",
            "parent_daily_2019_2022_pooled_NSE": 0.851044,
            "parent_daily_2019_2022_station_median_NSE": 0.617396,
            "parent_monthly_2019_2022_pooled_NSE": 0.938,
            "parent_monthly_2019_2022_station_median_NSE": 0.790,
            "component_status": previous["component_identifiability_status"],
        },
        "corrective_design": "Learn a joint low-capacity multiscale static attribute-to-parameter map through the conserving HBV equations, then test independent states conditionally.",
    }
    write_json(REPORTS / "prior_failure_audit.json", failure_audit)

    selected = parent_performance.loc[
        parent_performance.candidate.eq("GLOBAL_HBV_R0")
        & parent_performance.scope.isin(["pooled", "station_median"]),
        ["temporal_scale", "scope", "NSE", "KGE", "PBIAS_pct", "absolute_PBIAS_pct", "log_RMSE", "stations"],
    ].copy()
    selected.to_parquet(RUN / "outputs" / "frozen_parent_performance.parquet", index=False)

    report = f"""# 20260826_13 可微守恒水文程序注册与失败审计

## 结论

状态：`PASS_PROGRAM_REGISTERED`。新程序从`20260826_13`开始，`20260826_1–10`保持封存，`_11/_12`仍只属于旧程序完整性修复。不得自动创建`_23+`。

权威父模型保持`20260825_7 GLOBAL_HBV_R0`。它的总流量可以继续作为TN水量底座，但快/中/慢响应仍是模型内部状态，不能称作真实地表水、地下水、新水、老水或水龄。

## 上一轮究竟失败在哪里

上一轮否定的是已测试的低容量空间化，不是否定可微水文：一对一属性斜率和两个固定综合指数不足以表达珠江多尺度空间异质性，完整河树重训的搜索预算也偏小。新的首要候选因此固定为“多尺度静态属性→低容量参数网络→守恒HBV→河网汇流”，而不是自由LSTM改流量。

## 冻结时间与空间合同

- 2006–2009：周期spin-up。
- 2010–2015：优化。
- 2016：early stopping。
- 2017–2018：锁定development时间评价。
- 2019–2022：所有结构与参数锁定后的回顾检查。
- 完整河树评价固定删除目标树2010–2018全部流量；目标树历史不得进入任何训练、停止或选择。
- 河树固定为`1, 20, 22, 26, 56, 166, 212, 217`，损失先河树等权，再树内站点等权。

## 下一步授权

只授权`20260826_14`：在独立PyTorch环境实现可微HBV并精确复现父模型。必须先通过epoch-0父模型等价、逐日质量闭合、周期spin-up、restart和梯度检查，才允许进入静态参数网络训练。

## 文献边界

文献支持可微参数学习、多尺度属性和独立状态约束作为方法方向；它们不证明珠江快慢路径已经被识别。该边界已写入`literature_evidence_registry.json`。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_13\n\nProgram registry and failure audit for the differentiable, conserving hydrology stages 13–22. No candidate was fitted in this folder.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
