"""Register the bounded DYN3P-HBV successor program before candidate training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_23"
REPORTS = RUN / "reports"
OLD_CLOSE = ROOT / "5_Test" / "20260826_21"

INPUTS = {
    "daily_forcing_2006_2022": ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet",
    "daily_discharge_development_2010_2018": ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet",
    "gauge_representativeness": ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet",
    "topology": ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv",
    "bankfull_geometry": ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet",
    "reach_lines": ROOT / "5_Test" / "20260814_9" / "inputs" / "spatial" / "reaches_topology.shp",
    "static_features": ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet",
    "pml_aet": ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet",
    "grace_basin": ROOT / "5_Test" / "20260826_18" / "outputs" / "grace_parent_basin_monthly_2010_2018.parquet",
}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((OLD_CLOSE / "program_manifest.json").read_text(encoding="utf-8"))
    old_decision = json.loads((OLD_CLOSE / "reports" / "final_program_decision.json").read_text(encoding="utf-8"))
    if not old_decision["program_complete"] or old_manifest["authorized_successor"] is not None:
        raise RuntimeError("The predecessor program is not cleanly closed")

    for label, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing registered input {label}: {path}")

    full_lock = json.loads(
        (ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json").read_text(encoding="utf-8")
    )
    stage16_contract = json.loads(
        (ROOT / "5_Test" / "20260826_16" / "experiment_contract.json").read_text(encoding="utf-8")
    )
    if full_lock["fit_period"] != "2010-2018" or stage16_contract["locked_temporal_evaluation"] != "2017-2018":
        raise RuntimeError("The registered evaluation-leakage fact pattern changed")

    manifest = {
        "program": "20260826 dynamic conserving three-path hydrology for TN",
        "registered_at": now(),
        "authorization_basis": "Explicit user instruction after closure of 20260826_13-to-21; this is a new program, not an automatic successor.",
        "predecessor": "20260826_21 PROGRAM_COMPLETE_PARENT_RETAINED_STATIC_DPL_REJECTED_H1",
        "hypothesis": "A low-capacity state-dependent conserving flux gate plus calibrated component-preserving channel routing can improve total-flow prediction while producing stable operational fast/intermediate/slow response components for TN.",
        "stage_status": {
            "20260826_23": "RUNNING_PROGRAM_REGISTRATION",
            "20260826_24": "REGISTERED_CORE_IMPLEMENTATION_AND_PREFLIGHT",
            "20260826_25": "REGISTERED_TEMPORAL_EVALUATION",
            "20260826_26": "CONDITIONAL_ZERO_TARGET_HISTORY_SPATIAL_EVALUATION",
            "20260826_27": "CONDITIONAL_THREE_VS_TWO_PATH_IDENTIFIABILITY",
            "20260826_28": "CONDITIONAL_FULL_DEVELOPMENT_REFIT",
            "20260826_29": "CONDITIONAL_LOCKED_2019_2022_RETROSPECTIVE",
            "20260826_30": "REGISTERED_FINAL_TN_INTERFACE_AND_CLOSE",
            "20260826_31": "RESERVED_INTEGRITY_REPAIR_ONLY",
            "20260826_32": "RESERVED_INTEGRITY_REPAIR_ONLY",
        },
        "hard_stop": "No automatic creation of 20260826_33 or later folders",
        "authorized_successor": "20260826_24",
    }
    write_json(RUN / "program_manifest.json", manifest)

    contract = {
        "stage": "20260826_23",
        "status": "registered_before_candidate_training",
        "parent_role": "GLOBAL_HBV_R0 is an implementation oracle only; every evaluation fold independently refits its parent using permitted training observations.",
        "models": ["FOLD_PARENT", "DYN_FLUX", "HYD_ROUTE", "JOINT_DYN3P"],
        "allowed_parameters": [
            "eight bounded GLOBAL_HBV parameters refit independently within each fold",
            "one shared width-8 conserving dynamic flux gate",
            "bounded gate mixture strength lambda_gate",
            "bounded component-preserving channel routing scale lambda_channel",
        ],
        "forbidden": [
            "station identity or station-specific correction",
            "target station or target tree discharge history in spatial prediction",
            "full-development parameter lock as an evaluation-fold prior center",
            "unrestricted LSTM or residual total-Q correction",
            "reservoir equations, temperature, TN, WQD reference discharge",
            "claims of observed groundwater, surface water, new water, old water or water age",
        ],
        "time_contract": {
            "spinup": "2006-2009",
            "optimization": "2010-2015",
            "early_stopping": "2016",
            "locked_temporal_evaluation": "2017-2018",
            "full_development_refit_after_structure_lock": "2010-2018",
            "retrospective": "2019-2022 read only after structure and parameter locks",
        },
        "path_contract": "Three operational response paths are authorized only after predictive-need and stability gates; otherwise retrain two paths, otherwise total-only.",
        "tn_read": False,
        "candidate_training_performed": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    prior_audit = {
        "stage": "20260826_23",
        "status": "PASS_PRIOR_FAILURE_AND_EVALUATION_BOUNDARY_AUDIT",
        "facts": {
            "full_parent_fit_period": full_lock["fit_period"],
            "stage16_evaluation_period": stage16_contract["locked_temporal_evaluation"],
            "stage16_parameter_lock_path": str(ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"),
            "stage16_code_uses_full_lock_as_parent_and_prior_center": True,
            "stage17_fold_parent_started_from_full_development_lock": True,
        },
        "interpretation": "The previous high-flow and small-basin failure fingerprints remain useful diagnostics, but they are not a clean final rejection because the temporal comparator and candidate prior center were derived from 2010-2018, which includes the 2017-2018 evaluation period.",
        "mandatory_repair": "Refit FOLD_PARENT from allowed observations inside every temporal and held-out-tree fold, and center every candidate prior on that fold-specific parent only.",
        "confirmed_failure_fingerprint": [
            "static DPL primarily improved low flow",
            "high-flow residual variability worsened at most stations",
            "peak magnitude improved more than peak timing",
            "degradation concentrated in small and medium catchments",
            "uncalibrated bankfull channel routing R1 worsened log-RMSE and is not to be repeated",
        ],
    }
    write_json(REPORTS / "prior_failure_audit.json", prior_audit)

    literature = {
        "workflow": "nature-academic-search/multi-source-search plus public code inspection",
        "interpretation_boundary": "Design precedent only; no publication proves PRB response components are identified.",
        "entries": [
            {"key": "Feng_2023_regional_differentiable", "doi": "10.5194/hess-27-2357-2023", "use": "regional differentiable process models and ungauged evaluation"},
            {"key": "Frame_2023_mass_conservation", "doi": "10.1002/hyp.14847", "use": "strict conservation and forcing-error caution"},
            {"key": "Wang_Gupta_2024_MCP", "doi": "10.1029/2023WR036461", "use": "low-capacity mass-conserving learnable flux precedent"},
            {"key": "Bindas_2024_dMC", "doi": "10.1029/2023WR035337", "use": "differentiable hydraulic routing precedent"},
            {"key": "Rinaldo_2015_SAS", "doi": "10.1002/2015WR017273", "use": "storage-selection framework and water-quality interface rationale"},
            {"key": "Li_2024_SWAT_SAS", "doi": "10.1016/j.jhydrol.2024.131386", "use": "coupled hydrology and nitrate-legacy precedent"},
            {"key": "Mizukami_2019_high_flow", "doi": "10.5194/hess-23-2601-2019", "use": "separate high-flow calibration metrics"},
            {"key": "He_2015_partition_calibration", "doi": "10.5194/hess-19-1807-2015", "use": "hydrograph-partition diagnostic calibration"},
        ],
        "code": [
            {"repository": "neuralhydrology/neuralhydrology", "file": "neuralhydrology/modelzoo/mclstm.py", "finding": "mass input is distributed, redistributed and released under an exact conservation test"},
            {"repository": "kasProg/fusion_deltaHBV", "file": "README.md", "finding": "differentiable parameter learning retains an HBV process backbone"},
            {"repository": "taddyb/dMC", "file": "README.md", "finding": "repository is explicitly under construction, so it is evidence only and not imported"},
        ],
    }
    write_json(REPORTS / "literature_evidence_registry.json", literature)

    hashes = {
        label: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for label, path in INPUTS.items()
    }
    write_json(REPORTS / "input_hash_registry.json", hashes)

    validation = {
        "stage": "20260826_23",
        "checks": {
            "predecessor_closed": True,
            "new_explicit_authorization_recorded": True,
            "evaluation_leakage_documented": True,
            "all_registered_inputs_exist_and_hashed": len(hashes) == len(INPUTS),
            "TN_not_read": True,
            "candidate_training_not_started": True,
            "hard_stop_registered": manifest["hard_stop"].endswith("folders"),
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    if not validation["all_checks_pass"]:
        raise RuntimeError(f"Stage 23 validation failed: {validation}")

    (RUN / "README.md").write_text(
        "# 20260826_23 DYN3P-HBV program registration\n\n"
        "This stage registers the explicit user-authorized successor program before candidate training.\n",
        encoding="utf-8",
    )
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_23 程序注册与失败审计\n\n"
        "新程序已在任何候选训练前注册。上一轮静态DPL的高流、小流域和事件形状失败仍是结构诊断，"
        "但其时间评价父参数及候选先验中心来自2010–2018完整拟合，包含2017–2018评价期，因此不能作为干净的最终否定证据。\n\n"
        "本程序将每个时间或空间折内的父模型重新拟合，并只以该折父参数作为候选先验中心。正式结构限定为"
        "守恒动态通量门、分量守恒河道路由及其联合模型；禁止站点外挂校正、TN、温度、水库和WQD参考流量。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
