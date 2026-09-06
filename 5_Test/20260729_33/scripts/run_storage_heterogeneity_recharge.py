from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "storage_heterogeneity_recharge"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_32" / "reports"
    / "storage_heterogeneity_audit" / "gate.json"
)
HETEROGENEITY = (
    ROOT / "5_Test" / "20260729_32" / "outputs"
    / "reach_storage_heterogeneity.parquet"
)
CURRENT_CONTROL = (
    ROOT / "5_Test" / "20260729_30" / "outputs"
    / "reach_depth_reordered_vertical_share.parquet"
)
SUPPORT_SCRIPT = (
    ROOT / "5_Test" / "20260729_31" / "scripts"
    / "run_depth_reordered_recharge.py"
)
REFERENCE_SOURCE = (
    ROOT / "5_Test" / "20260729_28" / "scripts"
    / "run_component_ablation.py"
)
CURRENT_MODEL = (
    ROOT / "5_Test" / "20260729_26" / "outputs"
    / "distributed_storage_preferential_recharge_reach_month.parquet"
)
OBSERVED_MONTH = (
    ROOT / "5_Test" / "20260729_27" / "outputs"
    / "station_month_corrected_m3s.parquet"
)
OBSERVED_STATION = (
    ROOT / "5_Test" / "20260729_27" / "reports"
    / "tradeoff_and_unit_audit" / "station_corrected_unit_tradeoff.csv"
)
CONTRACT = RUN / "experiment_contract.md"
CFS_PER_M3S = 35.3146667215
FIXED_EXCLUSIONS = {
    "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def load_support():
    spec = importlib.util.spec_from_file_location(
        "_depth_test_support", SUPPORT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load verified support script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "RUN_ONE_CONFIRMED_STORAGE_HETEROGENEITY_MODEL"
    ):
        raise RuntimeError("Parent gate does not authorize this model")
    selected = parent["selected_discovery_attribute"]
    if selected != "storage_p90_over_mean":
        raise RuntimeError("Unexpected selected heterogeneity attribute")
    heterogeneity = pd.read_parquet(HETEROGENEITY)
    current_control = pd.read_parquet(CURRENT_CONTROL)[[
        "reach_id", "vertical_share_current"
    ]]
    mapping = current_control.merge(
        heterogeneity[["reach_id", selected]],
        on="reach_id",
        validate="one_to_one",
    )
    sorted_share = np.sort(
        mapping["vertical_share_current"].to_numpy(float)
    )
    order = np.argsort(
        mapping[selected].to_numpy(float), kind="mergesort"
    )
    reordered = np.empty(len(mapping), dtype=float)
    reordered[order] = sorted_share
    mapping["vertical_share_heterogeneity_reordered"] = reordered
    distribution_preserved = bool(np.array_equal(
        sorted_share, np.sort(reordered)
    ))
    control_path = (
        OUTPUTS / "reach_heterogeneity_reordered_vertical_share.parquet"
    )
    mapping.to_parquet(control_path, index=False)

    support = load_support()
    control = mapping.set_index("reach_id")[
        "vertical_share_heterogeneity_reordered"
    ]
    namespace, replacement_count = support.load_isolated_core(control)
    candidate, closure = namespace["run_scenario"](
        "heterogeneity_reordered", "distributed", True
    )
    candidate_path = (
        OUTPUTS / "storage_heterogeneity_recharge_reach_month.parquet"
    )
    candidate.to_parquet(candidate_path, index=False)
    current = pd.read_parquet(CURRENT_MODEL).copy()
    current["scenario"] = "current_combined"
    current["channel_outflow_m3s"] = (
        current["channel_outflow_cfs"] / CFS_PER_M3S
    )
    observed_month = pd.read_parquet(OBSERVED_MONTH)
    observed_station = pd.read_csv(OBSERVED_STATION)
    metric_rows = []
    station_rows = []
    for scenario, model in {
        "current_combined": current,
        "heterogeneity_reordered": candidate,
    }.items():
        metrics, station = namespace["evaluate_scenario"](
            scenario, model, observed_month, observed_station
        )
        station = support.add_lowflow_error(
            station, scenario, model, observed_month
        )
        metrics["median_lowflow_log_error"] = float(
            station["lowflow_log_error"].median()
        )
        metric_rows.append(metrics)
        station_rows.append(station)
    metrics = pd.DataFrame(metric_rows).set_index("scenario")
    station_long = pd.concat(station_rows, ignore_index=True)
    station_wide = station_long.pivot(
        index=["station_name", "reach_id", "observed_bfi"],
        columns="scenario",
        values=[
            "simulated_bfi", "absolute_bfi_error", "nse", "log_nse",
            "kge", "relative_bias", "lowflow_log_error",
        ],
    )
    station_wide.columns = [
        f"{metric}_{scenario}" for metric, scenario in station_wide.columns
    ]
    station_wide = station_wide.reset_index()
    for metric, better in {
        "absolute_bfi_error": "lower",
        "nse": "higher",
        "log_nse": "higher",
        "lowflow_log_error": "lower",
    }.items():
        candidate_column = f"{metric}_heterogeneity_reordered"
        current_column = f"{metric}_current_combined"
        station_wide[f"{metric}_improved"] = (
            station_wide[candidate_column] < station_wide[current_column]
            if better == "lower"
            else station_wide[candidate_column] > station_wide[current_column]
        )
    metrics.reset_index().to_csv(
        REPORT / "scenario_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_wide.to_csv(
        REPORT / "station_current_vs_heterogeneity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base = metrics.loc["current_combined"]
    new = metrics.loc["heterogeneity_reordered"]
    deltas = {
        "spatial_bfi_spearman": float(
            new["spatial_bfi_spearman"] - base["spatial_bfi_spearman"]
        ),
        "median_absolute_bfi_error": float(
            new["median_absolute_bfi_error"]
            - base["median_absolute_bfi_error"]
        ),
        "median_nse": float(new["median_nse"] - base["median_nse"]),
        "median_log_nse": float(
            new["median_log_nse"] - base["median_log_nse"]
        ),
        "median_lowflow_log_error": float(
            new["median_lowflow_log_error"]
            - base["median_lowflow_log_error"]
        ),
        "absolute_pml_aet_bias": float(
            abs(new["pml_aet_domain_relative_bias"])
            - abs(base["pml_aet_domain_relative_bias"])
        ),
    }
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "selected_attribute_frozen": selected == "storage_p90_over_mean",
        "candidate_distribution_exactly_preserved": distribution_preserved,
        "isolated_formula_replacement_exactly_once": replacement_count == 1,
        "candidate_rows_35880": len(candidate) == 35880,
        "same_97_stations": (
            len(station_wide) == 97
            and station_wide["reach_id"].nunique() == 97
        ),
        "shijiao_present": "石角站" in set(station_wide["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(station_wide["station_name"])
        ),
        "strict_system_closure": (
            closure["maximum_system_relative_closure"] < 1e-8
        ),
        "strict_component_closure": (
            closure["maximum_component_relative_closure"] < 1e-8
        ),
        "spatial_bfi_improves_at_least_0_03": (
            deltas["spatial_bfi_spearman"] >= 0.03
        ),
        "bfi_error_not_worse_over_0_02": (
            deltas["median_absolute_bfi_error"] <= 0.02
        ),
        "nse_not_worse_over_0_05": deltas["median_nse"] >= -0.05,
        "log_nse_not_worse_over_0_05": (
            deltas["median_log_nse"] >= -0.05
        ),
        "lowflow_error_not_worse_over_0_05": (
            deltas["median_lowflow_log_error"] <= 0.05
        ),
        "pml_aet_bias_not_worse_over_0_02": (
            deltas["absolute_pml_aet_bias"] <= 0.02
        ),
        "zero_parameter_calibration": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    performance_keys = [
        "spatial_bfi_improves_at_least_0_03",
        "bfi_error_not_worse_over_0_02",
        "nse_not_worse_over_0_05",
        "log_nse_not_worse_over_0_05",
        "lowflow_error_not_worse_over_0_05",
        "pml_aet_bias_not_worse_over_0_02",
    ]
    retained = all(checks[key] for key in performance_keys)
    decision = (
        "RETAIN_STORAGE_HETEROGENEITY_RECHARGE_CONTROL"
        if retained
        else "REJECT_STORAGE_HETEROGENEITY_RECHARGE_CONTROL"
    )
    next_action = (
        "RUN_TEMPORAL_AND_STATION_ROBUSTNESS_GATE"
        if retained
        else "REVIEW_OR_ACQUIRE_TEXTURE_AND_DRAINAGE_ATTRIBUTES"
    )
    improvement_fractions = {
        "bfi_error": float(
            station_wide["absolute_bfi_error_improved"].mean()
        ),
        "nse": float(station_wide["nse_improved"].mean()),
        "log_nse": float(station_wide["log_nse_improved"].mean()),
        "lowflow_error": float(
            station_wide["lowflow_log_error_improved"].mean()
        ),
    }
    gate = {
        "run_id": "20260729_33",
        "phase": "storage_heterogeneity_recharge_model_test",
        "created_utc": utc_now(),
        "checks": checks,
        "scenario_metrics": metrics.reset_index().to_dict("records"),
        "candidate_deltas": deltas,
        "station_improvement_fractions": improvement_fractions,
        "closure": closure,
        "decision": decision,
        "authorized_next_action": next_action,
        "candidate_retained": retained,
        "selected_attribute": selected,
        "parameters_calibrated": False,
        "observed_discharge_unit": "m3/s",
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = f"""# 相对高容量尾部空间重排模型测试

## 结论

`{decision}`

| 指标 | 当前组合 | 异质性重排 | 变化 |
|---|---:|---:|---:|
| BFI | {base['median_bfi']:.3f} | {new['median_bfi']:.3f} | {new['median_bfi']-base['median_bfi']:+.3f} |
| BFI绝对误差 | {base['median_absolute_bfi_error']:.3f} | {new['median_absolute_bfi_error']:.3f} | {deltas['median_absolute_bfi_error']:+.3f} |
| BFI Spearman | {base['spatial_bfi_spearman']:.3f} | {new['spatial_bfi_spearman']:.3f} | {deltas['spatial_bfi_spearman']:+.3f} |
| NSE | {base['median_nse']:.3f} | {new['median_nse']:.3f} | {deltas['median_nse']:+.3f} |
| log-NSE | {base['median_log_nse']:.3f} | {new['median_log_nse']:.3f} | {deltas['median_log_nse']:+.3f} |
| 低流对数误差 | {base['median_lowflow_log_error']:.3f} | {new['median_lowflow_log_error']:.3f} | {deltas['median_lowflow_log_error']:+.3f} |

- BFI误差改善站比例：{improvement_fractions['bfi_error']:.1%}；
- NSE改善站比例：{improvement_fractions['nse']:.1%}；
- log-NSE改善站比例：{improvement_fractions['log_nse']:.1%}；
- 低流误差改善站比例：{improvement_fractions['lowflow_error']:.1%}；
- 最大系统/分量闭合误差：
  `{closure['maximum_system_relative_closure']:.3e}` /
  `{closure['maximum_component_relative_closure']:.3e}`。

下一步：`{next_action}`
"""
    report_path = REPORT / "technical_report.md"
    report_path.write_text(report, encoding="utf-8")
    sources = [
        PARENT_GATE, HETEROGENEITY, CURRENT_CONTROL, SUPPORT_SCRIPT,
        REFERENCE_SOURCE, CURRENT_MODEL, OBSERVED_MONTH, OBSERVED_STATION,
        CONTRACT, RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "run_storage_heterogeneity_recharge.py",
        RUN / "scripts" / "validate_storage_heterogeneity_recharge.py",
    ]
    products = [
        control_path, candidate_path, REPORT / "scenario_metrics.csv",
        REPORT / "station_current_vs_heterogeneity.csv",
        gate_path, report_path,
    ]
    provenance = {
        "run_id": "20260729_33",
        "created_utc": utc_now(),
        "sources": [
            record(path, "heterogeneity_model_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "heterogeneity_model_product", "derived")
            for path in products
        ],
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (LOGS / "runtime_log.md").write_text(
        "# Runtime log\n\n"
        "`conda --no-plugins run -n sparrow python "
        "E:\\SPARROW\\5_Test\\20260729_33\\scripts\\"
        "run_storage_heterogeneity_recharge.py`\n\n"
        f"Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "current_spatial_bfi": float(base["spatial_bfi_spearman"]),
        "candidate_spatial_bfi": float(new["spatial_bfi_spearman"]),
        "spatial_bfi_delta": deltas["spatial_bfi_spearman"],
        "current_nse": float(base["median_nse"]),
        "candidate_nse": float(new["median_nse"]),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
