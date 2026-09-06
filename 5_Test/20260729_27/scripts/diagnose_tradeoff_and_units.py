from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "tradeoff_and_unit_audit"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_26" / "reports"
    / "distributed_storage_preferential_recharge" / "gate.json"
)
MONTH = (
    ROOT / "5_Test" / "20260729_26" / "outputs"
    / "station_month_candidate_vs_baseline.parquet"
)
STATIONS = (
    ROOT / "5_Test" / "20260729_26" / "reports"
    / "distributed_storage_preferential_recharge"
    / "station_candidate_vs_baseline.csv"
)
CANDIDATE_MODEL = (
    ROOT / "5_Test" / "20260729_26" / "outputs"
    / "distributed_storage_preferential_recharge_reach_month.parquet"
)
CAPACITY_STATS = (
    ROOT / "5_Test" / "20260729_26" / "inputs"
    / "capacity_zonal_stats.csv"
)
UNIT_IMAGE = (
    ROOT / "0_hydro_sediment_data" / "discharge" / "check"
    / "20260602" / "03_覆盖率或全年缺失结构问题" / "2010坪岭"
    / "ocr_main__page_001_layout_overlay.png"
)
CFS_PER_M3S = 35.3146667215
REQUIRED_PARENT_ACTION = (
    "DIAGNOSE_DISTRIBUTED_STORAGE_PREF_RECHARGE_TRADEOFF_WITHOUT_TUNING"
)


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


def scores(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    denominator = np.sum((observed - observed.mean()) ** 2)
    nse = 1.0 - np.sum((simulated - observed) ** 2) / denominator
    log_observed = np.log1p(observed)
    log_simulated = np.log1p(np.maximum(simulated, 0.0))
    log_denominator = np.sum((log_observed - log_observed.mean()) ** 2)
    log_nse = (
        1.0
        - np.sum((log_simulated - log_observed) ** 2) / log_denominator
    )
    correlation = float(np.corrcoef(observed, simulated)[0, 1])
    alpha = simulated.std(ddof=0) / observed.std(ddof=0)
    beta = simulated.mean() / observed.mean()
    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    )
    return {
        "nse": float(nse),
        "log_nse": float(log_nse),
        "kge": float(kge),
        "relative_bias": float(beta - 1.0),
        "correlation": correlation,
    }


def safe_spearman(first: pd.Series, second: pd.Series) -> float:
    valid = first.notna() & second.notna()
    return float(
        first.loc[valid].astype(float).corr(
            second.loc[valid].astype(float), method="spearman"
        )
    )


def main() -> None:
    for directory in [REPORT, OUTPUTS, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != REQUIRED_PARENT_ACTION:
        raise RuntimeError("Parent gate does not authorize tradeoff audit")
    if not UNIT_IMAGE.exists():
        raise RuntimeError("Original unit-evidence image is missing")

    month = pd.read_parquet(MONTH)
    station_prior = pd.read_csv(STATIONS)
    candidate_model = pd.read_parquet(CANDIDATE_MODEL)
    capacity = pd.read_csv(CAPACITY_STATS)
    month = month.copy()
    month["observed_q_m3s"] = month["observed_q_cfs"]
    month["baseline_q_m3s"] = (
        month["channel_outflow_cfs_baseline"] / CFS_PER_M3S
    )
    month["candidate_q_m3s"] = (
        month["channel_outflow_cfs_candidate"] / CFS_PER_M3S
    )

    station_rows: list[dict] = []
    for station_name, group in month.groupby("station_name"):
        prior = station_prior.loc[
            station_prior["station_name"].eq(station_name)
        ].iloc[0]
        observed = group["observed_q_m3s"].to_numpy(float)
        baseline = group["baseline_q_m3s"].to_numpy(float)
        candidate = group["candidate_q_m3s"].to_numpy(float)
        baseline_scores = scores(observed, baseline)
        candidate_scores = scores(observed, candidate)
        threshold = np.quantile(observed, 0.25)
        low = observed <= threshold
        baseline_low_error = float(
            np.median(
                np.abs(np.log1p(baseline[low]) - np.log1p(observed[low]))
            )
        )
        candidate_low_error = float(
            np.median(
                np.abs(np.log1p(candidate[low]) - np.log1p(observed[low]))
            )
        )
        station_rows.append(
            {
                "station_name": station_name,
                "reach_id": int(group["reach_id"].iloc[0]),
                "observed_bfi": float(prior["observed_bfi"]),
                "baseline_bfi": float(prior["baseline_bfi"]),
                "candidate_bfi": float(prior["candidate_bfi"]),
                "bfi_absolute_error_improvement": float(
                    prior["baseline_absolute_bfi_difference"]
                    - prior["candidate_absolute_bfi_difference"]
                ),
                **{
                    f"baseline_{key}": value
                    for key, value in baseline_scores.items()
                },
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_scores.items()
                },
                "nse_delta": (
                    candidate_scores["nse"] - baseline_scores["nse"]
                ),
                "log_nse_delta": (
                    candidate_scores["log_nse"]
                    - baseline_scores["log_nse"]
                ),
                "baseline_lowflow_log_error": baseline_low_error,
                "candidate_lowflow_log_error": candidate_low_error,
                "lowflow_log_error_delta": (
                    candidate_low_error - baseline_low_error
                ),
            }
        )
    station = pd.DataFrame(station_rows)

    process = (
        candidate_model.groupby("reach_id", as_index=False)
        .agg(
            mean_preferential_precipitation_fraction=(
                "preferential_fraction", "mean"
            ),
            preferential_recharge_sum_mm=(
                "preferential_recharge_mm", "sum"
            ),
            total_recharge_sum_mm=("groundwater_recharge_mm", "sum"),
            interflow_sum_mm=("interflow_mm", "sum"),
            excess_sum_mm=("excess_mm", "sum"),
        )
    )
    process["preferential_share_of_recharge"] = (
        process["preferential_recharge_sum_mm"]
        / process["total_recharge_sum_mm"]
    )
    capacity["capacity_cv"] = (
        capacity["capacity_stdev_mm"] / capacity["capacity_mean_mm"]
    )
    station = (
        station.merge(process, on="reach_id", validate="one_to_one")
        .merge(
            capacity[["reach_id", "capacity_cv"]],
            on="reach_id",
            validate="one_to_one",
        )
    )
    station.to_csv(
        REPORT / "station_corrected_unit_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    month[
        [
            "station_name", "reach_id", "year", "month",
            "observed_q_m3s", "baseline_q_m3s", "candidate_q_m3s",
            "observed_baseflow_fraction",
            "routed_baseflow_fraction_baseline",
            "routed_baseflow_fraction_candidate",
        ]
    ].to_parquet(
        OUTPUTS / "station_month_corrected_m3s.parquet", index=False
    )

    metrics = {
        "station_count": int(len(station)),
        "cfs_per_m3s": CFS_PER_M3S,
        "median_observed_bfi": float(station["observed_bfi"].median()),
        "median_baseline_bfi": float(station["baseline_bfi"].median()),
        "median_candidate_bfi": float(station["candidate_bfi"].median()),
        "fraction_stations_bfi_error_improved": float(
            station["bfi_absolute_error_improvement"].gt(0).mean()
        ),
        "median_bfi_absolute_error_improvement": float(
            station["bfi_absolute_error_improvement"].median()
        ),
        "baseline_spatial_bfi_spearman": safe_spearman(
            station["observed_bfi"], station["baseline_bfi"]
        ),
        "candidate_spatial_bfi_spearman": safe_spearman(
            station["observed_bfi"], station["candidate_bfi"]
        ),
        "median_baseline_nse_corrected": float(
            station["baseline_nse"].median()
        ),
        "median_candidate_nse_corrected": float(
            station["candidate_nse"].median()
        ),
        "median_nse_delta_corrected": float(station["nse_delta"].median()),
        "fraction_stations_nse_improved_corrected": float(
            station["nse_delta"].gt(0).mean()
        ),
        "median_baseline_log_nse_corrected": float(
            station["baseline_log_nse"].median()
        ),
        "median_candidate_log_nse_corrected": float(
            station["candidate_log_nse"].median()
        ),
        "median_log_nse_delta_corrected": float(
            station["log_nse_delta"].median()
        ),
        "fraction_stations_log_nse_improved_corrected": float(
            station["log_nse_delta"].gt(0).mean()
        ),
        "median_baseline_relative_bias_corrected": float(
            station["baseline_relative_bias"].median()
        ),
        "median_candidate_relative_bias_corrected": float(
            station["candidate_relative_bias"].median()
        ),
        "median_lowflow_log_error_delta": float(
            station["lowflow_log_error_delta"].median()
        ),
        "fraction_stations_lowflow_log_error_improved": float(
            station["lowflow_log_error_delta"].lt(0).mean()
        ),
        "domain_preferential_share_of_recharge": float(
            candidate_model["preferential_recharge_mm"].sum()
            / candidate_model["groundwater_recharge_mm"].sum()
        ),
        "preference_vs_bfi_gain_spearman": safe_spearman(
            station["mean_preferential_precipitation_fraction"],
            station["candidate_bfi"] - station["baseline_bfi"],
        ),
        "capacity_cv_vs_bfi_gain_spearman": safe_spearman(
            station["capacity_cv"],
            station["candidate_bfi"] - station["baseline_bfi"],
        ),
    }
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["environment_name"] == "sparrow"
            and RUNTIME_IDENTITY["conda_default_env"] == "sparrow"
            and RUNTIME_IDENTITY["sys_prefix"]
            == RUNTIME_IDENTITY["expected_prefix"]
        ),
        "original_unit_evidence_exists": UNIT_IMAGE.exists(),
        "observed_unit_is_m3s": True,
        "same_97_station_sample": len(station) == 97,
        "protected_shijiao_present": (
            "石角站" in set(station["station_name"])
        ),
        "fixed_exclusions_absent": {
            "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
        }.isdisjoint(set(station["station_name"])),
        "all_station_bfi_errors_improve": (
            metrics["fraction_stations_bfi_error_improved"] == 1.0
        ),
        "corrected_nse_guardrail_passes": (
            metrics["median_nse_delta_corrected"] >= -0.05
        ),
        "corrected_log_nse_guardrail_passes": (
            metrics["median_log_nse_delta_corrected"] >= -0.05
        ),
        "corrected_lowflow_error_guardrail_passes": (
            metrics["median_lowflow_log_error_delta"] <= 0.05
        ),
        "spatial_bfi_guardrail_still_fails": (
            metrics["candidate_spatial_bfi_spearman"]
            - metrics["baseline_spatial_bfi_spearman"] < -0.10
        ),
        "preferential_recharge_dominates_candidate_recharge": (
            metrics["domain_preferential_share_of_recharge"] >= 0.75
        ),
        "no_model_rerun_or_calibration": True,
        "no_confirmation_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    corrected_flow_passed = (
        checks["corrected_nse_guardrail_passes"]
        and checks["corrected_log_nse_guardrail_passes"]
        and checks["corrected_lowflow_error_guardrail_passes"]
    )
    if corrected_flow_passed and checks["spatial_bfi_guardrail_still_fails"]:
        decision = (
            "UNIT_ERROR_INVALIDATED_FLOW_TRADEOFF_BUT_SPATIAL_BFI_FAILURE_REMAINS"
        )
        next_action = "RUN_PARAMETER_FREE_CAPACITY_VS_PREFERENCE_COMPONENT_ABLATION"
    elif not corrected_flow_passed:
        decision = "CORRECTED_UNITS_CONFIRM_FLOW_TRADEOFF"
        next_action = "REJECT_COMBINED_CANDIDATE_WITHOUT_TUNING"
    else:
        decision = "COMBINED_CANDIDATE_REQUIRES_FULL_GATE_REINTERPRETATION"
        next_action = "REASSESS_COMBINED_CANDIDATE_GATE"
    gate = {
        "run_id": "20260729_27",
        "phase": "tradeoff_and_discharge_unit_audit",
        "checks": checks,
        "metrics": metrics,
        "decision": decision,
        "authorized_next_action": next_action,
        "model_rerun": False,
        "parameters_changed": False,
        "observed_discharge_unit": "m3/s",
        "legacy_observed_column_names_are_wrong": True,
        "bfi_metrics_affected_by_unit_error": False,
        "magnitude_metrics_affected_by_unit_error": True,
        "confirmation_years_used": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 结构权衡与流量单位重审",
        "",
        f"- 结论：`{decision}`",
        f"- 下一动作：`{next_action}`",
        "",
        "## 单位审计",
        "",
        "- 原始年鉴表头：流量 `m³/s`。",
        "- `_23` 的观测列名 `q_cfs`/`observed_q_cfs` 错误，数值未转换。",
        "- 本轮把模型 cfs 除以 35.3146667215 后，与观测 m³/s 比较。",
        "- BFI 等无量纲指标不受影响。",
        "",
        "## 纠正后的候选效果",
        "",
        (
            f"- BFI 中位数：{metrics['median_baseline_bfi']:.3f} → "
            f"{metrics['median_candidate_bfi']:.3f}；观测 "
            f"{metrics['median_observed_bfi']:.3f}"
        ),
        (
            "- 所有站 BFI 误差均改善；改善中位数："
            f"{metrics['median_bfi_absolute_error_improvement']:.3f}"
        ),
        (
            f"- NSE 中位数：{metrics['median_baseline_nse_corrected']:.3f} → "
            f"{metrics['median_candidate_nse_corrected']:.3f}"
        ),
        (
            f"- log-NSE 中位数："
            f"{metrics['median_baseline_log_nse_corrected']:.3f} → "
            f"{metrics['median_candidate_log_nse_corrected']:.3f}"
        ),
        (
            f"- 流量相对偏差中位数："
            f"{metrics['median_baseline_relative_bias_corrected']:.1%} → "
            f"{metrics['median_candidate_relative_bias_corrected']:.1%}"
        ),
        (
            f"- 站际 BFI Spearman："
            f"{metrics['baseline_spatial_bfi_spearman']:.3f} → "
            f"{metrics['candidate_spatial_bfi_spearman']:.3f}"
        ),
        (
            f"- 优先补给占候选总补给："
            f"{metrics['domain_preferential_share_of_recharge']:.1%}"
        ),
        "",
        "## 解释",
        "",
        (
            "旧的极端负 NSE/log-NSE 主要由单位错配造成，不能据此否定候选。"
            "但 BFI 空间排序下降是无量纲真实失败，且候选补给几乎由优先路径主导。"
            "组合试验不能因果区分容量分布与优先路径，因此下一轮只允许无参数组件消融。"
        ),
    ]
    (REPORT / "technical_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    sources = [
        PARENT_GATE, MONTH, STATIONS, CANDIDATE_MODEL, CAPACITY_STATS,
        UNIT_IMAGE, RUN / "experiment_contract.md",
        RUN / "scripts" / "diagnose_tradeoff_and_units.py",
    ]
    products = [
        REPORT / "station_corrected_unit_tradeoff.csv",
        OUTPUTS / "station_month_corrected_m3s.parquet",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_identity": RUNTIME_IDENTITY,
        "sources": [
            record(path, "tradeoff_unit_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "tradeoff_unit_product", "derived")
            for path in products
        ],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(provenance["sources"] + provenance["products"]).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
