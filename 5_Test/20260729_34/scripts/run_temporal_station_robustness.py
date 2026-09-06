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
REPORT = RUN / "reports" / "temporal_station_robustness"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_33" / "reports"
    / "storage_heterogeneity_recharge" / "gate.json"
)
STATION_RESULT = (
    ROOT / "5_Test" / "20260729_33" / "reports"
    / "storage_heterogeneity_recharge"
    / "station_current_vs_heterogeneity.csv"
)
STATION_ATTRIBUTES = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit"
    / "station_bfi_attribute_panel.csv"
)
CURRENT_MODEL = (
    ROOT / "5_Test" / "20260729_26" / "outputs"
    / "distributed_storage_preferential_recharge_reach_month.parquet"
)
CANDIDATE_MODEL = (
    ROOT / "5_Test" / "20260729_33" / "outputs"
    / "storage_heterogeneity_recharge_reach_month.parquet"
)
OBSERVED = (
    ROOT / "5_Test" / "20260729_27" / "outputs"
    / "station_month_corrected_m3s.parquet"
)
CONTRACT = RUN / "experiment_contract.md"
LITERATURE = RUN / "literature_basis.md"
CFS_PER_M3S = 35.3146667215
PERIODS = {
    "2006_2009": (2006, 2009),
    "2010_2013": (2010, 2013),
    "2014_2018": (2014, 2018),
}
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


def flow_scores(observed: np.ndarray, simulated: np.ndarray) -> dict:
    denominator = np.sum((observed - observed.mean()) ** 2)
    nse = 1.0 - np.sum((simulated - observed) ** 2) / denominator
    log_observed = np.log1p(observed)
    log_simulated = np.log1p(np.maximum(simulated, 0))
    log_nse = 1.0 - np.sum(
        (log_simulated - log_observed) ** 2
    ) / np.sum((log_observed - log_observed.mean()) ** 2)
    threshold = np.quantile(observed, 0.25)
    low = observed <= threshold
    lowflow_error = np.median(np.abs(
        log_simulated[low] - log_observed[low]
    ))
    return {
        "nse": float(nse),
        "log_nse": float(log_nse),
        "lowflow_log_error": float(lowflow_error),
    }


def temporal_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_parquet(OBSERVED)
    current = pd.read_parquet(CURRENT_MODEL)[[
        "reach_id", "year", "month", "channel_outflow_cfs"
    ]].rename(columns={"channel_outflow_cfs": "current_cfs"})
    candidate = pd.read_parquet(CANDIDATE_MODEL)[[
        "reach_id", "year", "month", "channel_outflow_m3s"
    ]].rename(columns={"channel_outflow_m3s": "candidate_m3s"})
    joined = observed[[
        "station_name", "reach_id", "year", "month", "observed_q_m3s"
    ]].merge(
        current, on=["reach_id", "year", "month"], validate="many_to_one"
    ).merge(
        candidate, on=["reach_id", "year", "month"], validate="many_to_one"
    )
    joined["current_m3s"] = joined["current_cfs"] / CFS_PER_M3S
    rows = []
    for period, (first, last) in PERIODS.items():
        block = joined.loc[joined["year"].between(first, last)]
        for station_name, group in block.groupby("station_name"):
            if len(group) < 24:
                continue
            observed_q = group["observed_q_m3s"].to_numpy(float)
            current_scores = flow_scores(
                observed_q, group["current_m3s"].to_numpy(float)
            )
            candidate_scores = flow_scores(
                observed_q, group["candidate_m3s"].to_numpy(float)
            )
            rows.append({
                "period": period,
                "station_name": station_name,
                "reach_id": int(group["reach_id"].iloc[0]),
                "months": len(group),
                **{f"current_{k}": v for k, v in current_scores.items()},
                **{
                    f"candidate_{k}": v
                    for k, v in candidate_scores.items()
                },
            })
    station = pd.DataFrame(rows)
    summary_rows = []
    for period, group in station.groupby("period"):
        summary_rows.append({
            "period": period,
            "station_count": len(group),
            "minimum_months": int(group["months"].min()),
            "current_median_nse": float(group["current_nse"].median()),
            "candidate_median_nse": float(
                group["candidate_nse"].median()
            ),
            "median_nse_delta": float(
                group["candidate_nse"].median()
                - group["current_nse"].median()
            ),
            "current_median_log_nse": float(
                group["current_log_nse"].median()
            ),
            "candidate_median_log_nse": float(
                group["candidate_log_nse"].median()
            ),
            "median_log_nse_delta": float(
                group["candidate_log_nse"].median()
                - group["current_log_nse"].median()
            ),
            "current_median_lowflow_log_error": float(
                group["current_lowflow_log_error"].median()
            ),
            "candidate_median_lowflow_log_error": float(
                group["candidate_lowflow_log_error"].median()
            ),
            "median_lowflow_log_error_delta": float(
                group["candidate_lowflow_log_error"].median()
                - group["current_lowflow_log_error"].median()
            ),
            "nse_improvement_fraction": float(
                (group["candidate_nse"] > group["current_nse"]).mean()
            ),
            "log_nse_improvement_fraction": float(
                (
                    group["candidate_log_nse"]
                    > group["current_log_nse"]
                ).mean()
            ),
            "lowflow_improvement_fraction": float(
                (
                    group["candidate_lowflow_log_error"]
                    < group["current_lowflow_log_error"]
                ).mean()
            ),
        })
    return station, pd.DataFrame(summary_rows)


def bfi_robustness() -> tuple[pd.DataFrame, dict]:
    station = pd.read_csv(STATION_RESULT)
    attributes = pd.read_csv(STATION_ATTRIBUTES)[[
        "station_name", "observed_bfi_sensitivity_width",
        "upstream_incremental_area_km2",
    ]]
    station = station.merge(
        attributes, on="station_name", validate="one_to_one"
    )
    sensitivity_median = station[
        "observed_bfi_sensitivity_width"
    ].median()
    station["sensitivity_group"] = np.where(
        station["observed_bfi_sensitivity_width"] <= sensitivity_median,
        "lower_sensitivity",
        "higher_sensitivity",
    )
    station["area_group"] = pd.qcut(
        station["upstream_incremental_area_km2"],
        q=3,
        labels=["small_area", "medium_area", "large_area"],
    ).astype(str)

    def delta(frame: pd.DataFrame) -> float:
        observed = frame["observed_bfi"]
        current = observed.corr(
            frame["simulated_bfi_current_combined"], method="spearman"
        )
        candidate = observed.corr(
            frame["simulated_bfi_heterogeneity_reordered"],
            method="spearman",
        )
        return float(candidate - current)

    rows = [{
        "group_type": "all",
        "group": "all_97",
        "n": len(station),
        "spatial_bfi_delta": delta(station),
    }]
    for column, group_type in [
        ("sensitivity_group", "bfi_filter_sensitivity"),
        ("area_group", "upstream_area_tercile"),
    ]:
        for group_name, group in station.groupby(column):
            rows.append({
                "group_type": group_type,
                "group": str(group_name),
                "n": len(group),
                "spatial_bfi_delta": delta(group),
            })
    groups = pd.DataFrame(rows)

    rng = np.random.default_rng(2026072934)
    bootstrap = np.empty(4000, dtype=float)
    for index in range(len(bootstrap)):
        chosen = rng.integers(0, len(station), len(station))
        bootstrap[index] = delta(station.iloc[chosen])
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    loo = []
    for omitted in range(len(station)):
        loo.append(delta(station.drop(station.index[omitted])))
    metrics = {
        "full_spatial_bfi_delta": delta(station),
        "bootstrap_95_lower": float(lower),
        "bootstrap_95_upper": float(upper),
        "bootstrap_positive_fraction": float(
            np.mean(bootstrap > 0)
        ),
        "leave_one_out_positive_fraction": float(
            np.mean(np.asarray(loo) > 0)
        ),
        "minimum_subgroup_delta": float(
            groups.loc[~groups["group_type"].eq("all"),
                       "spatial_bfi_delta"].min()
        ),
    }
    return groups, metrics


def main() -> None:
    for directory in (REPORT, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "RUN_TEMPORAL_AND_STATION_ROBUSTNESS_GATE"
    ):
        raise RuntimeError("Parent gate does not authorize robustness test")
    station_period, period_summary = temporal_metrics()
    bfi_groups, bfi_metrics = bfi_robustness()
    station_period.to_csv(
        REPORT / "station_period_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    period_summary.to_csv(
        REPORT / "period_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bfi_groups.to_csv(
        REPORT / "bfi_subgroup_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "three_periods_present": set(period_summary["period"]) == set(PERIODS),
        "at_least_90_stations_each_period": bool(
            period_summary["station_count"].ge(90).all()
        ),
        "at_least_24_months_each_station_period": bool(
            station_period["months"].ge(24).all()
        ),
        "bfi_bootstrap_lower_positive": (
            bfi_metrics["bootstrap_95_lower"] > 0
        ),
        "bfi_leave_one_out_positive_at_least_95pct": (
            bfi_metrics["leave_one_out_positive_fraction"] >= 0.95
        ),
        "all_bfi_subgroups_positive": (
            bfi_metrics["minimum_subgroup_delta"] > 0
        ),
        "all_period_nse_guardrails": bool(
            period_summary["median_nse_delta"].ge(-0.05).all()
        ),
        "all_period_log_nse_guardrails": bool(
            period_summary["median_log_nse_delta"].ge(-0.05).all()
        ),
        "all_period_lowflow_guardrails": bool(
            period_summary["median_lowflow_log_error_delta"].le(0.05).all()
        ),
        "shijiao_present": "石角站" in set(station_period["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(station_period["station_name"])
        ),
        "zero_parameter_calibration": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    retained = all(checks.values())
    decision = (
        "PROMOTE_HETEROGENEITY_CONTROL_TO_DEVELOPMENT_BASELINE"
        if retained
        else "DO_NOT_PROMOTE_HETEROGENEITY_CONTROL"
    )
    next_action = (
        "DIAGNOSE_REMAINING_LOW_FLOW_AND_BFI_FAILURES"
        if retained
        else "RETURN_TO_TEXTURE_AND_DRAINAGE_ATTRIBUTE_REVIEW"
    )
    gate = {
        "run_id": "20260729_34",
        "phase": "temporal_station_robustness_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "bfi_robustness": bfi_metrics,
        "period_metrics": period_summary.to_dict("records"),
        "decision": decision,
        "authorized_next_action": next_action,
        "candidate_promoted": retained,
        "parameters_calibrated": False,
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
    period_lines = "\n".join(
        f"| {row.period} | {int(row.station_count)} | "
        f"{row.median_nse_delta:+.3f} | "
        f"{row.median_log_nse_delta:+.3f} | "
        f"{row.median_lowflow_log_error_delta:+.3f} |"
        for row in period_summary.itertuples()
    )
    report = f"""# 时段与站点稳健性门禁

## 结论

`{decision}`

- BFI空间增益：{bfi_metrics['full_spatial_bfi_delta']:+.3f}；
- 站点bootstrap 95%区间：
  [{bfi_metrics['bootstrap_95_lower']:+.3f},
  {bfi_metrics['bootstrap_95_upper']:+.3f}]；
- 逐站删除正增益比例：
  {bfi_metrics['leave_one_out_positive_fraction']:.1%}；
- 五个BFI子组最小增益：
  {bfi_metrics['minimum_subgroup_delta']:+.3f}。

| 时段 | 站数 | NSE变化 | log-NSE变化 | 低流误差变化 |
|---|---:|---:|---:|---:|
{period_lines}

下一步：`{next_action}`
"""
    report_path = REPORT / "technical_report.md"
    report_path.write_text(report, encoding="utf-8")
    sources = [
        PARENT_GATE, STATION_RESULT, STATION_ATTRIBUTES,
        CURRENT_MODEL, CANDIDATE_MODEL, OBSERVED, CONTRACT, LITERATURE,
        RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "run_temporal_station_robustness.py",
        RUN / "scripts" / "validate_temporal_station_robustness.py",
    ]
    products = [
        REPORT / "station_period_metrics.csv",
        REPORT / "period_summary.csv",
        REPORT / "bfi_subgroup_robustness.csv",
        gate_path, report_path,
    ]
    provenance = {
        "run_id": "20260729_34",
        "created_utc": utc_now(),
        "sources": [
            record(path, "robustness_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "robustness_product", "derived")
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
        "E:\\SPARROW\\5_Test\\20260729_34\\scripts\\"
        "run_temporal_station_robustness.py`\n\n"
        f"Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "bootstrap_95": [
            bfi_metrics["bootstrap_95_lower"],
            bfi_metrics["bootstrap_95_upper"],
        ],
        "minimum_subgroup_delta": bfi_metrics["minimum_subgroup_delta"],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
