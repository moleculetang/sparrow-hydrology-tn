from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_18"
LEDGER = ROOT / "5_Test" / "20260729_9"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "aet_process_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "aet_process_contract.json"
PARENT_GATE = PARENT / "reports" / "spatial_q78_nat_core_gate" / "gate.json"
MODEL_PATH = PARENT / "outputs" / "spatial_q78_nat_reach_month.parquet"
FORCING_PATH = LEDGER / "inputs" / "reach_month_forcing_2006_2018.parquet"
STATIC_PATH = STATIC_RUN / "inputs" / "reach_static_direction_final.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.std(ddof=0) <= 0 or y.std(ddof=0) <= 0:
        return np.nan
    return float(x.corr(y))


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize AET diagnosis")

    model = pd.read_parquet(MODEL_PATH)
    forcing = pd.read_parquet(FORCING_PATH)[[
        "reach_id", "year", "month", "AET_diagnostic_mm"
    ]]
    static = pd.read_parquet(STATIC_PATH)[["reach_id", "inc_area_km2"]]
    data = model[[
        "reach_id", "year", "month", "AET_mm", "PET_mm"
    ]].merge(
        forcing, on=["reach_id", "year", "month"], how="left", validate="one_to_one"
    ).merge(static, on="reach_id", how="left", validate="many_to_one")
    if len(data) != 35880 or data.isna().any().any():
        raise RuntimeError("AET diagnostic join is incomplete")
    if data["AET_diagnostic_mm"].lt(0).any():
        raise RuntimeError("Independent AET diagnostic contains negative values")

    reach_rows = []
    for reach, group in data.groupby("reach_id"):
        model_total = group["AET_mm"].sum()
        diagnostic_total = group["AET_diagnostic_mm"].sum()
        climatology = group.groupby("month")[["AET_mm", "AET_diagnostic_mm"]].mean()
        reach_rows.append({
            "reach_id": int(reach),
            "model_aet_total_mm": model_total,
            "diagnostic_aet_total_mm": diagnostic_total,
            "relative_bias": (
                (model_total - diagnostic_total) / diagnostic_total
                if diagnostic_total > 0 else np.nan
            ),
            "monthly_correlation": safe_corr(
                group["AET_mm"], group["AET_diagnostic_mm"]
            ),
            "climatology_correlation": safe_corr(
                climatology["AET_mm"], climatology["AET_diagnostic_mm"]
            ),
            "normalized_rmse": float(
                np.sqrt(np.mean(
                    (group["AET_mm"] - group["AET_diagnostic_mm"]) ** 2
                )) / max(group["AET_diagnostic_mm"].mean(), 1e-30)
            ),
        })
    reach_metrics = pd.DataFrame(reach_rows)
    area_factor = data["inc_area_km2"] * 1000.0
    model_volume = float((data["AET_mm"] * area_factor).sum())
    diagnostic_volume = float((data["AET_diagnostic_mm"] * area_factor).sum())
    domain_bias = (model_volume - diagnostic_volume) / diagnostic_volume
    summary = {
        "domain_volume_relative_bias": domain_bias,
        "reach_absolute_bias_median": float(reach_metrics["relative_bias"].abs().median()),
        "reach_fraction_absolute_bias_le_40pct": float(
            reach_metrics["relative_bias"].abs().le(0.40).mean()
        ),
        "monthly_correlation_median": float(
            reach_metrics["monthly_correlation"].median()
        ),
        "climatology_correlation_median": float(
            reach_metrics["climatology_correlation"].median()
        ),
        "normalized_rmse_median": float(
            reach_metrics["normalized_rmse"].median()
        ),
    }
    thresholds = cfg["thresholds"]
    checks = {
        "parent_authorization": True,
        "rows_35880": len(data) == 35880,
        "reaches_230": data["reach_id"].nunique() == 230,
        "model_aet_nonnegative": data["AET_mm"].ge(-1e-12).all(),
        "model_aet_not_above_pet": (data["AET_mm"] - data["PET_mm"]).max() <= 1e-9,
        "diagnostic_aet_complete_nonnegative": (
            data["AET_diagnostic_mm"].notna().all()
            and data["AET_diagnostic_mm"].ge(0).all()
        ),
        "domain_volume_bias_within_20pct": (
            abs(domain_bias) <= thresholds["domain_volume_bias_abs_max"]
        ),
        "median_reach_bias_within_30pct": (
            summary["reach_absolute_bias_median"]
            <= thresholds["reach_bias_abs_median_max"]
        ),
        "at_least_70pct_reaches_within_40pct_bias": (
            summary["reach_fraction_absolute_bias_le_40pct"]
            >= thresholds["reach_fraction_bias_abs_le_40pct_min"]
        ),
        "median_monthly_correlation_at_least_0_60": (
            summary["monthly_correlation_median"]
            >= thresholds["monthly_correlation_median_min"]
        ),
        "median_climatology_correlation_at_least_0_80": (
            summary["climatology_correlation_median"]
            >= thresholds["climatology_correlation_median_min"]
        ),
        "station_observations_not_read": True,
        "management_fluxes_not_read": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    next_action = cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    reach_path = OUTPUTS / "aet_reach_diagnostics.parquet"
    reach_metrics.to_parquet(reach_path, index=False)
    pd.DataFrame([summary]).to_csv(
        REPORT / "aet_domain_summary.csv", index=False, encoding="utf-8-sig"
    )
    gate = {
        "run_id": cfg["run_id"],
        "phase": "aet_process_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "metrics": summary,
        "thresholds": thresholds,
        "decision": "AET_PROCESS_PASSED" if passed else "AET_PROCESS_FAILED",
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Assess groundwater and baseflow process plausibility without station-flow calibration."
            if passed else
            "Revise only the AET structure or prior using independent AET evidence; station-flow calibration remains forbidden."
        ),
        "aet_used_as_model_forcing": False,
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [name for name, value in checks.items() if not value]
    report = f"""# Q78-NAT 独立 AET 过程门禁

## 结论

AET 过程门禁{'通过' if passed else '未通过'}。ERA5-Land AET 只作为独立诊断，没有进入模型 forcing。

| 指标 | 数值 |
|---|---:|
| 全域面积加权累计相对偏差 | {domain_bias:.3%} |
| reach 绝对偏差中位数 | {summary['reach_absolute_bias_median']:.3%} |
| 偏差绝对值≤40%的reach比例 | {summary['reach_fraction_absolute_bias_le_40pct']:.3%} |
| 月相关系数中位数 | {summary['monthly_correlation_median']:.3f} |
| 12月气候态相关系数中位数 | {summary['climatology_correlation_median']:.3f} |
| 归一化RMSE中位数 | {summary['normalized_rmse_median']:.3f} |

失败检查：{', '.join(failed) if failed else '无'}。

本轮不涉及地下水、baseflow、站点流量技能或 2019–2022。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(MODEL_PATH), str(FORCING_PATH), str(STATIC_PATH)],
        "aet_diagnostic_role": "independent_diagnostic_only_not_model_forcing",
        "station_observation_files_read": [],
        "management_flux_files_read": [],
        "period_2019_2022_read": False,
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        CONFIG, RUN_DIR / "experiment_contract.md", RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "run_aet_process_gate.py",
        RUN_DIR / "scripts" / "validate_aet_process_gate.py",
        PARENT_GATE, MODEL_PATH, FORCING_PATH, STATIC_PATH,
    ]
    products = [
        reach_path, REPORT / "aet_domain_summary.csv", REPORT / "gate.json",
        REPORT / "technical_report.md", REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(path, "aet_gate_source", "reported_or_derived") for path in sources],
        "products": [record(path, "aet_gate_product", "derived") for path in products],
        "aet_used_as_model_forcing": False,
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "passed": passed,
        "metrics": summary,
        "failed_checks": failed,
        "authorized_next_action": next_action,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

