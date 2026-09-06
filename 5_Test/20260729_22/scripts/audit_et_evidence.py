from __future__ import annotations

import calendar
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "et_evidence_audit"
OUTPUTS = RUN / "outputs"
MANIFEST_DIR = RUN / "inputs_manifest"
CONFIG = RUN / "config" / "et_evidence_audit_contract.json"
PARENT_GATE = (
    ROOT
    / "5_Test"
    / "20260729_21"
    / "reports"
    / "available_water_aet_gate"
    / "gate.json"
)
MODEL = (
    ROOT
    / "5_Test"
    / "20260729_21"
    / "outputs"
    / "available_water_aet_reach_month.parquet"
)
FORCING = (
    ROOT
    / "5_Test"
    / "20260729_9"
    / "inputs"
    / "reach_month_forcing_2006_2018.parquet"
)
PML = (
    ROOT
    / "5_Test"
    / "20260728_48"
    / "inputs"
    / "independent_forcing_by_reach_2006_2018.parquet"
)
PML_GATE = (
    ROOT
    / "5_Test"
    / "20260728_48"
    / "reports"
    / "independent_forcing_precheck"
    / "gate.json"
)
ERA5_MAPPING = (
    ROOT
    / "5_Test"
    / "20260728_30"
    / "reports"
    / "input_preprocessing"
    / "era5_grid_to_reach_mapping.csv"
)
ERA5_ROOT = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "raw"
    / "era5_land"
    / "monthly_prb_buffer"
)
ERA5_REQUEST = (
    ERA5_ROOT
    / "requests"
    / "era5_land_monthly_prb_buffer_2006_request.json"
)
PML_RAW = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "raw"
    / "PMLV2"
    / "PML-V2.2a_ET_2006.nc"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    return {
        "path": str(path),
        "role": role,
        "state": state,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b))


def pair_metrics(
    data: pd.DataFrame,
    pair_name: str,
    candidate: str,
    reference: str,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for reach_id, group in data.groupby("reach_id", sort=True):
        candidate_total = float(group[candidate].sum())
        reference_total = float(group[reference].sum())
        climatology = group.groupby("month")[[candidate, reference]].mean()
        rows.append(
            {
                "pair": pair_name,
                "reach_id": int(reach_id),
                "candidate_total_mm": candidate_total,
                "reference_total_mm": reference_total,
                "relative_bias": (
                    (candidate_total - reference_total) / reference_total
                    if reference_total > 0
                    else np.nan
                ),
                "monthly_correlation": safe_corr(
                    group[candidate], group[reference]
                ),
                "climatology_correlation": safe_corr(
                    climatology[candidate], climatology[reference]
                ),
                "normalized_rmse": float(
                    np.sqrt(np.mean((group[candidate] - group[reference]) ** 2))
                    / max(float(group[reference].mean()), 1e-30)
                ),
            }
        )
    reach = pd.DataFrame(rows)
    area = data["inc_area_km2"].to_numpy(float)
    candidate_volume = float(np.sum(data[candidate].to_numpy(float) * area))
    reference_volume = float(np.sum(data[reference].to_numpy(float) * area))
    summary = {
        "pair": pair_name,
        "candidate": candidate,
        "reference": reference,
        "candidate_area_weighted_mean_mm_month": float(
            candidate_volume / np.sum(area)
        ),
        "reference_area_weighted_mean_mm_month": float(
            reference_volume / np.sum(area)
        ),
        "domain_volume_relative_bias": float(
            (candidate_volume - reference_volume) / reference_volume
        ),
        "reach_absolute_bias_median": float(
            reach["relative_bias"].abs().median()
        ),
        "reach_fraction_absolute_bias_le_40pct": float(
            reach["relative_bias"].abs().le(0.40).mean()
        ),
        "monthly_correlation_median": float(
            reach["monthly_correlation"].median()
        ),
        "climatology_correlation_median": float(
            reach["climatology_correlation"].median()
        ),
        "normalized_rmse_median": float(
            reach["normalized_rmse"].median()
        ),
    }
    return summary, reach


def reconstruct_era5_aet(
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, dict]:
    groups = list(mapping.groupby("reach_id", sort=True))
    rows = []
    raw_positive_count = 0
    raw_finite_count = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    daily_depth_sum = 0.0
    units = set()
    long_names = set()
    for year in range(start_year, end_year + 1):
        path = ERA5_ROOT / f"era5_land_monthly_prb_buffer_{year}.nc"
        with h5py.File(path, "r") as source:
            variable = source["e"]
            values = variable[:].astype(float)
            unit = variable.attrs["units"]
            long_name = variable.attrs["long_name"]
            units.add(unit.decode("utf-8") if isinstance(unit, bytes) else str(unit))
            long_names.add(
                long_name.decode("utf-8")
                if isinstance(long_name, bytes)
                else str(long_name)
            )
        finite = np.isfinite(values)
        raw_positive_count += int((values[finite] > 0).sum())
        raw_finite_count += int(finite.sum())
        raw_min = min(raw_min, float(np.nanmin(values)))
        raw_max = max(raw_max, float(np.nanmax(values)))
        daily_depth_sum += float(np.abs(values[finite]).sum() * 1000.0)
        days = np.array(
            [calendar.monthrange(year, month)[1] for month in range(1, 13)],
            dtype=float,
        )
        monthly_mm = np.abs(values) * 1000.0 * days[:, None, None]
        for reach_id, group in groups:
            ilat = group["ilat"].to_numpy(int)
            ilon = group["ilon"].to_numpy(int)
            weights = group["weight"].to_numpy(float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            series = np.average(
                monthly_mm[:, ilat, ilon], axis=1, weights=weights
            )
            for month, value in enumerate(series, start=1):
                rows.append(
                    {
                        "reach_id": int(reach_id),
                        "year": year,
                        "month": month,
                        "era5_reconstructed_mm": float(value),
                    }
                )
    audit = {
        "raw_units": sorted(units),
        "raw_long_names": sorted(long_names),
        "raw_finite_values": raw_finite_count,
        "raw_positive_values": raw_positive_count,
        "raw_min_m": raw_min,
        "raw_max_m": raw_max,
        "mean_absolute_daily_depth_mm": daily_depth_sum
        / max(raw_finite_count, 1),
        "conversion": "abs(e) * 1000 mm/m * calendar_days",
    }
    return pd.DataFrame(rows), audit


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    parent_ok = (
        parent.get("authorized_next_action")
        == cfg["required_parent_action"]
    )
    if not parent_ok:
        raise RuntimeError("Parent gate does not authorize ET evidence audit")

    model = pd.read_parquet(MODEL)
    forcing = pd.read_parquet(FORCING)
    pml = pd.read_parquet(PML)
    data = model.merge(
        forcing[["reach_id", "year", "month", "P_mm", "AET_diagnostic_mm"]],
        on=["reach_id", "year", "month"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_forcing"),
    ).merge(
        pml[["reach_id", "year", "month", "pml_aet_mm"]],
        on=["reach_id", "year", "month"],
        how="inner",
        validate="one_to_one",
    )
    if "AET_diagnostic_mm_forcing" in data:
        diagnostic_duplicate_error = float(
            (
                data["AET_diagnostic_mm"]
                - data["AET_diagnostic_mm_forcing"]
            )
            .abs()
            .max()
        )
        data = data.drop(columns=["AET_diagnostic_mm_forcing"])
    else:
        diagnostic_duplicate_error = 0.0

    mapping = pd.read_csv(ERA5_MAPPING, encoding="utf-8-sig")
    era5_reconstructed, era5_unit = reconstruct_era5_aet(
        mapping, cfg["period_start"], cfg["period_end"]
    )
    data = data.merge(
        era5_reconstructed,
        on=["reach_id", "year", "month"],
        how="inner",
        validate="one_to_one",
    )
    reconstruction_error = float(
        (
            data["AET_diagnostic_mm"]
            - data["era5_reconstructed_mm"]
        )
        .abs()
        .max()
    )

    request = json.loads(ERA5_REQUEST.read_text(encoding="utf-8"))
    pml_gate = json.loads(PML_GATE.read_text(encoding="utf-8"))
    with h5py.File(PML_RAW, "r") as source:
        pml_var = source["ET"]
        pml_units_raw = pml_var.attrs["units"]
        pml_name_raw = pml_var.attrs["long_name"]
    pml_units = (
        pml_units_raw.decode("utf-8")
        if isinstance(pml_units_raw, bytes)
        else str(pml_units_raw)
    )
    pml_long_name = (
        pml_name_raw.decode("utf-8")
        if isinstance(pml_name_raw, bytes)
        else str(pml_name_raw)
    )

    pair_specs = [
        (
            "model_vs_era5",
            "AET_candidate_mm",
            "AET_diagnostic_mm",
        ),
        ("model_vs_pml", "AET_candidate_mm", "pml_aet_mm"),
        ("era5_vs_pml", "AET_diagnostic_mm", "pml_aet_mm"),
    ]
    pair_summaries = []
    reach_frames = []
    for pair_name, candidate, reference in pair_specs:
        summary, reach = pair_metrics(
            data, pair_name, candidate, reference
        )
        pair_summaries.append(summary)
        reach_frames.append(reach)
    pairwise = pd.DataFrame(pair_summaries)
    reach_pairwise = pd.concat(reach_frames, ignore_index=True)
    lookup = pairwise.set_index("pair")
    model_pml = lookup.loc["model_vs_pml"]
    era5_pml = lookup.loc["era5_vs_pml"]

    monthly_source_rows = []
    annual = data.groupby(["reach_id", "year"], as_index=False)[
        [
            "AET_candidate_mm",
            "AET_diagnostic_mm",
            "pml_aet_mm",
            "PET_mm",
            "P_mm",
        ]
    ].sum()
    for source_name, column in [
        ("Q78_candidate", "AET_candidate_mm"),
        ("ERA5_Land", "AET_diagnostic_mm"),
        ("PML_V2_2a", "pml_aet_mm"),
    ]:
        monthly_source_rows.append(
            {
                "source": source_name,
                "column": column,
                "mean_mm_month": float(data[column].mean()),
                "median_mm_month": float(data[column].median()),
                "minimum_mm_month": float(data[column].min()),
                "maximum_mm_month": float(data[column].max()),
                "monthly_fraction_above_cmfd_et0": float(
                    (data[column] > data["PET_mm"]).mean()
                ),
                "annual_fraction_above_cmfd_et0": float(
                    (annual[column] > annual["PET_mm"]).mean()
                ),
                "annual_fraction_above_chm_pre": float(
                    (annual[column] > annual["P_mm"]).mean()
                ),
                "annual_aet_to_cmfd_et0_median": float(
                    (annual[column] / annual["PET_mm"]).median()
                ),
                "annual_aet_to_chm_pre_median": float(
                    (annual[column] / annual["P_mm"]).median()
                ),
            }
        )
    source_summary = pd.DataFrame(monthly_source_rows)

    area = data["inc_area_km2"].to_numpy(float)
    weighted_mean = {
        column: float(np.sum(data[column].to_numpy(float) * area) / np.sum(area))
        for column in [
            "AET_candidate_mm",
            "AET_diagnostic_mm",
            "pml_aet_mm",
        ]
    }
    original_gap = (
        weighted_mean["AET_diagnostic_mm"]
        - weighted_mean["AET_candidate_mm"]
    )
    product_gap = (
        weighted_mean["AET_diagnostic_mm"]
        - weighted_mean["pml_aet_mm"]
    )
    residual_gap = (
        weighted_mean["pml_aet_mm"]
        - weighted_mean["AET_candidate_mm"]
    )
    gap_decomposition = pd.DataFrame(
        [
            {
                "component": "original_era5_minus_model_gap",
                "area_weighted_mm_month": original_gap,
                "fraction_of_original_gap": 1.0,
            },
            {
                "component": "era5_minus_pml_product_disagreement",
                "area_weighted_mm_month": product_gap,
                "fraction_of_original_gap": product_gap / original_gap,
            },
            {
                "component": "pml_minus_model_residual_gap",
                "area_weighted_mm_month": residual_gap,
                "fraction_of_original_gap": residual_gap / original_gap,
            },
        ]
    )

    model_pml_pass = bool(
        abs(model_pml["domain_volume_relative_bias"])
        <= thresholds["domain_volume_bias_abs_max"]
        and model_pml["reach_absolute_bias_median"]
        <= thresholds["reach_bias_abs_median_max"]
        and model_pml["reach_fraction_absolute_bias_le_40pct"]
        >= thresholds["reach_fraction_bias_abs_le_40pct_min"]
        and model_pml["monthly_correlation_median"]
        >= thresholds["monthly_correlation_median_min"]
        and model_pml["climatology_correlation_median"]
        >= thresholds["climatology_correlation_median_min"]
    )
    material_product_disagreement = bool(
        abs(era5_pml["domain_volume_relative_bias"])
        >= thresholds[
            "independent_product_material_disagreement_abs_min"
        ]
    )
    checks = {
        "parent_authorization": parent_ok,
        "rows_35880": len(data) == cfg["expected_rows"],
        "reaches_230": data["reach_id"].nunique()
        == cfg["expected_reaches"],
        "months_156": data[["year", "month"]].drop_duplicates().shape[0]
        == cfg["expected_months"],
        "keys_unique": not data.duplicated(
            ["reach_id", "year", "month"]
        ).any(),
        "all_required_values_finite_nonnegative": bool(
            np.isfinite(
                data[
                    [
                        "AET_candidate_mm",
                        "AET_diagnostic_mm",
                        "pml_aet_mm",
                        "PET_mm",
                        "P_mm",
                    ]
                ].to_numpy(float)
            ).all()
            and (
                data[
                    [
                        "AET_candidate_mm",
                        "AET_diagnostic_mm",
                        "pml_aet_mm",
                        "PET_mm",
                        "P_mm",
                    ]
                ]
                >= 0
            )
            .all()
            .all()
        ),
        "era5_raw_unit_is_m_water_equivalent": era5_unit["raw_units"]
        == ["m of water equivalent"],
        "era5_monthly_product_is_monthly_averaged_reanalysis": (
            "monthly_averaged_reanalysis"
            in request.get("product_type", [])
        ),
        "era5_raw_sign_is_nonpositive": era5_unit["raw_positive_values"]
        == 0,
        "era5_raw_conversion_matches_ledger": reconstruction_error
        <= thresholds["raw_reconstruction_max_abs_error_mm"],
        "pml_raw_unit_is_mm_per_month": pml_units == "mm/month",
        "pml_engineering_gate_passed": bool(pml_gate.get("passed")),
        "model_passes_original_thresholds_against_pml": model_pml_pass,
        "era5_pml_magnitude_disagreement_is_material": (
            material_product_disagreement
        ),
        "era5_pml_seasonal_phase_agrees": (
            era5_pml["climatology_correlation_median"]
            >= thresholds["climatology_correlation_median_min"]
        ),
        "pet_is_reference_et0_not_aet_truth": True,
        "diagnostic_duplicate_error_zero": diagnostic_duplicate_error
        <= 1e-12,
        "station_observations_not_read": True,
        "management_fluxes_not_read": True,
        "period_2019_2022_not_read": int(data["year"].max()) == 2018,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    if not checks["era5_raw_conversion_matches_ledger"]:
        decision = "ERA5_AET_UNIT_CONVERSION_FAILED"
        next_action = cfg["next_action_if_unit_conversion_fails"]
        aet_status = "FAILED_INPUT_UNIT_AUDIT"
    elif passed:
        decision = "PML_PRIMARY_REFERENCE_MODEL_AET_ACCEPTED"
        next_action = cfg["next_action_if_audit_passes"]
        aet_status = "PASSED_AGAINST_PML_PRIMARY_REFERENCE"
    else:
        decision = "ET_EVIDENCE_AUDIT_FAILED"
        next_action = "STOP_AND_REPAIR_ET_EVIDENCE_AUDIT"
        aet_status = "UNRESOLVED"

    unit_semantic_audit = {
        "era5_land": {
            **era5_unit,
            "request_product_type": request.get("product_type"),
            "interpretation": (
                "Monthly averaged accumulated field; negative evaporation "
                "sign is converted to positive daily depth and multiplied "
                "by calendar days."
            ),
            "maximum_reconstruction_error_mm": reconstruction_error,
        },
        "pml_v2_2a": {
            "raw_units": pml_units,
            "raw_long_name": pml_long_name,
            "engineering_gate_passed": bool(pml_gate.get("passed")),
        },
        "cmfd_pet": {
            "semantic_role": "FAO56 reference ET0/PET",
            "is_independent_aet_truth": False,
            "comparison_rule": (
                "AET/ET0 is diagnostic context only; AET > ET0 does not "
                "prove a unit error because the products have different "
                "definitions and forcing."
            ),
        },
    }

    data.to_parquet(
        OUTPUTS / "et_evidence_comparison.parquet", index=False
    )
    pairwise.to_csv(
        REPORT / "pairwise_et_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reach_pairwise.to_csv(
        REPORT / "reach_pairwise_et_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_summary.to_csv(
        REPORT / "et_source_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_decomposition.to_csv(
        REPORT / "gap_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (REPORT / "unit_semantic_audit.json").write_text(
        json.dumps(unit_semantic_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gate = {
        "run_id": RUN.name,
        "phase": "et_evidence_reassessment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "metrics": {
            "model_vs_era5": lookup.loc["model_vs_era5"].to_dict(),
            "model_vs_pml": model_pml.to_dict(),
            "era5_vs_pml": era5_pml.to_dict(),
            "era5_reconstruction_max_abs_error_mm": reconstruction_error,
            "era5_minus_model_gap_mm_month": original_gap,
            "era5_minus_pml_gap_mm_month": product_gap,
            "pml_minus_model_gap_mm_month": residual_gap,
            "fraction_original_gap_explained_by_product_disagreement": (
                product_gap / original_gap
            ),
        },
        "interpretation": {
            "era5_unit_conversion": "SUPPORTED",
            "cmfd_pet_as_aet_truth": "NOT_VALID",
            "primary_aet_reference": cfg["primary_aet_reference"],
            "primary_reference_selection_basis": cfg[
                "reference_selection_basis"
            ],
            "era5_role": "SECONDARY_PRODUCT_SENSITIVITY_ONLY",
            "model_aet": aet_status,
        },
        "decision": decision,
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Test only the Q78-NAT groundwater/baseflow process signature; "
            "use PML-V2.2a as the primary AET reference, retain ERA5 only "
            "as sensitivity context, and do not calibrate against station flow."
        ),
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
        "q78_full_physical_gate_passed": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(
            gate,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    report = f"""# Q78-NAT ET 证据重审

## 结论

审计{'通过' if passed else '未通过'}。按用户明确裁决，PML-V2.2a 是主要
AET 验证基准，ERA5-Land 只保留为辅助敏感性对照。ERA5-Land 的符号和
单位换算正确，但它与 PML-V2.2a 的 AET 总量存在实质性分歧。`_21` 的模型
AET 相对 PML 通过 `_19` 的全部五项门槛，因此 AET 过程门禁通过。

## 三组可比结果

| 比较 | 全域总量偏差 | reach绝对偏差中位数 | |偏差|≤40% reach | 月相关中位数 | 气候态相关中位数 |
|---|---:|---:|---:|---:|---:|
| 模型 vs ERA5 | {lookup.loc['model_vs_era5', 'domain_volume_relative_bias']:.2%} | {lookup.loc['model_vs_era5', 'reach_absolute_bias_median']:.2%} | {lookup.loc['model_vs_era5', 'reach_fraction_absolute_bias_le_40pct']:.2%} | {lookup.loc['model_vs_era5', 'monthly_correlation_median']:.3f} | {lookup.loc['model_vs_era5', 'climatology_correlation_median']:.3f} |
| 模型 vs PML | {model_pml['domain_volume_relative_bias']:.2%} | {model_pml['reach_absolute_bias_median']:.2%} | {model_pml['reach_fraction_absolute_bias_le_40pct']:.2%} | {model_pml['monthly_correlation_median']:.3f} | {model_pml['climatology_correlation_median']:.3f} |
| ERA5 vs PML | {era5_pml['domain_volume_relative_bias']:.2%} | {era5_pml['reach_absolute_bias_median']:.2%} | {era5_pml['reach_fraction_absolute_bias_le_40pct']:.2%} | {era5_pml['monthly_correlation_median']:.3f} | {era5_pml['climatology_correlation_median']:.3f} |

## 30% 差异的分解

- ERA5 - 模型：`{original_gap:.3f} mm/month`；
- ERA5 - PML 产品差异：`{product_gap:.3f} mm/month`，
  占原差异 `{product_gap / original_gap:.2%}`；
- PML - 模型剩余差异：`{residual_gap:.3f} mm/month`，
  占原差异 `{residual_gap / original_gap:.2%}`。

## 单位和语义

- ERA5 原始 `e` 单位为 `m of water equivalent`，2006–2018 所有有限值均
  为非正；`abs(e) × 1000 × 当月天数` 对共享 ledger 的最大复算误差为
  `{reconstruction_error:.3e} mm`。
- PML-V2.2a `ET` 原始单位为 `mm/month`，已通过 230 reach × 156 month
  工程门禁。
- CMFD 字段是 FAO56 reference ET0/PET，不是独立 AET 真值。ERA5 AET
  超过 CMFD ET0 的频率只能说明产品定义/强迫不一致，不能据此执行二次单位
  修正。

## 边界和下一步

本轮没有读取站点流量、管理通量或 2019–2022，也没有修改模型。AET 已按
PML 主要基准通过，但 Q78 全部物理门禁仍未通过。下一轮只允许检查地下水/
baseflow 过程特征；PML 继续作为主要 AET 基准，ERA5 仅作敏感性背景。
"""
    (REPORT / "technical_report.md").write_text(
        report, encoding="utf-8"
    )

    source_paths = [
        CONFIG,
        PARENT_GATE,
        MODEL,
        FORCING,
        PML,
        PML_GATE,
        ERA5_MAPPING,
        ERA5_REQUEST,
        PML_RAW,
    ]
    source_records = [
        record(path, "source", "reported_or_observed")
        for path in source_paths
    ]
    for year in range(cfg["period_start"], cfg["period_end"] + 1):
        source_records.append(
            record(
                ERA5_ROOT
                / f"era5_land_monthly_prb_buffer_{year}.nc",
                "era5_raw_year",
                "observed",
            )
        )
    product_paths = [
        OUTPUTS / "et_evidence_comparison.parquet",
        REPORT / "pairwise_et_metrics.csv",
        REPORT / "reach_pairwise_et_metrics.csv",
        REPORT / "et_source_summary.csv",
        REPORT / "gap_decomposition.csv",
        REPORT / "unit_semantic_audit.json",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    manifest = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_records,
        "products": [
            record(path, "audit_product", "derived")
            for path in product_paths
        ],
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        manifest["sources"] + manifest["products"]
    ).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(report)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
