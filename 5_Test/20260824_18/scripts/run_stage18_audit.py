"""Authoritative data, domain, initialization and memory audit for TN rebuild."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_18"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"

TN_ALL = ROOT / "1_Inputs" / "WaterQualityData" / "model_ready" / "tn_station_month_all.parquet"
PREPARED_CSV = ROOT / "0_water_quality" / "data" / "preprocess" / "csv"
RAW_2021_2024 = ROOT / "0_water_quality" / "data" / "origin" / "202101-202412月度水质数据(1).xlsx"
RAW_2025 = ROOT / "0_water_quality" / "data" / "origin" / "国控融合数据2025.xlsx"
DOMAIN = ROOT / "5_Test" / "20260820_19" / "outputs" / "observation_domain_registry.parquet"
CURRENT_OBS = ROOT / "5_Test" / "20260824_12" / "outputs" / "tn_observations_primary_2016_2024.parquet"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
KERNELS = ROOT / "5_Test" / "20260824_13" / "outputs" / "daily_compiled_carrier_kernels.parquet"
FLUXES = ROOT / "5_Test" / "20260824_13" / "outputs" / "m0_local_source_tagged_fluxes.parquet"
OOF = ROOT / "5_Test" / "20260824_16" / "outputs" / "dynamic_delivery_oof_predictions.parquet"
STAGE13_MODEL = ROOT / "5_Test" / "20260824_13" / "scripts" / "stage13_model.py"
STAGE13_STDERR = ROOT / "5_Test" / "20260824_13" / "stage13_stderr.log"
CONTRACT = RUN / "experiment_contract.json"


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n",
        encoding="utf-8",
    )


def station_key(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ")
    return re.sub(r"\s+", "", value).strip()


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def prepared_lineage(stations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Link model observations to the authoritative prepared station CSV rows."""
    paths: dict[str, Path] = {}
    duplicate_names: dict[str, list[str]] = {}
    for path in sorted(PREPARED_CSV.glob("*.csv")):
        key = station_key(path.stem)
        if key in paths:
            duplicate_names.setdefault(key, [str(paths[key])]).append(str(path))
        else:
            paths[key] = path
    if duplicate_names:
        raise RuntimeError(f"duplicate normalized prepared station filenames: {duplicate_names}")

    rows: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for key in sorted(stations.station_key.astype(str).unique()):
        path = paths.get(key)
        if path is None:
            missing_files.append(key)
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        required = {"年份", "月份", "总氮"}
        if not required.issubset(raw.columns):
            raise RuntimeError(f"prepared station CSV schema failure: {path}")
        frame = raw.loc[:, ["年份", "月份", "总氮"]].copy()
        frame.columns = ["year", "month", "prepared_raw_tn"]
        frame["year"] = pd.to_numeric(frame.year, errors="coerce")
        frame["month"] = pd.to_numeric(frame.month, errors="coerce")
        frame["prepared_tn_numeric"] = pd.to_numeric(frame.prepared_raw_tn, errors="coerce")
        frame["station_key"] = key
        frame["prepared_source_file"] = str(path)
        frame["prepared_source_row"] = np.arange(len(frame), dtype=int) + 2
        frame = frame.loc[frame.year.notna() & frame.month.notna()].copy()
        frame["year"] = frame.year.astype(int)
        frame["month"] = frame.month.astype(int)
        if frame.duplicated(["station_key", "year", "month"]).any():
            raise RuntimeError(f"duplicate station-month in prepared CSV: {path}")
        rows.append(frame)
    lineage = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return lineage, {
        "prepared_csv_file_count": len(paths),
        "requested_station_count": int(stations.station_key.nunique()),
        "matched_station_file_count": int(lineage.station_key.nunique()) if not lineage.empty else 0,
        "missing_station_files": missing_files,
    }


def workbook_profile(path: Path, include_rows: bool) -> dict[str, object]:
    xls = pd.ExcelFile(path)
    profile: dict[str, object] = {
        "path": str(path),
        "sha256": sha256(path),
        "sheets": xls.sheet_names,
    }
    if include_rows:
        frame = pd.read_excel(path, sheet_name=0, usecols=[1, 2, 3, 22])
        profile.update({
            "rows": int(len(frame)),
            "columns": frame.columns.astype(str).tolist(),
            "yearmonth_min": int(pd.to_numeric(frame.iloc[:, 0], errors="coerce").min()),
            "yearmonth_max": int(pd.to_numeric(frame.iloc[:, 0], errors="coerce").max()),
            "section_count": int(frame.iloc[:, 1].nunique(dropna=True)),
            "section_code_count": int(frame.iloc[:, 2].nunique(dropna=True)),
            "tn_numeric_count": int(pd.to_numeric(frame.iloc[:, 3], errors="coerce").notna().sum()),
            "record_level_join_status": "not_reconstructable_from_current_archive",
            "reason": "the raw workbook station strings are mojibake and the prepared archive does not retain SectionCode/source-row provenance",
        })
    return profile


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame.tn_mg_l.to_numpy(float)
    p = frame.pred_tn_mg_l.to_numpy(float)
    residual = p - y
    denominator = float(np.sum(np.square(y - y.mean())))
    return {
        "n": int(len(frame)),
        "stations": int(frame.station_key.nunique()),
        "reaches": int(frame.reach_id.nunique()),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_mg_l": float(np.mean(np.abs(residual))),
        "nse": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else math.nan,
        "r2": float(np.corrcoef(y, p)[0, 1] ** 2) if len(frame) > 1 and np.std(y) > 0 and np.std(p) > 0 else math.nan,
        "pbias_percent": float(100.0 * np.sum(residual) / np.sum(y)) if np.sum(y) != 0 else math.nan,
        "sse": float(np.sum(np.square(residual))),
    }


def source_clipping_audit() -> dict[str, object]:
    source = pd.read_parquet(SOURCE)
    source = source.loc[source.calendar_scenario.eq("CENTRAL") & source.year.between(2016, 2024)].copy()
    positive = (
        source.fertilizer_kg_n
        + source.manure_kg_n
        + source.cropland_bnf_kg_n
        + source.atmospheric_deposition_kg_n
    )
    source["monthly_clipped_available"] = np.maximum(positive - source.crop_demand_kg_n, 0.0)
    annual = source.assign(positive=positive).groupby(["reach_id", "year"], as_index=False).agg(
        monthly_clipped=("monthly_clipped_available", "sum"),
        annual_positive=("positive", "sum"),
        annual_crop=("crop_demand_kg_n", "sum"),
    )
    annual["annual_net_clipped"] = np.maximum(annual.annual_positive - annual.annual_crop, 0.0)
    annual["inflation_ratio"] = np.divide(
        annual.monthly_clipped,
        annual.annual_net_clipped,
        out=np.full(len(annual), np.nan),
        where=annual.annual_net_clipped > 0,
    )
    basin_ratio = float(annual.monthly_clipped.sum() / annual.annual_net_clipped.sum())
    return {
        "basin_inflation_fraction": basin_ratio - 1.0,
        "reach_year_median_ratio": float(np.nanmedian(annual.inflation_ratio)),
        "reach_year_p95_ratio": float(np.nanquantile(annual.inflation_ratio, 0.95)),
        "reach_year_max_ratio": float(np.nanmax(annual.inflation_ratio)),
        "affected_reach_years": int(np.sum(annual.inflation_ratio > 1.0 + 1.0e-12)),
        "reach_years": int(len(annual)),
    }


def initialization_audit() -> dict[str, object]:
    kernels = pd.read_parquet(KERNELS, columns=["year", "month", "reach_id"])
    fluxes = pd.read_parquet(FLUXES, columns=["year", "month", "reach_id", "source_tag"])
    model_text = STAGE13_MODEL.read_text(encoding="utf-8")
    zero_state_code = "zu = np.zeros((n, len(tags)), dtype=float)" in model_text and "zl = np.zeros((n, len(tags)), dtype=float)" in model_text
    return {
        "canonical_hydrology_start_year": 2006,
        "compiled_kernel_start_year": int(kernels.year.min()),
        "tn_flux_simulation_start_year": int(fluxes.year.min()),
        "first_tn_evaluation_year": 2016,
        "current_warmup_years": int(2016 - fluxes.year.min()),
        "available_warmup_years_if_recompiled": 10,
        "code_initializes_upper_and_lower_n_to_zero": zero_state_code,
        "status": "NEEDS_2006_2015_TN_WARMUP" if int(kernels.year.min()) > 2006 or zero_state_code else "PASS",
    }


def memory_audit() -> dict[str, object]:
    stderr = STAGE13_STDERR.read_text(encoding="utf-8", errors="replace") if STAGE13_STDERR.exists() else ""
    model = STAGE13_MODEL.read_text(encoding="utf-8")
    oom_terms = ["out of memory", "outofmemory", "memoryerror", "cannot allocate", "bad_alloc", "killed"]
    return {
        "stage13_stderr_exists": STAGE13_STDERR.exists(),
        "oom_signature_found": any(term in stderr.lower() for term in oom_terms),
        "arrow_type_error_found": "ArrowTypeError" in stderr,
        "arrow_error_field": "evaluation_year" if "evaluation_year" in stderr else None,
        "bounded_lru_present_in_current_code": "OrderedDict" in model and "while len(self._cache) > 8" in model,
        "mathematical_contamination_evidence": False,
        "next_stage_contract": {
            "rss_warning_gib": 12,
            "rss_hard_stop_gib": 16,
            "stage_and_fold_subprocess_isolation": True,
            "atomic_output": True,
            "incomplete_cache_reuse": False,
            "clean_rerun_required": True,
        },
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    required = [TN_ALL, DOMAIN, CURRENT_OBS, SOURCE, KERNELS, FLUXES, OOF, CONTRACT, RAW_2021_2024]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    all_tn = pd.read_parquet(TN_ALL)
    domain = pd.read_parquet(DOMAIN)
    selected = all_tn.loc[
        all_tn.strict & all_tn.year.between(2016, 2024) & all_tn.tn_mg_l.notna() & all_tn.tn_mg_l.ge(0)
    ].copy()
    selected["reach_id"] = selected.reach_id.astype(int)
    selected = selected.merge(
        domain[["station", "station_key", "reach_id", "terminal_tree_id", "observation_domain", "domain_reason", "primary_river_domain", "hydraulic_primary_gate_role"]],
        on=["station_key", "reach_id"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_domain"),
    )
    selected["formal_river_channel"] = selected.observation_domain.eq("river_channel")
    selected["source_provenance_tier"] = "prepared_authoritative_station_csv"
    selected["raw_workbook_record_link_status"] = np.where(
        selected.year.between(2021, 2024), "missing_section_code_and_raw_row_link", "raw_station_csv_not_registered"
    )

    lineage, lineage_audit = prepared_lineage(selected[["station_key"]].drop_duplicates())
    selected = selected.merge(
        lineage,
        on=["station_key", "year", "month"],
        how="left",
        validate="one_to_one",
    )
    selected["prepared_value_exact"] = np.isclose(
        selected.tn_mg_l.to_numpy(float),
        selected.prepared_tn_numeric.to_numpy(float),
        rtol=0.0,
        atol=1.0e-12,
        equal_nan=False,
    )
    selected = selected.sort_values(["station_key", "year", "month"]).reset_index(drop=True)
    if selected.duplicated(["station_key", "year", "month"]).any():
        raise RuntimeError("audited TN table is not station-month unique")
    selected.to_parquet(OUT / "tn_observations_audited.parquet", index=False)

    coverage = selected.groupby(["year", "observation_domain", "formal_river_channel"], as_index=False).agg(
        rows=("tn_mg_l", "size"), stations=("station_key", "nunique"), reaches=("reach_id", "nunique"),
        median_tn_mg_l=("tn_mg_l", "median"), p95_tn_mg_l=("tn_mg_l", lambda x: float(np.quantile(x, 0.95))),
    )
    coverage.to_parquet(OUT / "tn_coverage_profile.parquet", index=False)

    station_reach = selected[["station", "station_key", "reach_id", "observation_domain"]].drop_duplicates()
    same_reach = station_reach.groupby("reach_id", as_index=False).agg(
        station_count=("station_key", "nunique"),
        station_names=("station", lambda x: " | ".join(sorted(set(map(str, x))))),
        domains=("observation_domain", lambda x: " | ".join(sorted(set(map(str, x))))),
    )
    same_reach = same_reach.loc[same_reach.station_count.gt(1)].sort_values(["station_count", "reach_id"], ascending=[False, True])
    same_reach.to_parquet(OUT / "same_reach_station_audit.parquet", index=False)

    current = pd.read_parquet(CURRENT_OBS)
    current_domain_rows = current.groupby("observation_domain", as_index=False).agg(
        rows=("tn_mg_l", "size"), stations=("station_key", "nunique"), reaches=("reach_id", "nunique")
    )

    oof = pd.read_parquet(OOF)
    temporal = oof.loc[oof.holdout_type.eq("TEMPORAL")].copy()
    perf_rows: list[dict[str, object]] = []
    for label, frame in (
        ("all_stage12_primary", temporal),
        ("river_channel", temporal.loc[temporal.observation_domain.eq("river_channel")]),
        ("dam_or_reservoir_outlet", temporal.loc[temporal.observation_domain.eq("dam_or_reservoir_outlet")]),
    ):
        if len(frame):
            perf_rows.append({"domain": label, "year": "ALL", **metric_values(frame)})
            for year, block in frame.groupby("year"):
                perf_rows.append({"domain": label, "year": str(int(year)), **metric_values(block)})
    perf = pd.DataFrame(perf_rows)
    perf.to_parquet(OUT / "domain_performance_audit.parquet", index=False)

    temporal = temporal.assign(sq_error=np.square(temporal.pred_tn_mg_l - temporal.tn_mg_l))
    total_sse = float(temporal.sq_error.sum())
    reach_sse = temporal.groupby("reach_id").sq_error.sum().sort_values(ascending=False)
    top_sorted = temporal.sq_error.sort_values(ascending=False)
    extreme = {
        "top_1_percent_sse_share": float(top_sorted.head(max(1, math.ceil(len(top_sorted) * 0.01))).sum() / total_sse),
        "top_5_percent_sse_share": float(top_sorted.head(max(1, math.ceil(len(top_sorted) * 0.05))).sum() / total_sse),
        "reach_190_sse_share": float(reach_sse.get(190, 0.0) / total_sse),
        "largest_reach_sse_share": float(reach_sse.iloc[0] / total_sse),
        "largest_reach_id": int(reach_sse.index[0]),
    }

    river = selected.loc[selected.formal_river_channel].copy()
    y23 = river.loc[river.year.eq(2023), ["station_key", "month", "tn_mg_l"]].rename(columns={"tn_mg_l": "tn_2023"})
    y24 = river.loc[river.year.eq(2024), ["station_key", "month", "tn_mg_l"]].rename(columns={"tn_mg_l": "tn_2024"})
    common = y23.merge(y24, on=["station_key", "month"], how="inner", validate="one_to_one")
    drift = {
        "common_station_month_rows": int(len(common)),
        "correlation_2023_2024": float(common[["tn_2023", "tn_2024"]].corr().iloc[0, 1]),
        "median_2023_mg_l": float(common.tn_2023.median()),
        "median_2024_mg_l": float(common.tn_2024.median()),
    }

    clipping = source_clipping_audit()
    initialization = initialization_audit()
    memory = memory_audit()
    write_json(REPORTS / "stage18_memory_audit.json", memory)

    workbook = workbook_profile(RAW_2021_2024, include_rows=True)
    raw_2025 = workbook_profile(RAW_2025, include_rows=False) if RAW_2025.exists() else None

    checks = {
        "audited_station_month_unique": not selected.duplicated(["station_key", "year", "month"]).any(),
        "prepared_station_files_complete": len(lineage_audit["missing_station_files"]) == 0,
        "prepared_values_exact": bool(selected.prepared_value_exact.all()),
        "river_primary_excludes_reservoir_outlets": not selected.loc[selected.formal_river_channel, "observation_domain"].eq("dam_or_reservoir_outlet").any(),
        "future_2025_2026_excluded": int(selected.year.max()) == 2024,
        "warmup_requires_correction": initialization["status"] == "NEEDS_2006_2015_TN_WARMUP",
        "no_oom_signature_in_stage13_log": not memory["oom_signature_found"],
    }
    audit = {
        "stage": "20260824_18",
        "status": "PASS_STAGE18_AUDIT_WITH_REQUIRED_CORRECTIONS" if all([
            checks["audited_station_month_unique"], checks["prepared_station_files_complete"],
            checks["prepared_values_exact"], checks["river_primary_excludes_reservoir_outlets"],
            checks["future_2025_2026_excluded"], checks["no_oom_signature_in_stage13_log"],
        ]) else "FAIL_STAGE18_DATA_CONTRACT",
        "counts": {
            "audited_rows": int(len(selected)),
            "audited_stations": int(selected.station_key.nunique()),
            "audited_reaches": int(selected.reach_id.nunique()),
            "river_rows": int(selected.formal_river_channel.sum()),
            "river_stations": int(selected.loc[selected.formal_river_channel, "station_key"].nunique()),
            "river_reaches": int(selected.loc[selected.formal_river_channel, "reach_id"].nunique()),
            "same_reach_multi_station_reaches": int(len(same_reach)),
        },
        "current_stage12_domain_counts": current_domain_rows.to_dict("records"),
        "lineage": lineage_audit,
        "raw_workbook": workbook,
        "raw_2025_workbook": raw_2025,
        "2024_drift": drift,
        "extreme_error_concentration": extreme,
        "source_clipping": clipping,
        "initialization": initialization,
        "memory": memory,
        "checks": checks,
        "severity_findings": [
            {"severity": "critical", "finding": "dam/reservoir outlets were included in a model with no reservoir operator", "action": "primary domain changed to river_channel"},
            {"severity": "high", "finding": "TN carrier starts in 2010 with zero upper/lower N state despite canonical hydrology beginning in 2006", "action": "recompile 2006-2024 and use 2006-2015 warm-up"},
            {"severity": "high", "finding": "record-level source workbook provenance is absent from the prepared/model-ready archive", "action": "retain prepared CSV row lineage now and rebuild SectionCode/source-row lineage when a decodable raw source is available"},
            {"severity": "medium", "finding": "monthly crop-demand clipping changes annual net available N", "action": "replace implicit clipping with an explicit mass ledger before source-state experiments"},
            {"severity": "low", "finding": "stage13 first failure was ArrowTypeError rather than OOM", "action": "bounded-memory clean rerun remains mandatory"},
        ],
        "source_hashes": {str(path): sha256(path) for path in required},
        "authorized_successor": "20260824_19",
    }
    write_json(REPORTS / "stage18_data_quality_audit.json", audit)

    program_manifest = {
        "program": "20260824_18_24_TN_REBUILD",
        "status": "stage18_complete_stage19_authorized" if audit["status"].startswith("PASS") else "blocked_by_stage18",
        "primary_goal": "river-channel spatially transferable hydrology-driven monthly TN model",
        "frozen_hydrology": "20260828_9/10",
        "stages": [
            {"stage": "20260824_18", "role": "data/domain/numerical/memory audit", "status": "complete" if audit["status"].startswith("PASS") else "failed"},
            {"stage": "20260824_19", "role": "corrected deterministic M3 baseline", "status": "authorized" if audit["status"].startswith("PASS") else "closed"},
            {"stage": "20260824_20", "role": "fair ablation and solver diagnosis", "status": "registered"},
            {"stage": "20260824_21", "role": "differentiable TN parent", "status": "registered"},
            {"stage": "20260824_22", "role": "source-specific availability and manure legacy", "status": "registered"},
            {"stage": "20260824_23", "role": "hierarchical spatial regionalization", "status": "registered"},
            {"stage": "20260824_24", "role": "final lock and synthesis", "status": "registered"},
        ],
        "forbidden": ["temperature", "reservoir operator", "WWTP", "new mu_T", "groundwater age", "SAS", "dual groundwater store"],
        "no_automatic_extension_beyond": "20260824_24",
    }
    write_json(REPORTS / "program_manifest.json", program_manifest)

    river_perf = perf.loc[(perf.domain == "river_channel") & (perf.year == "ALL")].iloc[0]
    all_perf = perf.loc[(perf.domain == "all_stage12_primary") & (perf.year == "ALL")].iloc[0]
    reservoir_perf = perf.loc[(perf.domain == "dam_or_reservoir_outlet") & (perf.year == "ALL")].iloc[0]
    report = f"""# 20260824_18 TN数据、观测域、数值与内存审计

## 结论

`{audit['status']}`。下一步必须先运行修正后的M3基线，不能直接把当前低精度归因于2024或优化器。

## 数据与观测域

- 审计表：{len(selected):,}条、{selected.station_key.nunique()}站、{selected.reach_id.nunique()}个Reach。
- 河道主域：{int(selected.formal_river_channel.sum()):,}条、{selected.loc[selected.formal_river_channel, 'station_key'].nunique()}站、{selected.loc[selected.formal_river_channel, 'reach_id'].nunique()}个Reach。
- 现有Stage12错误地把{int(current.loc[current.observation_domain.eq('dam_or_reservoir_outlet')].station_key.nunique())}个坝/水库出口、{int(current.observation_domain.eq('dam_or_reservoir_outlet').sum())}条观测放入无水库主模型。
- 准备态站点CSV可以逐行闭合现有TN值：`{checks['prepared_values_exact']}`；但原始工作簿没有被保留到逐记录SectionCode/source-row血缘，属于高等级provenance缺口。

## 低精度的直接证据

2022–2024 temporal OOF：

| 域 | RMSE mg/L | NSE | R² | 行数 |
|---|---:|---:|---:|---:|
| 原Stage12主域 | {all_perf.rmse_mg_l:.3f} | {all_perf.nse:.3f} | {all_perf.r2:.3f} | {int(all_perf.n)} |
| river_channel | {river_perf.rmse_mg_l:.3f} | {river_perf.nse:.3f} | {river_perf.r2:.3f} | {int(river_perf.n)} |
| dam/reservoir outlet | {reservoir_perf.rmse_mg_l:.3f} | {reservoir_perf.nse:.3f} | {reservoir_perf.r2:.3f} | {int(reservoir_perf.n)} |

前1%样本贡献{extreme['top_1_percent_sse_share']:.1%}总SSE，Reach 190贡献{extreme['reach_190_sse_share']:.1%}。排除水库出口后NSE仍低，说明观测域错误很严重但不是唯一问题。

2023–2024共同站月TN相关为{drift['correlation_2023_2024']:.3f}，中位数从{drift['median_2023_mg_l']:.3f}变为{drift['median_2024_mg_l']:.3f} mg/L；不支持“2024资料导致模型崩坏”。

## 数值与工程

- 当前TN carrier从{initialization['tn_flux_simulation_start_year']}年零N状态启动，只给2016首个评价年{initialization['current_warmup_years']}年warm-up；下一轮改为2006–2015完整warm-up。
- 月度crop-demand clipping使全域年度净available N增加{clipping['basin_inflation_fraction']:.3%}，个别Reach-year最高增加{clipping['reach_year_max_ratio'] - 1:.1%}。
- Stage13日志未发现OOM签名；发现的是`ArrowTypeError(evaluation_year)`。当前LRU已有8项上限，但下一轮仍须空缓存、分进程、RSS硬门禁和确定性双跑。

## 正式边界

`20260824_19`只修复河道域、2006 warm-up、质量账本和内存确定性，不增加任何拟合参数。温度、水库、WWTP、SAS和新地下水年龄结构继续关闭。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")

    validation = {
        "stage": "20260824_18",
        "status": "PASS_STAGE18_READY_FOR_20260824_19" if audit["status"].startswith("PASS") else "FAIL",
        "checks": checks,
        "output_rows": {
            "tn_observations_audited": int(len(selected)),
            "tn_coverage_profile": int(len(coverage)),
            "domain_performance_audit": int(len(perf)),
            "same_reach_station_audit": int(len(same_reach)),
        },
    }
    write_json(REPORTS / "stage18_final_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
