from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RUN_DIR = Path(__file__).resolve().parents[1]
SPARROW_ROOT = Path(r"E:\SPARROW")
BASELINE = SPARROW_ROOT / "5_Test" / "20260805_2"
TOPOLOGY_ROOT = SPARROW_ROOT / "0_reach_topology"
EXPECTED_PYTHON = Path(r"D:\ProgramData\anaconda3\envs\sparrow\python.exe")
DEVELOPMENT_START = 2006
DEVELOPMENT_END = 2018
EXPECTED_FOLDS = {
    "fit_2006_2011_eval_2012_2013": (2012, 2013),
    "fit_2006_2013_eval_2014_2015": (2014, 2015),
    "fit_2006_2015_eval_2016_2018": (2016, 2018),
}

SOURCE_PATHS = {
    "canonical_oof": BASELINE / "outputs" / "q72_three_fold_oof_predictions.parquet",
    "noncanonical_oof_csv": BASELINE / "outputs" / "q72_three_fold_oof_predictions.csv",
    "input_panel": BASELINE / "inputs" / "indata.parquet",
    "topology_edges": BASELINE / "inputs" / "topology" / "topology_edges.csv",
    "q72_code": BASELINE / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py",
    "grid_mapping": TOPOLOGY_ROOT / "data" / "processed" / "rainfall2_prb" / "chm_pre_v2_grid_to_reach_mapping.csv",
    "monthly_reach_precip": TOPOLOGY_ROOT / "data" / "processed" / "rainfall2_prb" / "chm_pre_v2_monthly_by_reach_2006_2022.parquet",
}
FOLD_SOURCE_PATHS = {
    fold: BASELINE / "reports" / "q72_baseline" / "blocked_folds" / fold / "evaluation_predictions.csv"
    for fold in EXPECTED_FOLDS
}
DAILY_DIR = TOPOLOGY_ROOT / "data" / "raw" / "rainfall_2" / "CHM_PRE V2" / "daily"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_guard() -> dict[str, object]:
    actual = Path(sys.executable).resolve()
    expected = EXPECTED_PYTHON.resolve()
    exact = str(actual).casefold() == str(expected).casefold()
    if not exact:
        raise RuntimeError(f"Gate 0 requires {expected}; got {actual}")
    packages = {}
    for name in ["numpy", "pandas", "scipy", "xarray", "pyarrow", "h5py", "scikit-learn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "sys_executable": str(actual),
        "expected_executable": str(expected),
        "runtime_exact_conda_sparrow": exact,
        "python_version": sys.version,
        "packages": packages,
    }
    write_json(RUN_DIR / "logs" / "runtime_identity.json", payload)
    return payload


def require_sources() -> list[Path]:
    paths = list(SOURCE_PATHS.values())
    paths.extend(FOLD_SOURCE_PATHS.values())
    paths.extend(DAILY_DIR / f"CHM_PRE_V2_daily_{year}.nc" for year in range(DEVELOPMENT_START, DEVELOPMENT_END + 1))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Gate 0 sources:\n" + "\n".join(str(p) for p in missing))
    return paths


def snapshot_inputs() -> None:
    target = RUN_DIR / "inputs" / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    copies = {
        SOURCE_PATHS["canonical_oof"]: target / "canonical_q72_oof.parquet",
        SOURCE_PATHS["input_panel"]: target / "q72_input_panel_2006_2018.parquet",
        SOURCE_PATHS["topology_edges"]: target / "topology_edges.csv",
        SOURCE_PATHS["grid_mapping"]: target / "chm_pre_v2_grid_to_reach_mapping.csv",
        SOURCE_PATHS["monthly_reach_precip"]: target / "chm_pre_v2_monthly_by_reach_2006_2018.parquet",
        SOURCE_PATHS["q72_code"]: RUN_DIR / "src" / "reference" / SOURCE_PATHS["q72_code"].name,
    }
    # Files containing post-2018 rows are filtered, never copied wholesale.
    shutil.copy2(SOURCE_PATHS["canonical_oof"], copies[SOURCE_PATHS["canonical_oof"]])
    indata = pd.read_parquet(SOURCE_PATHS["input_panel"], filters=[("year", ">=", DEVELOPMENT_START), ("year", "<=", DEVELOPMENT_END)])
    indata.to_parquet(copies[SOURCE_PATHS["input_panel"]], index=False)
    shutil.copy2(SOURCE_PATHS["topology_edges"], copies[SOURCE_PATHS["topology_edges"]])
    shutil.copy2(SOURCE_PATHS["grid_mapping"], copies[SOURCE_PATHS["grid_mapping"]])
    monthly = pd.read_parquet(
        SOURCE_PATHS["monthly_reach_precip"],
        filters=[("year", ">=", DEVELOPMENT_START), ("year", "<=", DEVELOPMENT_END)],
    )
    monthly.to_parquet(copies[SOURCE_PATHS["monthly_reach_precip"]], index=False)
    copies[SOURCE_PATHS["q72_code"]].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PATHS["q72_code"], copies[SOURCE_PATHS["q72_code"]])
    fold_target = target / "fold_sources"
    fold_target.mkdir(parents=True, exist_ok=True)
    for fold, path in FOLD_SOURCE_PATHS.items():
        shutil.copy2(path, fold_target / f"{fold}.csv")


def audit_oof() -> dict[str, object]:
    canonical = pd.read_parquet(SOURCE_PATHS["canonical_oof"])
    mirror = pd.read_csv(SOURCE_PATHS["noncanonical_oof_csv"])
    duplicate_keys = int(canonical.duplicated(["station_name", "year", "month"]).sum())
    fold_rows = []
    fold_contract = True
    for fold, (year_min, year_max) in EXPECTED_FOLDS.items():
        part = canonical[canonical["fold_id"] == fold]
        ok = bool(len(part) > 0 and part["year"].min() == year_min and part["year"].max() == year_max)
        fold_contract &= ok
        fold_rows.append(
            {
                "fold_id": fold,
                "rows": int(len(part)),
                "stations": int(part["station_name"].nunique()),
                "year_min": int(part["year"].min()) if len(part) else None,
                "year_max": int(part["year"].max()) if len(part) else None,
                "contract_passed": ok,
            }
        )
    pd.DataFrame(fold_rows).to_csv(RUN_DIR / "reports" / "oof_fold_contract.csv", index=False, encoding="utf-8-sig")
    canonical_keys = canonical[["station_name", "year", "month"]].sort_values(["station_name", "year", "month"]).reset_index(drop=True)
    mirror_keys = mirror[["station_name", "year", "month"]].sort_values(["station_name", "year", "month"]).reset_index(drop=True)
    mirror_matches = bool(len(canonical) == len(mirror) and canonical_keys.equals(mirror_keys))
    reconstructed = pd.concat([pd.read_csv(FOLD_SOURCE_PATHS[fold]) for fold in EXPECTED_FOLDS], ignore_index=True)
    reconstructed = reconstructed.rename(
        columns={"q_site": "station_name", "comid": "reach_id", "actual": "observed_cfs", "predict": "predicted_cfs"}
    )
    reconstructed["observed_m3s"] = reconstructed["observed_cfs"] / 35.3146667
    reconstructed["predicted_m3s"] = reconstructed["predicted_cfs"] / 35.3146667
    sort_cols = ["fold_id", "station_name", "year", "month"]
    canonical_sorted = canonical.sort_values(sort_cols).reset_index(drop=True)
    reconstructed_sorted = reconstructed.sort_values(sort_cols).reset_index(drop=True)
    same_columns = set(canonical_sorted.columns) == set(reconstructed_sorted.columns)
    numeric_max_abs_difference = 0.0
    nonnumeric_equal = True
    if len(canonical_sorted) == len(reconstructed_sorted) and same_columns:
        reconstructed_sorted = reconstructed_sorted[canonical_sorted.columns]
        for column in canonical_sorted.columns:
            left = canonical_sorted[column]
            right = reconstructed_sorted[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                both = left.notna() & right.notna()
                if bool(both.any()):
                    numeric_max_abs_difference = max(
                        numeric_max_abs_difference,
                        float((left[both].astype(float) - right[both].astype(float)).abs().max()),
                    )
                if not left.isna().equals(right.isna()):
                    nonnumeric_equal = False
            elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
                nonnumeric_equal = False
    reconstructed_matches = bool(
        len(canonical_sorted) == len(reconstructed_sorted)
        and same_columns
        and nonnumeric_equal
        and numeric_max_abs_difference <= 1e-12
    )
    payload = {
        "canonical_format": "parquet",
        "canonical_selection_basis": "20260805_2 validation.json and reports/model_audit/gate.json both register 8738 OOF rows",
        "canonical_rows": int(len(canonical)),
        "canonical_stations": int(canonical["station_name"].nunique()),
        "canonical_key_duplicates": duplicate_keys,
        "canonical_year_min": int(canonical["year"].min()),
        "canonical_year_max": int(canonical["year"].max()),
        "canonical_positive_finite": bool(
            np.isfinite(canonical["observed_cfs"]).all()
            and np.isfinite(canonical["predicted_cfs"]).all()
            and (canonical["observed_cfs"] > 0).all()
            and (canonical["predicted_cfs"] > 0).all()
        ),
        "fold_contract_passed": fold_contract,
        "noncanonical_csv_rows": int(len(mirror)),
        "noncanonical_csv_stations": int(mirror["station_name"].nunique()),
        "noncanonical_csv_matches_canonical": mirror_matches,
        "noncanonical_csv_disposition": "INTERNALLY_INCONSISTENT_NONCANONICAL_CSV; ROOT_CAUSE_NOT_ESTABLISHED; DO_NOT_USE" if not mirror_matches else "MATCHES_CANONICAL",
        "reconstructed_from_fold_sources_rows": int(len(reconstructed)),
        "reconstructed_from_fold_sources_matches_canonical": reconstructed_matches,
        "reconstructed_numeric_max_abs_difference": numeric_max_abs_difference,
        "fold_source_sha256": {fold: sha256_file(path) for fold, path in FOLD_SOURCE_PATHS.items()},
    }
    write_json(RUN_DIR / "reports" / "oof_contract_audit.json", payload)
    return payload


def audit_area_closure() -> dict[str, object]:
    edges = pd.read_csv(SOURCE_PATHS["topology_edges"])
    panel = pd.read_parquet(
        SOURCE_PATHS["input_panel"],
        filters=[("year", "==", DEVELOPMENT_START), ("month", "==", 1)],
        columns=["comid", "IncAreaKm2", "CumAreaKm2"],
    ).drop_duplicates("comid")
    inc = dict(zip(panel["comid"].astype(int), panel["IncAreaKm2"].astype(float)))
    cum = dict(zip(panel["comid"].astype(int), panel["CumAreaKm2"].astype(float)))
    upstream = {}
    for row in edges.itertuples(index=False):
        value = row.upstream_reaches
        upstream[int(row.reach_id)] = [] if pd.isna(value) else [int(float(x)) for x in str(value).split(",")]
    memo: dict[int, float] = {}
    visiting: set[int] = set()

    def sum_incremental(reach: int) -> float:
        if reach in memo:
            return memo[reach]
        if reach in visiting:
            raise RuntimeError(f"Topology cycle encountered at reach {reach}")
        visiting.add(reach)
        value = inc[reach] + sum(sum_incremental(up) for up in upstream.get(reach, []))
        visiting.remove(reach)
        memo[reach] = value
        return value

    rows = []
    for reach in sorted(upstream):
        total = sum_incremental(reach)
        expected = cum[reach]
        diff = total - expected
        rows.append(
            {
                "reach_id": reach,
                "sum_upstream_incremental_area_km2": total,
                "registered_cumulative_area_km2": expected,
                "difference_km2": diff,
                "relative_difference": diff / expected if expected else np.nan,
                "passed_1pct": bool(abs(diff) <= 0.01 * max(expected, 1.0)),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(RUN_DIR / "reports" / "area_closure_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "reaches": int(len(audit)),
        "topology_cycle_free": True,
        "all_reaches_close_within_1pct": bool(audit["passed_1pct"].all()),
        "max_absolute_difference_km2": float(audit["difference_km2"].abs().max()),
        "max_absolute_relative_difference": float(audit["relative_difference"].abs().max()),
    }


def build_daily_reach_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    mapping = pd.read_csv(SOURCE_PATHS["grid_mapping"]).reset_index(drop=True)
    reach_groups = [(int(reach), group.index.to_numpy(), group["weight"].to_numpy(dtype=float)) for reach, group in mapping.groupby("reach_id", sort=True)]
    ilat = xr.DataArray(mapping["ilat"].to_numpy(dtype=int), dims="point")
    ilon = xr.DataArray(mapping["ilon"].to_numpy(dtype=int), dims="point")
    frames = []
    source_rows = []
    for year in range(DEVELOPMENT_START, DEVELOPMENT_END + 1):
        path = DAILY_DIR / f"CHM_PRE_V2_daily_{year}.nc"
        with xr.open_dataset(path) as ds:
            if ds["prec"].attrs.get("units") != "mm/day":
                raise RuntimeError(f"Unexpected precipitation unit in {path}: {ds['prec'].attrs.get('units')}")
            values = ds["prec"].isel(lat=ilat, lon=ilon).values.astype(float)
            dates = pd.to_datetime(ds["time"].values)
        expected_days = 366 if monthrange(year, 2)[1] == 29 else 365
        if len(dates) != expected_days or dates.min().year != year or dates.max().year != year:
            raise RuntimeError(f"Incomplete daily time axis for {year}")
        year_frames = []
        for reach, indices, weights in reach_groups:
            part = values[:, indices]
            valid = np.isfinite(part)
            denominator = np.where(valid, weights, 0.0).sum(axis=1)
            if (denominator <= 0).any():
                raise RuntimeError(f"No valid CHM_PRE cell for reach {reach} in {year}")
            ppt = np.where(valid, part * weights, 0.0).sum(axis=1) / denominator
            year_frames.append(
                pd.DataFrame(
                    {
                        "reach_id": reach,
                        "date": dates,
                        "PPT_daily_mm": ppt,
                        "n_grid_cells": len(indices),
                        "valid_grid_fraction": valid.mean(axis=1),
                    }
                )
            )
        frames.append(pd.concat(year_frames, ignore_index=True))
        source_rows.append(
            {
                "year": year,
                "path": str(path),
                "bytes": path.stat().st_size,
                "days": len(dates),
                "unit": "mm/day",
            }
        )
    daily = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    daily.to_parquet(RUN_DIR / "inputs" / "derived" / "chm_pre_v2_daily_by_reach_2006_2018.parquet", index=False)
    pd.DataFrame(source_rows).to_csv(RUN_DIR / "reports" / "daily_source_inventory.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(daily)),
        "reaches": int(daily["reach_id"].nunique()),
        "days": int(daily["date"].nunique()),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "key_duplicates": int(daily.duplicated(["reach_id", "date"]).sum()),
        "minimum_valid_grid_fraction": float(daily["valid_grid_fraction"].min()),
        "all_ppt_finite_nonnegative": bool(np.isfinite(daily["PPT_daily_mm"]).all() and (daily["PPT_daily_mm"] >= 0).all()),
    }
    return daily, summary


def audit_precipitation(daily: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    monthly = pd.read_parquet(
        SOURCE_PATHS["monthly_reach_precip"],
        filters=[("year", ">=", DEVELOPMENT_START), ("year", "<=", DEVELOPMENT_END)],
    )
    aggregate = daily.assign(year=daily["date"].dt.year, month=daily["date"].dt.month).groupby(
        ["reach_id", "year", "month"], as_index=False
    ).agg(PPT_daily_sum_mm=("PPT_daily_mm", "sum"), n_days=("date", "size"))
    mass = aggregate.merge(monthly[["reach_id", "year", "month", "PPT_rainfall2_mm"]], on=["reach_id", "year", "month"], how="outer", validate="one_to_one")
    mass["difference_mm"] = mass["PPT_daily_sum_mm"] - mass["PPT_rainfall2_mm"]
    mass["tolerance_mm"] = np.maximum(0.1, 0.001 * mass["PPT_rainfall2_mm"].abs())
    mass["passed"] = mass["difference_mm"].abs() <= mass["tolerance_mm"]
    mass.to_csv(RUN_DIR / "reports" / "daily_monthly_mass_audit.csv", index=False, encoding="utf-8-sig")

    panel = pd.read_parquet(
        SOURCE_PATHS["input_panel"],
        filters=[("year", ">=", DEVELOPMENT_START), ("year", "<=", DEVELOPMENT_END)],
        columns=["comid", "year", "month", "PPT"],
    ).rename(columns={"comid": "reach_id", "PPT": "PPT_q72_mm"})
    baseline = panel.merge(monthly[["reach_id", "year", "month", "PPT_rainfall2_mm"]], on=["reach_id", "year", "month"], how="outer", validate="one_to_one")
    baseline["difference_mm"] = baseline["PPT_q72_mm"] - baseline["PPT_rainfall2_mm"]
    baseline["tolerance_mm"] = np.maximum(0.1, 0.001 * baseline["PPT_rainfall2_mm"].abs())
    baseline["passed"] = baseline["difference_mm"].abs() <= baseline["tolerance_mm"]
    baseline.to_csv(RUN_DIR / "reports" / "q72_monthly_forcing_identity_audit.csv", index=False, encoding="utf-8-sig")
    return (
        {
            "rows": int(len(mass)),
            "all_reach_months_within_tolerance": bool(mass["passed"].all()),
            "max_absolute_difference_mm": float(mass["difference_mm"].abs().max()),
            "max_absolute_relative_difference": float((mass["difference_mm"].abs() / mass["PPT_rainfall2_mm"].abs().clip(lower=1e-12)).max()),
        },
        {
            "rows": int(len(baseline)),
            "q72_ppt_matches_processed_monthly": bool(baseline["passed"].all()),
            "max_absolute_difference_mm": float(baseline["difference_mm"].abs().max()),
        },
    )


def build_manifest(paths: list[Path]) -> dict[str, object]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": "read_only_source",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RUN_DIR / "inputs_manifest" / "source_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {"sources": rows, "source_count": len(rows)}
    write_json(RUN_DIR / "inputs_manifest" / "source_manifest.json", payload)
    return payload


def write_readme(validation: dict[str, object]) -> None:
    oof = validation["oof_contract"]
    daily = validation["daily_panel"]
    mass = validation["daily_monthly_mass"]
    area = validation["area_closure"]
    lines = [
        "# 20260807_5 Gate 0：环境、OOF、日—月降水与面积闭合合同",
        "",
        "## 本目录回答的原子问题",
        "",
        "> 冻结Q72基线、canonical OOF、拓扑和CHM_PRE日/月forcing是否满足后续空间与时间诊断的基础合同？",
        "",
        "## 结论",
        "",
        f"- Gate：`{validation['decision']}`",
        f"- 正式环境：`{validation['runtime']['sys_executable']}`",
        f"- canonical OOF：{oof['canonical_rows']}行、{oof['canonical_stations']}站、键重复{oof['canonical_key_duplicates']}条。",
        f"- 日降水面板：{daily['rows']}行、{daily['reaches']}个Reach、{daily['days']}天。",
        f"- 日值聚合与官方月值最大绝对差：{mass['max_absolute_difference_mm']:.3e} mm。",
        f"- 上游增量面积与CumArea最大相对差：{area['max_absolute_relative_difference']:.3e}。",
        "",
        "## 重要发现",
        "",
        f"`q72_three_fold_oof_predictions.parquet`与三个fold原始预测重建结果及基线验证文件一致，为{oof['canonical_rows']}行canonical OOF；同名CSV只有{oof['noncanonical_csv_rows']}行，`noncanonical_csv_matches_canonical={str(oof['noncanonical_csv_matches_canonical']).lower()}`。其根因未确定，标记为内部不一致的非canonical CSV，后续严格禁止读取。",
        "",
        "## 输入与边界",
        "",
        "- 只处理2006—2018日/月forcing；未读取2019—2022用于任何统计或模型选择。",
        "- 原始数据、基线和生产代码只读；全部新增结果位于本目录。",
        "- 本目录生成230个Reach的2006—2018日降水面板，为后续上游空间聚合提供冻结输入。",
        "",
        "## 复跑",
        "",
        "```powershell",
        "& 'D:\\ProgramData\\anaconda3\\envs\\sparrow\\python.exe' scripts\\run_gate0.py",
        "```",
        "",
        "## 主要证据",
        "",
        "- `validation.json`：全部Gate 0机器判定；",
        "- `gate.json`：门禁状态和下一步；",
        "- `reports/daily_monthly_mass_audit.csv`：逐Reach—月质量守恒；",
        "- `reports/q72_monthly_forcing_identity_audit.csv`：Q72 PPT与CHM_PRE月forcing一致性；",
        "- `reports/area_closure_audit.csv`：拓扑面积闭合；",
        "- `reports/oof_contract_audit.json`：fold来源重建、canonical OOF选择与非canonical CSV不一致；",
        "- `inputs/derived/chm_pre_v2_daily_by_reach_2006_2018.parquet`：冻结日降水面板。",
        "",
        "## 下一门禁",
        "",
        "Gate 0通过后，下一动态目录只回答`FORCING_SEMANTIC_AUDIT`，沿Q72计算图核实PPT、面积乘法、状态递推和流量单位链。",
    ]
    (RUN_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for relative in ["inputs/derived", "inputs/source_snapshot", "src/reference", "reports", "logs", "inputs_manifest"]:
        (RUN_DIR / relative).mkdir(parents=True, exist_ok=True)
    runtime = runtime_guard()
    sources = require_sources()
    snapshot_inputs()
    oof = audit_oof()
    area = audit_area_closure()
    daily, daily_summary = build_daily_reach_panel()
    mass, forcing_identity = audit_precipitation(daily)
    manifest = build_manifest(sources)
    checks = {
        "runtime_exact_conda_sparrow": runtime["runtime_exact_conda_sparrow"],
        "canonical_oof_8738_rows": oof["canonical_rows"] == 8738,
        "canonical_oof_110_stations": oof["canonical_stations"] == 110,
        "canonical_oof_key_unique": oof["canonical_key_duplicates"] == 0,
        "canonical_oof_fold_contract": oof["fold_contract_passed"],
        "canonical_oof_positive_finite": oof["canonical_positive_finite"],
        "fold_sources_reconstruct_canonical_oof": oof["reconstructed_from_fold_sources_matches_canonical"],
        "daily_panel_key_unique": daily_summary["key_duplicates"] == 0,
        "daily_panel_complete_2006_2018": daily_summary["date_min"] == "2006-01-01" and daily_summary["date_max"] == "2018-12-31",
        "daily_panel_finite_nonnegative": daily_summary["all_ppt_finite_nonnegative"],
        "daily_monthly_mass_within_tolerance": mass["all_reach_months_within_tolerance"],
        "q72_monthly_ppt_matches_chm_pre": forcing_identity["q72_ppt_matches_processed_monthly"],
        "topology_cycle_free": area["topology_cycle_free"],
        "area_closure_within_1pct": area["all_reaches_close_within_1pct"],
    }
    passed = bool(all(checks.values()))
    decision = "GATE_0_PASS_READY_FOR_FORCING_SEMANTIC_AUDIT" if passed else "INPUT_OR_MASS_CONTRACT_BLOCKED"
    validation = {
        "run_id": "20260807_5",
        "atomic_question": "Do the frozen Q72 baseline, canonical OOF, topology, and daily/monthly CHM_PRE forcing satisfy the pre-analysis contract?",
        "runtime": runtime,
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "check_count": len(checks),
        "passed": passed,
        "decision": decision,
        "oof_contract": oof,
        "daily_panel": daily_summary,
        "daily_monthly_mass": mass,
        "q72_monthly_forcing_identity": forcing_identity,
        "area_closure": area,
        "manifest_source_count": manifest["source_count"],
        "blind_period_policy": "2019-2022 not read for statistics, feature selection, or model selection",
    }
    write_json(RUN_DIR / "validation.json", validation)
    write_json(
        RUN_DIR / "gate.json",
        {
            "gate": "GATE_0",
            "passed": passed,
            "decision": decision,
            "next_gate": "GATE_S_1_FORCING_SEMANTIC_AUDIT" if passed else None,
            "nonblocking_warning": "The baseline CSV is internally inconsistent and noncanonical; root cause is not established. Fold sources reconstruct the canonical parquet exactly, and the CSV is forbidden.",
        },
    )
    write_readme(validation)
    print(json.dumps({"passed": passed, "decision": decision, "checks": checks}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
