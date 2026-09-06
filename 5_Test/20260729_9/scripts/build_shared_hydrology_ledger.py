from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
INPUTS = RUN / "inputs"
MANIFEST_DIR = RUN / "inputs_manifest"
REPORT_DIR = RUN / "reports" / "data_readiness_gate"
CONTRACT_PATH = RUN / "config" / "data_contract.json"

SOURCE_RUN = ROOT / "5_Test" / "20260728_30"
INDATA = SOURCE_RUN / "inputs" / "indata.parquet"
POLICY = SOURCE_RUN / "inputs" / "source_metadata" / "station_screening_policy.csv"
STATION_MATCH = SOURCE_RUN / "inputs" / "source_metadata" / "station_reach_match_fixed.csv"
REACH_SUMMARY = ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv"
TOPOLOGY = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
SOIL = (
    ROOT
    / "0_reach_topology"
    / "results"
    / "tables"
    / "reach_catchment_soil_storage_eff.csv"
)
BEDROCK_TABLE = INPUTS / "reach_depth_to_bedrock.csv"
BEDROCK_RASTER = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "processed"
    / "soil_prb"
    / "depth_to_bedrock_prb_buffer_100m.tif"
)
HSWUD = ROOT / "5_Test" / "20260608_2" / "inputs" / "hswud_reach_monthly.csv"
TARGET_PLAN = (
    ROOT
    / "5_Test"
    / "20260729_8"
    / "SPARROW_Q72_Q78_coherent_rebuild_and_terminal_test_plan.md"
)

CFS_TO_M3_S = 0.028316846592
FIXED_EXCLUDED = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
PROTECTED = {"石角站"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def month_seconds(year: pd.Series, month: pd.Series) -> np.ndarray:
    start = pd.to_datetime(
        {"year": year.astype(int), "month": month.astype(int), "day": 1}
    )
    next_month = start + pd.offsets.MonthBegin(1)
    return (next_month - start).dt.total_seconds().to_numpy()


def write_parquet_with_hash(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_parquet(path, index=False)
    return file_record(path, "ledger_product", "derived")


def main() -> None:
    for folder in (INPUTS, MANIFEST_DIR, REPORT_DIR, RUN / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    start_year = int(contract["development_start_year"])
    end_year = int(contract["development_end_year"])
    expected_reaches = int(contract["expected_reaches"])
    expected_months = int(contract["expected_months_per_reach"])

    # Parquet filters prevent the locked 2019-2022 rows from entering process memory.
    panel = pd.read_parquet(
        INDATA,
        filters=[[("year", ">=", start_year), ("year", "<=", end_year)]],
    )
    panel["reach_id"] = panel["comid"].astype(int)
    require(
        not panel.duplicated(["reach_id", "year", "month"]).any(),
        "The source panel is not unique by reach-year-month",
    )
    require(panel["reach_id"].nunique() == expected_reaches, "Reach count mismatch")
    require(
        panel.groupby("reach_id").size().eq(expected_months).all(),
        "Development months are incomplete for one or more reaches",
    )
    require(panel["year"].max() == end_year, "Locked years leaked into development data")

    summary = pd.read_csv(REACH_SUMMARY)
    topology = pd.read_csv(TOPOLOGY)
    soil = pd.read_csv(SOIL)
    bedrock = pd.read_csv(BEDROCK_TABLE, encoding="utf-8-sig")
    for table, name in [
        (summary, "reach summary"),
        (topology, "topology"),
        (soil, "soil"),
        (bedrock, "bedrock"),
    ]:
        require(
            len(table) == expected_reaches and table["reach_id"].nunique() == expected_reaches,
            f"{name} does not contain exactly {expected_reaches} reaches",
        )

    # Legacy slope is an all-zero placeholder. Do not carry it as a physical zero.
    slope_audit = (
        panel.groupby("reach_id", as_index=False)["SLOPE"]
        .agg(["min", "max", "nunique"])
        .reset_index()
    )
    slope_placeholder = bool(
        panel["SLOPE"].notna().all() and np.allclose(panel["SLOPE"].to_numpy(), 0.0)
    )
    require(slope_placeholder, "Legacy slope state changed; this gate must be reviewed")

    static = (
        summary.merge(
            topology[
                [
                    "reach_id",
                    "fnode",
                    "tnode",
                    "upstream_reaches",
                    "downstream_reach",
                    "hydseq",
                    "frac",
                    "iftran",
                ]
            ],
            on="reach_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            soil[["reach_id", "soil_storage_eff_mm", "valid_soilgrid_cells"]],
            on="reach_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            bedrock[
                ["reach_id", "depth_to_bedrock_m", "valid_bedrock_cells"]
            ],
            on="reach_id",
            how="left",
            validate="one_to_one",
        )
    )
    static["catchment_area_km2"] = static["inc_area_km2"]
    static["upstream_area_km2"] = static["tot_area_km2"]
    static["slope_m_m"] = np.nan
    static["slope_status"] = "missing_requires_dem_rebuild"
    static["reservoir_storage_capacity_m3"] = np.nan
    static["reservoir_commission_year"] = pd.Series(
        pd.array([pd.NA] * len(static), dtype="Int64")
    )
    static["reservoir_status"] = "missing"
    static = static.sort_values("reach_id").reset_index(drop=True)
    require(static["soil_storage_eff_mm"].notna().all(), "Soil storage is incomplete")
    require(static["depth_to_bedrock_m"].notna().all(), "Bedrock depth is incomplete")
    require(static["catchment_area_km2"].gt(0).all(), "Catchment area is non-positive")

    forcing = panel[
        ["reach_id", "year", "month", "PPT", "PET", "AET"]
    ].rename(
        columns={
            "PPT": "P_mm",
            "PET": "PET_mm",
            "AET": "AET_diagnostic_mm",
        }
    )
    forcing["P_status"] = "observed"
    forcing["PET_status"] = "observed"
    forcing["AET_status"] = "diagnostic_only"
    forcing = forcing.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    require(
        forcing[["P_mm", "PET_mm", "AET_diagnostic_mm"]].notna().all().all(),
        "Forcing contains missing values",
    )
    require(forcing["P_mm"].ge(0).all(), "Negative precipitation detected")
    require(forcing["PET_mm"].gt(0).all(), "Non-positive PET detected")

    observations = (
        panel.loc[
            panel["Q_obsv_cfs"].notna() & panel["q_site"].notna(),
            [
                "reach_id",
                "year",
                "month",
                "station_id",
                "q_site",
                "Q_obsv_cfs",
            ],
        ]
        .rename(
            columns={
                "q_site": "station_name",
                "Q_obsv_cfs": "observed_discharge_cfs",
            }
        )
        .sort_values(["station_name", "year", "month"])
        .reset_index(drop=True)
    )
    active_stations = set(observations["station_name"].astype(str).unique())
    require(
        FIXED_EXCLUDED.isdisjoint(active_stations),
        f"Fixed excluded station leaked into observations: {FIXED_EXCLUDED & active_stations}",
    )
    require(PROTECTED.issubset(active_stations), "Protected Shijiao station is absent")
    require(
        observations["observed_discharge_cfs"].gt(0).all(),
        "Observed discharge contains non-positive values",
    )

    hswud = pd.read_csv(HSWUD)
    hswud = hswud.loc[hswud["year"].between(start_year, end_year)].copy()
    require(
        not hswud.duplicated(["reach_id", "year", "month"]).any(),
        "HSWUD is not unique by reach-year-month",
    )
    require(
        hswud["reach_id"].nunique() == expected_reaches
        and len(hswud) == expected_reaches * expected_months,
        "HSWUD development coverage is incomplete",
    )
    area = static.set_index("reach_id")["catchment_area_km2"]
    hswud["catchment_area_km2"] = hswud["reach_id"].map(area)
    seconds = month_seconds(hswud["year"], hswud["month"])
    primary_sector_cfs = [
        "hswud_domestic_loc_cfs",
        "hswud_irrigation_loc_cfs",
        "hswud_manufacturing_loc_cfs",
        "hswud_thermal_loc_cfs",
    ]
    hswud["hswud_gross_total_recomputed_cfs"] = hswud[primary_sector_cfs].sum(axis=1)
    # The historical aggregator formed urban_industrial=domestic+manufacturing
    # and then included that derived column in total, double-counting both
    # sectors. Preserve the audit result but never expose the old total as a
    # ledger flux.
    hswud["source_total_double_count_error_cfs"] = (
        hswud["hswud_total_loc_cfs"] - hswud["hswud_gross_total_recomputed_cfs"]
    )
    source_total_is_double_counted = bool(
        np.allclose(
            hswud["source_total_double_count_error_cfs"],
            hswud["hswud_domestic_loc_cfs"]
            + hswud["hswud_manufacturing_loc_cfs"],
            rtol=1e-10,
            atol=1e-10,
        )
    )
    require(
        source_total_is_double_counted,
        "HSWUD total semantics changed; review the aggregation audit",
    )
    local_cfs = primary_sector_cfs + ["hswud_gross_total_recomputed_cfs"]
    for column in local_cfs:
        stem = column.removesuffix("_cfs")
        hswud[f"{stem}_m3_month"] = hswud[column] * CFS_TO_M3_S * seconds
        hswud[f"{stem}_mm"] = (
            hswud[f"{stem}_m3_month"] / (hswud["catchment_area_km2"] * 1000.0)
        )
    management_columns = [
        "reach_id",
        "year",
        "month",
        "catchment_area_km2",
        "source_total_double_count_error_cfs",
    ]
    management_columns += local_cfs
    management_columns += [
        c
        for c in hswud.columns
        if c.endswith("_m3_month") or c.endswith("_mm")
    ]
    management = hswud[management_columns].copy()
    management["gross_withdrawal_status"] = "downscaled;scenario_only"
    management["surface_withdrawal_mm"] = np.nan
    management["surface_withdrawal_status"] = "missing"
    management["groundwater_withdrawal_mm"] = np.nan
    management["groundwater_withdrawal_status"] = "missing"
    management["surface_return_flow_mm"] = np.nan
    management["surface_return_flow_status"] = "missing"
    management["groundwater_return_flow_mm"] = np.nan
    management["groundwater_return_flow_status"] = "missing"
    management["interbasin_transfer_mm"] = np.nan
    management["interbasin_transfer_status"] = "missing"
    management["boundary_inflow_mm"] = np.nan
    management["boundary_inflow_status"] = "missing"
    management = management.sort_values(["reach_id", "year", "month"]).reset_index(
        drop=True
    )

    policy = pd.read_csv(POLICY, encoding="utf-8-sig")
    policy_excluded = set(
        policy.loc[policy["exclude_before_training"].astype(bool), "station_name"].astype(str)
    )
    require(
        FIXED_EXCLUDED.issubset(policy_excluded),
        f"Policy does not encode every fixed exclusion: {FIXED_EXCLUDED - policy_excluded}",
    )
    require(
        not policy.loc[policy["station_name"].isin(PROTECTED), "exclude_before_training"]
        .astype(bool)
        .any(),
        "Protected station is excluded by policy",
    )

    topology_out = topology.sort_values("reach_id").reset_index(drop=True)
    policy_out = policy.sort_values("station_name").reset_index(drop=True)
    output_records = [
        write_parquet_with_hash(static, INPUTS / "reach_static.parquet"),
        write_parquet_with_hash(
            forcing, INPUTS / "reach_month_forcing_2006_2018.parquet"
        ),
        write_parquet_with_hash(
            management, INPUTS / "management_flux_scenario_2006_2018.parquet"
        ),
        write_parquet_with_hash(
            observations, INPUTS / "station_observation_2006_2018.parquet"
        ),
    ]
    topology_path = INPUTS / "topology_edges.csv"
    topology_out.to_csv(topology_path, index=False, encoding="utf-8-sig")
    output_records.append(file_record(topology_path, "ledger_product", "derived"))
    policy_path = INPUTS / "station_policy.csv"
    policy_out.to_csv(policy_path, index=False, encoding="utf-8-sig")
    output_records.append(file_record(policy_path, "ledger_product", "copied_with_sort"))

    locked = {
        "source": str(INDATA.resolve()),
        "source_sha256": sha256(INDATA),
        "locked_years": [2019, 2020, 2021, 2022],
        "observations_copied": False,
        "rule": "Only the final unique candidate may read these years once; no retuning.",
    }
    locked_path = MANIFEST_DIR / "locked_confirmation_manifest.json"
    locked_path.write_text(
        json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_records.append(
        file_record(locked_path, "locked_confirmation_contract", "metadata_only")
    )

    source_specs = [
        (INDATA, "latest model panel", "observed/derived"),
        (POLICY, "station exclusion policy", "reported"),
        (STATION_MATCH, "station reach mapping", "reported"),
        (REACH_SUMMARY, "reach static topology summary", "reported"),
        (TOPOLOGY, "directed topology edges", "reported"),
        (SOIL, "effective soil water storage", "derived"),
        (BEDROCK_RASTER, "depth-to-bedrock raster", "observed"),
        (BEDROCK_TABLE, "reach depth-to-bedrock aggregate", "derived"),
        (HSWUD, "sector gross water use", "downscaled/scenario_only"),
        (TARGET_PLAN, "series decision contract", "reported"),
        (CONTRACT_PATH, "run data contract", "reported"),
    ]
    source_records = [file_record(*spec) for spec in source_specs]
    provenance = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_window": [start_year, end_year],
        "locked_confirmation_window": [2019, 2022],
        "sources": source_records,
        "products": output_records,
    }
    provenance_path = MANIFEST_DIR / "provenance_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(source_records + output_records).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )

    field_rows = [
        ("P_mm", "observed", True, True, "CHM_PRE V2.1 monthly precipitation"),
        ("PET_mm", "observed", True, True, "CMFD FAO56 monthly PET"),
        (
            "AET_diagnostic_mm",
            "diagnostic_only",
            True,
            False,
            "ERA5-Land AET; independent Q78 process check only",
        ),
        ("catchment_area_km2", "reported", True, True, "local reach catchment"),
        ("topology_edges", "reported", True, True, "directed reach graph"),
        ("soil_storage_eff_mm", "derived", True, True, "230/230 reaches"),
        ("depth_to_bedrock_m", "derived", True, True, "230/230 reaches"),
        (
            "slope_m_m",
            "missing",
            False,
            False,
            "legacy SLOPE is an all-zero placeholder; rebuild from DEM",
        ),
        (
            "HSWUD_sector_gross",
            "downscaled;scenario_only",
            True,
            False,
            "four primary sectors; gross local withdrawals only",
        ),
        (
            "HSWUD_source_total",
            "missing",
            False,
            False,
            "historical total double-counts domestic and manufacturing; excluded",
        ),
        (
            "surface_withdrawal",
            "missing",
            False,
            False,
            "no source split from gross HSWUD",
        ),
        (
            "groundwater_withdrawal",
            "missing",
            False,
            False,
            "no source split from gross HSWUD",
        ),
        (
            "consumptive_use",
            "missing",
            False,
            False,
            "no reported sector net abstraction",
        ),
        (
            "surface_return_flow",
            "missing",
            False,
            False,
            "no reported destination, fraction, or lag",
        ),
        (
            "groundwater_return_flow",
            "missing",
            False,
            False,
            "no reported destination, fraction, or lag",
        ),
        (
            "reservoir_storage_capacity",
            "missing",
            False,
            False,
            "legacy reservoir fields are all-zero placeholders",
        ),
        (
            "reservoir_operation",
            "missing",
            False,
            False,
            "no monthly storage/release series",
        ),
        (
            "interbasin_transfer",
            "missing",
            False,
            False,
            "legacy div_transfer is an all-zero placeholder",
        ),
        (
            "boundary_inflow",
            "missing",
            False,
            False,
            "legacy boundary_cfs is an all-zero placeholder",
        ),
        (
            "observed_discharge",
            "observed",
            True,
            False,
            "physically separated station-observation product",
        ),
    ]
    field_status = pd.DataFrame(
        field_rows,
        columns=[
            "field",
            "semantic_state",
            "coverage_pass",
            "formal_model_input_ready",
            "evidence",
        ],
    )
    field_status.to_csv(
        REPORT_DIR / "field_status.csv", index=False, encoding="utf-8-sig"
    )

    ledger_integrity = bool(
        len(static) == expected_reaches
        and len(forcing) == expected_reaches * expected_months
        and forcing["year"].max() == end_year
        and FIXED_EXCLUDED.isdisjoint(active_stations)
        and PROTECTED.issubset(active_stations)
    )
    nat_conservation_ready = bool(
        ledger_integrity
        and static["soil_storage_eff_mm"].notna().all()
        and static["depth_to_bedrock_m"].notna().all()
        and forcing[["P_mm", "PET_mm"]].notna().all().all()
    )
    nat_full_routing_ready = bool(nat_conservation_ready and not slope_placeholder)
    man_formal_ready = False
    gate = {
        "run_id": RUN.name,
        "phase": "shared_hydrology_ledger_and_data_readiness_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_window": [start_year, end_year],
        "locked_confirmation_window": [2019, 2022],
        "locked_observations_loaded": False,
        "counts": {
            "reaches": int(len(static)),
            "reach_months": int(len(forcing)),
            "development_months_per_reach": expected_months,
            "active_observation_stations": int(len(active_stations)),
            "observation_rows": int(len(observations)),
        },
        "station_policy": {
            "fixed_excluded": sorted(FIXED_EXCLUDED),
            "excluded_absent": FIXED_EXCLUDED.isdisjoint(active_stations),
            "protected_retained": sorted(PROTECTED),
            "protected_present": PROTECTED.issubset(active_stations),
        },
        "legacy_placeholder_audit": {
            "SLOPE_all_zero": slope_placeholder,
            "SLOPE_unique_reach_rows": int(len(slope_audit)),
            "boundary_cfs_all_zero": bool(np.allclose(panel["boundary_cfs"], 0.0)),
            "div_transfer_all_zero": bool(np.allclose(panel["div_transfer"], 0.0)),
            "Flow_mgd_4952_all_zero": bool(np.allclose(panel["Flow_mgd_4952"], 0.0)),
            "Flow_mgd_INDU_all_zero": bool(np.allclose(panel["Flow_mgd_INDU"], 0.0)),
            "SurfAre_all_zero": bool(np.allclose(panel["SurfAre"], 0.0)),
            "WB_AreaKm2_o_all_zero": bool(np.allclose(panel["WB_AreaKm2_o"], 0.0)),
            "HSWUD_source_total_double_counts_domestic_and_manufacturing": (
                source_total_is_double_counted
            ),
        },
        "gates": {
            "shared_ledger_integrity": ledger_integrity,
            "q78_nat_conservation_core_data_ready": nat_conservation_ready,
            "q78_nat_full_routing_data_ready": nat_full_routing_ready,
            "q78_man_formal_data_ready": man_formal_ready,
            "hswud_gross_scenario_data_ready": True,
            "aet_independent_diagnostic_ready": True,
        },
        "decision": "CONTINUE_NAT_AFTER_SLOPE_REPAIR_MAN_BLOCKED",
        "authorized_next_action": "REPAIR_REACH_SLOPE",
        "authorized_scope": (
            "In the next dynamic-number folder, rebuild non-negative downstream "
            "reach slope from local DEM/reach geometry, preserve direction QA, "
            "and rerun the NAT routing-data gate. Do not fit Q72 or Q78 yet."
        ),
        "formal_management_blockers": [
            "HSWUD does not distinguish surface-water and groundwater sources",
            "historical HSWUD total double-counted domestic and manufacturing; ledger uses a recomputed four-sector gross total",
            "gross withdrawal cannot be converted to net abstraction without independent consumption/return evidence",
            "reservoir capacity, commissioning year, and monthly operation are missing",
            "interbasin transfer and cross-border boundary inflow are missing",
        ],
        "series_terminal": False,
        "passed": ledger_integrity and nat_conservation_ready,
    }
    gate_path = REPORT_DIR / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# 共享水文账本与数据门禁",
        "",
        f"- 开发窗口：{start_year}–{end_year}；2019–2022 未载入，只有源文件哈希。",
        f"- 河段：{len(static)}；河段月：{len(forcing)}；观测站：{len(active_stations)}。",
        f"- 固定排除站全部缺席：{FIXED_EXCLUDED.isdisjoint(active_stations)}。",
        f"- 石角站存在：{PROTECTED.issubset(active_stations)}。",
        f"- 土壤储水覆盖：{int(static['soil_storage_eff_mm'].notna().sum())}/{len(static)}。",
        f"- 基岩深度覆盖：{int(static['depth_to_bedrock_m'].notna().sum())}/{len(static)}。",
        "",
        "## 门禁结论",
        "",
        f"- 共享账本完整性：{'PASS' if ledger_integrity else 'FAIL'}。",
        f"- Q78-NAT 守恒核心数据：{'PASS' if nat_conservation_ready else 'FAIL'}。",
        f"- Q78-NAT 完整路由数据：{'PASS' if nat_full_routing_ready else 'FAIL'}；SLOPE 全零占位，必须从 DEM 重建。",
        "- Q78-MAN 正式数据：FAIL；HSWUD 只允许作为 gross-withdrawal 情景量。",
        "- HSWUD 旧 total：FAIL；其重复计入生活与制造，本账本改用四个原始部门重算 gross total。",
        "- AET：可作为独立过程诊断，不得作为 Q78 强迫输入。",
        "",
        "## 下一授权动作",
        "",
        "`REPAIR_REACH_SLOPE`：在下一个动态编号目录从本地 DEM 与 reach 几何重建顺流非负坡度，完成方向审计后再决定是否进入 Q78-NAT。",
    ]
    (REPORT_DIR / "gate.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    run_manifest = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_contract": "conda sparrow",
        "python": sys.version,
        "platform": platform.platform(),
        "source_manifest_sha256": sha256(provenance_path),
        "gate_sha256": sha256(gate_path),
        "products": output_records,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "SHARED_LEDGER_GATE "
        f"passed={gate['passed']} nat_core={nat_conservation_ready} "
        f"nat_routing={nat_full_routing_ready} man={man_formal_ready} "
        f"next={gate['authorized_next_action']}"
    )
    if not gate["passed"]:
        raise SystemExit("Shared ledger gate failed")


if __name__ == "__main__":
    main()
