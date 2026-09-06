from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "soil_storage_semantics"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "soilgrids_prb_buffer"
CATCHMENTS = (
    ROOT / "0_reach_topology" / "results" / "vectors"
    / "reach_catchments.shp"
)
CURRENT_TABLE = (
    ROOT / "0_reach_topology" / "results" / "tables"
    / "reach_catchment_soil_storage_eff.csv"
)
CURRENT_MANIFEST = (
    ROOT / "0_reach_topology" / "data" / "processed" / "soil_prb"
    / "soil_storage_eff_manifest.json"
)
MODEL = (
    ROOT / "5_Test" / "20260729_23" / "outputs"
    / "q78_nat_groundwater_reach_month.parquet"
)
FLUX = (
    ROOT / "5_Test" / "20260729_24" / "reports"
    / "baseflow_failure_attribution" / "local_flux_attribution.parquet"
)
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_24" / "reports"
    / "baseflow_failure_attribution" / "gate.json"
)
DEPTHS = [
    (0, 5, "0-5cm"),
    (5, 15, "5-15cm"),
    (15, 30, "15-30cm"),
    (30, 60, "30-60cm"),
    (60, 100, "60-100cm"),
    (100, 200, "100-200cm"),
]


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


def gdal_path() -> Path:
    candidate = Path(sys.prefix) / "Library" / "bin" / "gdal.exe"
    if not candidate.exists():
        raise RuntimeError(f"GDAL command missing in sparrow env: {candidate}")
    return candidate


def build_full_profile_awc() -> tuple[Path, list[Path], Path]:
    output = OUTPUTS / "soil_awc_0_200cm_no_bedrock_truncation.tif"
    table = OUTPUTS / "full_profile_awc_reach_zonal.geojson"
    intermediate = OUTPUTS / "gdal_intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    source_paths = []
    layer_outputs = []
    for top, bottom, label in DEPTHS:
        theta33 = RAW / "wv0033" / f"wv0033_{label}_mean_prb_buffer.tif"
        theta1500 = RAW / "wv1500" / f"wv1500_{label}_mean_prb_buffer.tif"
        coarse = RAW / "cfvo" / f"cfvo_{label}_mean_prb_buffer.tif"
        for path in [theta33, theta1500, coarse]:
            if not path.exists():
                raise RuntimeError(f"Missing local SoilGrids source: {path}")
            source_paths.append(path)
        safe_label = label.replace("-", "_")
        difference = intermediate / f"water_fraction_difference_{safe_label}.tif"
        coarse_complement = intermediate / f"coarse_complement_{safe_label}.tif"
        layer_output = intermediate / f"awc_{safe_label}.tif"
        subprocess.run([
            str(gdal_path()), "raster", "calc", "--quiet", "--overwrite",
            "--dialect", "builtin", "--calc", "diff",
            "--propagate-nodata", "--nodata", "-9999",
            "--output-data-type", "Float32",
            "--input", f"A={theta33}", "--input", f"B={theta1500}",
            "--output", str(difference),
        ], check=True)
        subprocess.run([
            str(gdal_path()), "raster", "scale", "--quiet", "--overwrite",
            "--src-min", "0", "--src-max", "10000",
            "--dst-min", "1", "--dst-max", "0", "--no-clip",
            "--output-data-type", "Float32",
            "--input", str(coarse), "--output", str(coarse_complement),
        ], check=True)
        thickness_scale = (bottom - top) * 10.0 * 0.001
        subprocess.run([
            str(gdal_path()), "raster", "calc", "--quiet", "--overwrite",
            "--dialect", "builtin",
            "--calc", f"mul(k={thickness_scale},propagateNoData=true)",
            "--propagate-nodata", "--nodata", "-9999",
            "--output-data-type", "Float32",
            "--input", f"A={difference}",
            "--input", f"B={coarse_complement}",
            "--output", str(layer_output),
        ], check=True)
        layer_outputs.append(layer_output)
    sum_command = [
        str(gdal_path()), "raster", "calc", "--quiet", "--overwrite",
        "--dialect", "builtin",
        "--calc", "sum(propagateNoData=true)",
        "--propagate-nodata", "--nodata", "-9999",
        "--output-data-type", "Float32",
    ]
    for index, layer_output in enumerate(layer_outputs):
        sum_command.extend([
            "--input", f"{chr(ord('A') + index)}={layer_output}"
        ])
    sum_command.extend(["--output", str(output)])
    subprocess.run(sum_command, check=True)
    return output, source_paths, table


def zonal_means(raster: Path) -> pd.DataFrame:
    output = OUTPUTS / "full_profile_awc_reach_zonal.geojson"
    reprojected = OUTPUTS / "reach_catchments_esri54009.geojson"
    subprocess.run([
        str(gdal_path()), "vector", "reproject", "--quiet",
        "--overwrite", "--dst-crs", "ESRI:54009",
        str(CATCHMENTS), str(reprojected),
    ], check=True)
    command = [
        str(gdal_path()), "raster", "zonal-stats", "--quiet",
        "--overwrite", "--zones", str(CATCHMENTS),
        "--include-field", "reach_id",
        "--stat", "mean", "--stat", "count",
        str(raster), str(output),
    ]
    command[command.index(str(CATCHMENTS))] = str(reprojected)
    subprocess.run(command, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = []
    for feature in payload["features"]:
        properties = feature["properties"]
        mean_key = next(
            key for key in properties
            if key.lower().endswith("mean") or key.lower() == "mean"
        )
        count_key = next(
            key for key in properties
            if key.lower().endswith("count") or key.lower() == "count"
        )
        rows.append({
            "reach_id": int(properties["reach_id"]),
            "soil_awc_full_0_200cm_mm": float(properties[mean_key]),
            "valid_full_profile_cells": int(properties[count_key]),
        })
    return pd.DataFrame(rows).sort_values("reach_id")


def main() -> None:
    for directory in [REPORT, OUTPUTS, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if (
        parent["authorized_next_action"]
        != "REBUILD_SOIL_WATER_PARTITION_BEFORE_ANY_KG_OR_ROUTING_TUNING"
    ):
        raise RuntimeError("Parent gate does not authorize soil semantic audit")
    raster, raw_sources, full_table = build_full_profile_awc()
    full = zonal_means(raster)
    current = pd.read_csv(CURRENT_TABLE)[
        ["reach_id", "soil_storage_eff_mm"]
    ]
    comparison = current.merge(
        full, on="reach_id", validate="one_to_one"
    )
    comparison["full_to_current_capacity_ratio"] = (
        comparison["soil_awc_full_0_200cm_mm"]
        / comparison["soil_storage_eff_mm"]
    )
    comparison["bedrock_truncation_removed_capacity_mm"] = (
        comparison["soil_awc_full_0_200cm_mm"]
        - comparison["soil_storage_eff_mm"]
    )
    comparison.to_csv(
        REPORT / "reach_soil_capacity_semantic_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model = pd.read_parquet(MODEL)
    flux = pd.read_parquet(FLUX)
    monthly = flux.merge(
        model[
            ["reach_id", "year", "month", "P_mm", "AET_mm"]
        ],
        on=["reach_id", "year", "month"],
        validate="one_to_one",
    ).merge(
        comparison, on="reach_id", validate="many_to_one"
    )
    monthly["climatic_water_input_after_aet_mm"] = np.maximum(
        monthly["P_mm"] - monthly["AET_mm"], 0.0
    )
    monthly["water_input_exceeds_current_capacity"] = (
        monthly["climatic_water_input_after_aet_mm"]
        > monthly["soil_storage_eff_mm"]
    )
    monthly["water_input_exceeds_full_profile_awc"] = (
        monthly["climatic_water_input_after_aet_mm"]
        > monthly["soil_awc_full_0_200cm_mm"]
    )
    monthly.to_parquet(
        OUTPUTS / "reach_month_capacity_scale_audit.parquet", index=False
    )
    excess_volume = float(
        (monthly["excess_m3"]).sum()
    )
    total_generation = float(
        (
            monthly["recharge_m3"]
            + monthly["interflow_m3"]
            + monthly["excess_m3"]
        ).sum()
    )
    metrics = {
        "reach_count": int(len(comparison)),
        "current_capacity_median_mm": float(
            comparison["soil_storage_eff_mm"].median()
        ),
        "full_profile_awc_median_mm": float(
            comparison["soil_awc_full_0_200cm_mm"].median()
        ),
        "full_to_current_ratio_median": float(
            comparison["full_to_current_capacity_ratio"].median()
        ),
        "full_to_current_ratio_min": float(
            comparison["full_to_current_capacity_ratio"].min()
        ),
        "fraction_months_input_exceeds_current_capacity": float(
            monthly["water_input_exceeds_current_capacity"].mean()
        ),
        "fraction_months_input_exceeds_full_profile_awc": float(
            monthly["water_input_exceeds_full_profile_awc"].mean()
        ),
        "excess_fraction_of_total_generation": excess_volume / total_generation,
        "current_product_is_bedrock_truncated_awc": True,
        "current_product_is_total_pore_storage": False,
        "weathered_or_fractured_bedrock_storage_represented": False,
        "subgrid_storage_distribution_represented": False,
        "preferential_recharge_path_represented": False,
        "monthly_precipitation_is_lumped_before_drainage": True,
    }
    checks = {
        "all_230_reaches_recomputed": len(comparison) == 230,
        "full_profile_awc_materially_larger": (
            metrics["full_to_current_ratio_median"] >= 2.0
        ),
        "monthly_input_frequently_exceeds_current_capacity": (
            metrics[
                "fraction_months_input_exceeds_current_capacity"
            ] >= 0.20
        ),
        "overflow_is_dominant_generation": (
            metrics["excess_fraction_of_total_generation"] >= 0.75
        ),
        "storage_semantics_are_not_equivalent": (
            metrics["current_product_is_bedrock_truncated_awc"]
            and not metrics["current_product_is_total_pore_storage"]
        ),
        "missing_heterogeneity_and_preferential_recharge": (
            not metrics["subgrid_storage_distribution_represented"]
            and not metrics["preferential_recharge_path_represented"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    mismatch = all(checks.values())
    action = (
        "TEST_ONE_DISTRIBUTED_STORAGE_PLUS_PREFERENTIAL_RECHARGE_CANDIDATE"
        if mismatch
        else "STOP_AND_ADD_INDEPENDENT_SOIL_OR_WEATHERED_BEDROCK_EVIDENCE"
    )
    gate = {
        "run_id": "20260729_25",
        "phase": "soil_storage_semantic_audit",
        "checks": checks,
        "metrics": metrics,
        "semantic_mismatch_confirmed": mismatch,
        "decision": (
            "CURRENT_AWC_CANNOT_SERVE_AS_COMPLETE_MONTHLY_HYDROLOGIC_STORAGE"
            if mismatch else "SEMANTIC_MISMATCH_NOT_CONFIRMED"
        ),
        "authorized_next_action": action,
        "capacity_multiplier_calibrated": False,
        "model_run": False,
        "station_discharge_read": False,
        "pml_primary_aet_reference": True,
        "confirmation_years_used": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 土壤储水容量与月尺度产流语义审计",
        "",
        f"- 结论：`{gate['decision']}`",
        f"- 下一动作：`{action}`",
        "",
        "## 数值",
        "",
        f"- 当前基岩截断 AWC 中位数：{metrics['current_capacity_median_mm']:.1f} mm",
        f"- 0–200 cm 完整 SoilGrids AWC 中位数：{metrics['full_profile_awc_median_mm']:.1f} mm",
        f"- 完整/当前容量比中位数：{metrics['full_to_current_ratio_median']:.2f}",
        (
            "- `max(P-AET,0)` 超过当前容量的 reach-month："
            f"{metrics['fraction_months_input_exceeds_current_capacity']:.1%}"
        ),
        (
            "- `max(P-AET,0)` 超过完整 2 m AWC 的 reach-month："
            f"{metrics['fraction_months_input_exceeds_full_profile_awc']:.1%}"
        ),
        f"- 溢流占总产流：{metrics['excess_fraction_of_total_generation']:.1%}",
        "",
        "## 解释",
        "",
        (
            "生产产品是基岩深度截断后的植物可利用水容量，不是饱和前总孔隙"
            "储水，也不包含风化层/裂隙基岩储水。当前月模型却把它当作唯一"
            "储水上限，并在月降水整块加入后、排水发生前先产生溢流；同时没有"
            "网格内容量分布或优先补给路径。两者物理语义不等价。"
        ),
        "",
        (
            "本轮不授权简单容量倍乘。下一轮只能测试一个质量守恒的分布式容量"
            "加优先补给候选，并继续以 PML 约束 AET、以日流量 BFI 作独立门禁。"
        ),
    ]
    (REPORT / "technical_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    sources = [
        *raw_sources, CATCHMENTS, CURRENT_TABLE, CURRENT_MANIFEST,
        MODEL, FLUX, PARENT_GATE, RUN / "experiment_contract.md",
        RUN / "literature_basis.md",
        RUN / "scripts" / "audit_soil_storage_semantics.py",
    ]
    products = [
        raster,
        full_table,
        OUTPUTS / "reach_month_capacity_scale_audit.parquet",
        REPORT / "reach_soil_capacity_semantic_comparison.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [record(path, "soil_semantic_source", "reported_or_derived") for path in sources],
        "products": [record(path, "soil_semantic_product", "derived") for path in products],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
