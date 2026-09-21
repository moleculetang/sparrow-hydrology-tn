"""Build registered local and upstream multiscale static DPL attributes.

No discharge, TN, station identity or evaluation-period hydroclimate is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_15"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE14 = ROOT / "5_Test" / "20260826_14"
CORE = ROOT / "5_Test" / "20260825_3" / "scripts"
sys.path.insert(0, str(CORE))

from hydrology_core import load_topology  # noqa: E402


SOIL = ROOT / "5_Test" / "20260825_5" / "outputs" / "mpr_attributes_by_reach.parquet"
DEM = ROOT / "5_Test" / "20260826_2" / "outputs" / "reach_static_attributes_dem_recomputed.parquet"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
STAGE13_HASHES = ROOT / "5_Test" / "20260826_13" / "reports" / "input_hash_registry.json"

LOCAL_FEATURES = [
    "log_awc_0_200_mm",
    "bulk_density_0_30_g_cm3",
    "glhymps_log10_permeability_m2",
    "glhymps_porosity",
    "log_dem_slope",
    "log_predev_annual_precipitation_mm",
    "log_predev_annual_pet_mm",
]
TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardized(frame: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    center = frame[fields].mean()
    scale = frame[fields].std(ddof=0)
    if bool((scale <= 0.0).any()):
        raise RuntimeError("A registered feature has zero variance")
    result = frame[["reach_id"]].copy()
    result[fields] = (frame[fields] - center) / scale
    return result, {"center": center.to_dict(), "scale": scale.to_dict()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    authorization = json.loads((STAGE14 / "program_manifest.json").read_text(encoding="utf-8"))
    if authorization["authorized_successor"] != "20260826_15":
        raise RuntimeError("Stage 15 is not authorized by a passing Stage 14")
    stage14_validation = json.loads((STAGE14 / "reports" / "validation.json").read_text(encoding="utf-8"))
    if not stage14_validation["all_checks_pass"]:
        raise RuntimeError("Stage 14 validation is not passing")

    registered = json.loads(STAGE13_HASHES.read_text(encoding="utf-8"))
    registered_by_role = {row["role"]: row for row in registered["files"]}
    for role in (
        "soil_hydroclimate_attributes",
        "dem_recomputed_attributes",
        "daily_forcing",
        "q72_reach_area_bridge",
        "topology",
    ):
        row = registered_by_role[role]
        if sha256(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"Registered input changed: {role}")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    index = {int(reach): position for position, reach in enumerate(reach_ids)}

    soil = pd.read_parquet(
        SOIL,
        columns=[
            "reach_id",
            "awc_0_200_mm",
            "bulk_density_0_30_g_cm3",
            "glhymps_log10_permeability_m2",
            "glhymps_porosity",
        ],
    ).set_index("reach_id").reindex(reach_ids)
    dem = pd.read_parquet(
        DEM, columns=["reach_id", "slope_for_spatial_mapping", "slope_below_dem_resolution_censored"]
    ).set_index("reach_id").reindex(reach_ids)
    area = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64)
    )

    forcing = pd.read_parquet(
        FORCING,
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing["date"] = pd.to_datetime(forcing.date)
    forcing = forcing.loc[forcing.date.dt.year.between(2006, 2015)].copy()
    forcing["year"] = forcing.date.dt.year
    annual = (
        forcing.groupby(["reach_id", "year"], as_index=False)[["precipitation_daily_mm", "pet_fao56_mm_day"]]
        .sum()
        .groupby("reach_id")[["precipitation_daily_mm", "pet_fao56_mm_day"]]
        .mean()
        .reindex(reach_ids)
    )

    local = pd.DataFrame(
        {
            "reach_id": reach_ids,
            "log_awc_0_200_mm": np.log(soil.awc_0_200_mm.to_numpy(np.float64)),
            "bulk_density_0_30_g_cm3": soil.bulk_density_0_30_g_cm3.to_numpy(np.float64),
            "glhymps_log10_permeability_m2": soil.glhymps_log10_permeability_m2.to_numpy(np.float64),
            "glhymps_porosity": soil.glhymps_porosity.to_numpy(np.float64),
            "log_dem_slope": np.log(dem.slope_for_spatial_mapping.to_numpy(np.float64)),
            "log_predev_annual_precipitation_mm": np.log(annual.precipitation_daily_mm.to_numpy(np.float64)),
            "log_predev_annual_pet_mm": np.log(annual.pet_fao56_mm_day.to_numpy(np.float64)),
            "local_catchment_area_km2": area,
            "slope_censored": dem.slope_below_dem_resolution_censored.to_numpy(bool),
        }
    )
    if not np.isfinite(local[LOCAL_FEATURES + ["local_catchment_area_km2"]].to_numpy()).all():
        raise RuntimeError("Local attributes are incomplete or non-finite")

    # Row target, column source: fraction of each local source reaching target.
    support = np.eye(len(reach_ids), dtype=np.float64)
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            support[index[target], :] += fraction * support[index[reach], :]
    source_area = support * area[None, :]
    contributing_area = source_area.sum(axis=1)
    weights = source_area / contributing_area[:, None]
    values = local[LOCAL_FEATURES].to_numpy(np.float64)
    upstream_mean = weights @ values
    upstream_variance = np.maximum(weights @ (values**2) - upstream_mean**2, 0.0)
    upstream_std = np.sqrt(upstream_variance)

    multiscale = local[["reach_id"] + LOCAL_FEATURES].copy()
    upstream_mean_names = [f"upstream_mean_{name}" for name in LOCAL_FEATURES]
    upstream_std_names = [f"upstream_std_{name}" for name in LOCAL_FEATURES]
    multiscale[upstream_mean_names] = upstream_mean
    multiscale[upstream_std_names] = upstream_std
    multiscale["log_contributing_area_km2"] = np.log(contributing_area)
    multiscale_features = LOCAL_FEATURES + upstream_mean_names + upstream_std_names + ["log_contributing_area_km2"]
    if len(multiscale_features) != 22:
        raise RuntimeError("Multiscale contract must contain exactly 22 features")

    local_standard, local_scaling = standardized(local, LOCAL_FEATURES)
    multiscale_standard, multiscale_scaling = standardized(multiscale, multiscale_features)
    local.to_parquet(OUT / "local_static_features_raw.parquet", index=False)
    local_standard.to_parquet(OUT / "local_static_features_standardized.parquet", index=False)
    multiscale.to_parquet(OUT / "multiscale_static_features_raw.parquet", index=False)
    multiscale_standard.to_parquet(OUT / "multiscale_static_features_standardized.parquet", index=False)

    summary_rows = []
    for family, frame, fields in (
        ("local", local, LOCAL_FEATURES),
        ("multiscale", multiscale, multiscale_features),
    ):
        for field in fields:
            summary_rows.append(
                {
                    "family": family,
                    "feature": field,
                    "minimum": float(frame[field].min()),
                    "median": float(frame[field].median()),
                    "maximum": float(frame[field].max()),
                    "mean": float(frame[field].mean()),
                    "std_population": float(frame[field].std(ddof=0)),
                }
            )
    pd.DataFrame(summary_rows).to_parquet(OUT / "feature_summary.parquet", index=False)

    local_matrix = local_standard[LOCAL_FEATURES].to_numpy(np.float64)
    multiscale_matrix = multiscale_standard[multiscale_features].to_numpy(np.float64)
    qa = {
        "stage": "20260826_15",
        "reach_count": len(local),
        "local_feature_count": len(LOCAL_FEATURES),
        "multiscale_feature_count": len(multiscale_features),
        "local_matrix_rank": int(np.linalg.matrix_rank(local_matrix)),
        "multiscale_matrix_rank": int(np.linalg.matrix_rank(multiscale_matrix)),
        "local_max_abs_standardized_mean": float(np.max(np.abs(local_matrix.mean(axis=0)))),
        "local_max_abs_standardized_std_minus_one": float(np.max(np.abs(local_matrix.std(axis=0) - 1.0))),
        "multiscale_max_abs_standardized_mean": float(np.max(np.abs(multiscale_matrix.mean(axis=0)))),
        "multiscale_max_abs_standardized_std_minus_one": float(np.max(np.abs(multiscale_matrix.std(axis=0) - 1.0))),
        "support_diagonal_min": float(np.diag(support).min()),
        "support_negative_count": int((support < 0.0).sum()),
        "contributing_area_not_less_than_local": bool(np.all(contributing_area + 1.0e-12 >= area)),
        "maximum_upstream_weight_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        "slope_censored_count": int(local.slope_censored.sum()),
        "hydroclimate_years": "2006-2015 only",
        "2016_2022_forcing_used": False,
        "discharge_read": False,
        "TN_read": False,
        "station_identity_used": False,
        "terminal_tree_identity_used": False,
    }
    qa["all_checks_pass"] = bool(
        qa["reach_count"] == 230
        and qa["local_feature_count"] == 7
        and qa["multiscale_feature_count"] == 22
        and qa["local_matrix_rank"] == 7
        and qa["multiscale_matrix_rank"] == 22
        and qa["local_max_abs_standardized_mean"] <= 1.0e-12
        and qa["local_max_abs_standardized_std_minus_one"] <= 1.0e-12
        and qa["multiscale_max_abs_standardized_mean"] <= 1.0e-12
        and qa["multiscale_max_abs_standardized_std_minus_one"] <= 1.0e-12
        and qa["support_diagonal_min"] == 1.0
        and qa["support_negative_count"] == 0
        and qa["contributing_area_not_less_than_local"]
        and qa["maximum_upstream_weight_sum_error"] <= 1.0e-12
    )
    write_json(REPORTS / "multiscale_attribute_qa.json", qa)

    registry = {
        "created_at": now(),
        "local_features": LOCAL_FEATURES,
        "multiscale_features": multiscale_features,
        "local_scaling": local_scaling,
        "multiscale_scaling": multiscale_scaling,
        "upstream_definition": "inclusive topology support; local-area and routing-fraction weighted mean and population standard deviation",
        "area_feature": "log contributing local catchment area",
        "hydroclimate_definition": "mean annual sums from 2006-2015 daily forcing; no 2016-2022 forcing",
        "allowed_networks": {
            "DPL_HBV_LOCAL_STATIC": "7 -> 16 -> 8",
            "DPL_HBV_MULTISCALE_STATIC": "22 -> 16 -> 8",
        },
    }
    write_json(REPORTS / "feature_registry.json", registry)
    if not qa["all_checks_pass"]:
        raise RuntimeError(f"Stage 15 attribute QA failed: {qa}")

    contract = {
        "stage": "20260826_15",
        "status": "PASS_MULTISCALE_STATIC_ATTRIBUTE_QA",
        "candidate_training_performed": False,
        "discharge_read": False,
        "TN_read": False,
        "authorized_successor": "20260826_16",
    }
    write_json(RUN / "experiment_contract.json", contract)
    program = dict(authorization)
    program["stage_status"] = dict(authorization["stage_status"])
    program["stage_status"]["20260826_15"] = "PASS_MULTISCALE_STATIC_ATTRIBUTE_QA"
    program["stage_status"]["20260826_16"] = "authorized_next_Q_only_static_DPL_HBV"
    program["authorized_successor"] = "20260826_16"
    write_json(RUN / "program_manifest.json", program)
    report = f"""# 20260826_15 多尺度静态属性

状态：`PASS_MULTISCALE_STATIC_ATTRIBUTE_QA`。构建了7维本地属性和22维本地—上游属性；没有读取流量或TN，也没有训练网络。

22维结构为7个本地属性、7个上游面积加权均值、7个上游面积加权标准差和1个对数汇水面积。降水与PET气候态只使用2006–2015，避免2016–2022评价期信息进入静态参数映射。

本地矩阵秩为`{qa['local_matrix_rank']}`，多尺度矩阵秩为`{qa['multiscale_matrix_rank']}`；上游权重闭合最大误差为`{qa['maximum_upstream_weight_sum_error']:.3e}`。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_15\n\nLocal and upstream multiscale static DPL attributes. No candidate training or discharge read occurs here.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
