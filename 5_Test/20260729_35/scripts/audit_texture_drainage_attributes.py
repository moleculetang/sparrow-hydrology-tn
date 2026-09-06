from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "texture_drainage_attribute_audit"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_34" / "reports"
    / "temporal_station_robustness" / "gate.json"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
STATION_PANEL = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit" / "station_bfi_attribute_panel.csv"
)
CURRENT_CORRELATIONS = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit" / "attribute_bfi_spearman.csv"
)
CFVO = (
    ROOT / "0_reach_topology" / "data" / "raw"
    / "soilgrids_prb_buffer" / "cfvo"
)
AWC = (
    ROOT / "5_Test" / "20260729_25" / "outputs"
    / "gdal_intermediate"
)
CATCHMENTS = (
    ROOT / "5_Test" / "20260729_25" / "outputs"
    / "reach_catchments_esri54009.geojson"
)
CONTRACT = RUN / "experiment_contract.md"
LITERATURE = RUN / "literature_basis.md"
DEPTHS = [
    (0, 5, "0-5cm"),
    (5, 15, "5-15cm"),
    (15, 30, "15-30cm"),
    (30, 60, "30-60cm"),
    (60, 100, "60-100cm"),
    (100, 200, "100-200cm"),
]
CANDIDATES = [
    "cfvo_top_0_30_pct",
    "cfvo_deep_30_200_pct",
    "cfvo_full_0_200_pct",
    "cfvo_deep_minus_top_pct",
    "awc_deep_fraction",
    "coarse_low_retention_rank_index",
]
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


def gdal_path() -> Path:
    candidate = Path(sys.prefix) / "Library" / "bin" / "gdal.exe"
    if not candidate.exists():
        raise RuntimeError(f"GDAL missing in sparrow environment: {candidate}")
    return candidate


def parse_zonal(path: Path, value_name: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in payload["features"]:
        properties = feature["properties"]
        mean_keys = [
            key for key in properties
            if key.lower() == "mean" or key.lower().endswith("_mean")
        ]
        count_keys = [
            key for key in properties
            if key.lower() == "count" or key.lower().endswith("_count")
        ]
        if len(mean_keys) != 1 or len(count_keys) != 1:
            raise RuntimeError(
                f"Unexpected zonal fields in {path}: {sorted(properties)}"
            )
        rows.append({
            "reach_id": int(properties["reach_id"]),
            value_name: float(properties[mean_keys[0]]),
            f"{value_name}_valid_cells": int(properties[count_keys[0]]),
        })
    frame = pd.DataFrame(rows).sort_values("reach_id")
    if len(frame) != 230 or frame["reach_id"].nunique() != 230:
        raise RuntimeError(f"Expected 230 reach zones in {path}")
    return frame


def zonal_mean(
    raster: Path, value_name: str, output_name: str, logs: list[str]
) -> pd.DataFrame:
    if not raster.exists():
        raise RuntimeError(f"Missing local raster: {raster}")
    output = OUTPUTS / f"{output_name}.geojson"
    command = [
        str(gdal_path()), "raster", "zonal-stats", "--quiet",
        "--overwrite", "--zones", str(CATCHMENTS),
        "--include-field", "reach_id", "--stat", "mean", "--stat", "count",
        str(raster), str(output),
    ]
    result = subprocess.run(
        command, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    logs.append(
        f"$ {' '.join(command)}\n{result.stdout or ''}".rstrip()
    )
    return parse_zonal(output, value_name)


def build_reach_attributes() -> tuple[pd.DataFrame, list[Path]]:
    tables = []
    sources = []
    command_logs: list[str] = []
    for top, bottom, label in DEPTHS:
        safe = label.replace("-", "_")
        cfvo_path = CFVO / f"cfvo_{label}_mean_prb_buffer.tif"
        awc_path = AWC / f"awc_{safe}.tif"
        sources.extend([cfvo_path, awc_path])
        cfvo_table = zonal_mean(
            cfvo_path, f"cfvo_{safe}_raw",
            f"cfvo_{safe}_reach_zonal", command_logs,
        )
        awc_table = zonal_mean(
            awc_path, f"awc_{safe}_mm",
            f"awc_{safe}_reach_zonal", command_logs,
        )
        tables.extend([cfvo_table, awc_table])
    (LOGS / "gdal_zonal_commands.log").write_text(
        "\n\n".join(command_logs) + "\n", encoding="utf-8"
    )
    reaches = tables[0]
    for table in tables[1:]:
        reaches = reaches.merge(
            table, on="reach_id", validate="one_to_one"
        )
    all_count_columns = [
        column for column in reaches.columns
        if column.endswith("_valid_cells")
    ]
    if reaches[all_count_columns].le(0).any().any():
        raise RuntimeError("At least one reach has no valid soil pixels")
    cfvo_columns = []
    awc_columns = []
    thicknesses = []
    for top, bottom, label in DEPTHS:
        safe = label.replace("-", "_")
        cfvo_columns.append(f"cfvo_{safe}_raw")
        awc_columns.append(f"awc_{safe}_mm")
        thicknesses.append(bottom - top)
    # SoilGrids cfvo is stored with d_factor=10 in cm3/dm3.
    # raw / 10 gives cm3/dm3; division by another 10 converts to vol-%.
    cfvo_values = reaches[cfvo_columns].to_numpy(float) / 100.0
    awc_values = reaches[awc_columns].to_numpy(float)
    thickness = np.asarray(thicknesses, dtype=float)
    top = np.arange(6) < 3
    deep = ~top
    reaches["cfvo_top_0_30_pct"] = np.average(
        cfvo_values[:, top], axis=1, weights=thickness[top]
    )
    reaches["cfvo_deep_30_200_pct"] = np.average(
        cfvo_values[:, deep], axis=1, weights=thickness[deep]
    )
    reaches["cfvo_full_0_200_pct"] = np.average(
        cfvo_values, axis=1, weights=thickness
    )
    reaches["cfvo_deep_minus_top_pct"] = (
        reaches["cfvo_deep_30_200_pct"]
        - reaches["cfvo_top_0_30_pct"]
    )
    reaches["awc_top_0_30_mm"] = awc_values[:, top].sum(axis=1)
    reaches["awc_deep_30_200_mm"] = awc_values[:, deep].sum(axis=1)
    reaches["awc_full_0_200_mm"] = awc_values.sum(axis=1)
    reaches["awc_deep_fraction"] = (
        reaches["awc_deep_30_200_mm"]
        / reaches["awc_full_0_200_mm"]
    )
    low_retention_rank = (
        -reaches["awc_full_0_200_mm"] / 2000.0
    ).rank(method="average", pct=True)
    coarse_rank = reaches["cfvo_full_0_200_pct"].rank(
        method="average", pct=True
    )
    reaches["coarse_low_retention_rank_index"] = (
        coarse_rank + low_retention_rank
    ) / 2.0
    return reaches, sources


def build_graph(static: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming: dict[int, list[int]] = {}
    for row in static.itertuples():
        incoming.setdefault(int(row.tnode), []).append(int(row.reach_id))
    for row in static.itertuples():
        for upstream in incoming.get(int(row.fnode), []):
            if int(upstream) != int(row.reach_id):
                graph.add_edge(int(upstream), int(row.reach_id))
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen reach topology is not a DAG")
    return graph


def station_panel(
    stations: pd.DataFrame, reaches: pd.DataFrame,
    static: pd.DataFrame, graph: nx.DiGraph,
) -> pd.DataFrame:
    indexed = reaches.set_index("reach_id")
    area = static.set_index("reach_id")["inc_area_km2"]
    attributes = CANDIDATES + [
        "awc_top_0_30_mm", "awc_deep_30_200_mm",
        "awc_full_0_200_mm",
    ]
    rows = []
    for station in stations.itertuples():
        outlet = int(station.reach_id)
        ids = sorted(nx.ancestors(graph, outlet) | {outlet})
        subset = indexed.loc[ids]
        weights = area.loc[ids].to_numpy(float)
        row = {
            "station_name": station.station_name,
            "reach_id": outlet,
            "observed_bfi": float(station.observed_bfi),
            "observed_bfi_sensitivity_width": float(
                station.observed_bfi_sensitivity_width
            ),
            "upstream_area_km2": float(weights.sum()),
        }
        for attribute in attributes:
            row[f"upstream_aw_{attribute}"] = float(np.average(
                subset[attribute].to_numpy(float), weights=weights
            ))
        rows.append(row)
    panel = pd.DataFrame(rows).sort_values("reach_id").reset_index(drop=True)
    panel["split"] = np.where(
        np.arange(len(panel)) % 2 == 0,
        "discovery_49", "confirmation_48",
    )
    sensitivity_median = panel["observed_bfi_sensitivity_width"].median()
    panel["sensitivity_group"] = np.where(
        panel["observed_bfi_sensitivity_width"] > sensitivity_median,
        "higher_sensitivity", "lower_sensitivity",
    )
    panel["area_group"] = pd.qcut(
        panel["upstream_area_km2"], 3,
        labels=["small_area", "medium_area", "large_area"],
    ).astype(str)
    return panel


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all_97": panel,
        "discovery_49": panel.loc[panel["split"].eq("discovery_49")],
        "confirmation_48": panel.loc[panel["split"].eq("confirmation_48")],
        "higher_sensitivity": panel.loc[
            panel["sensitivity_group"].eq("higher_sensitivity")
        ],
        "lower_sensitivity": panel.loc[
            panel["sensitivity_group"].eq("lower_sensitivity")
        ],
        "small_area": panel.loc[panel["area_group"].eq("small_area")],
        "medium_area": panel.loc[panel["area_group"].eq("medium_area")],
        "large_area": panel.loc[panel["area_group"].eq("large_area")],
    }
    rows = []
    for subset_name, subset in subsets.items():
        for attribute in CANDIDATES:
            rho = subset["observed_bfi"].corr(
                subset[f"upstream_aw_{attribute}"], method="spearman"
            )
            rows.append({
                "subset": subset_name,
                "attribute": attribute,
                "n": int(len(subset)),
                "spearman": float(rho),
                "absolute_spearman": float(abs(rho)),
            })
    return pd.DataFrame(rows)


def bootstrap_aligned_correlation(
    panel: pd.DataFrame, attribute: str, direction: float,
    repetitions: int = 4000,
) -> tuple[float, float, float]:
    x = panel[f"upstream_aw_{attribute}"].to_numpy(float)
    y = panel["observed_bfi"].to_numpy(float)
    rng = np.random.default_rng(2026072935)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, len(panel), len(panel))
        values[index] = pd.Series(x[selected]).corr(
            pd.Series(y[selected]), method="spearman"
        ) * direction
    values = values[np.isfinite(values)]
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), float(np.mean(values > 0))


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "RETURN_TO_TEXTURE_AND_DRAINAGE_ATTRIBUTE_REVIEW"
    ):
        raise RuntimeError("Parent gate does not authorize this audit")
    reaches, raster_sources = build_reach_attributes()
    static = pd.read_parquet(STATIC)
    stations = pd.read_csv(STATION_PANEL)[[
        "station_name", "reach_id", "observed_bfi",
        "observed_bfi_sensitivity_width",
    ]]
    if len(stations) != 97 or stations["reach_id"].nunique() != 97:
        raise RuntimeError("Expected exactly 97 distinct station reaches")
    graph = build_graph(static)
    panel = station_panel(stations, reaches, static, graph)
    correlations = correlation_table(panel)
    discovery = correlations.loc[
        correlations["subset"].eq("discovery_49")
    ].sort_values(
        ["absolute_spearman", "attribute"], ascending=[False, True]
    )
    selected_attribute = str(discovery.iloc[0]["attribute"])
    indexed = correlations.set_index(["subset", "attribute"])
    selected = {
        subset: float(indexed.loc[
            (subset, selected_attribute), "spearman"
        ])
        for subset in [
            "all_97", "discovery_49", "confirmation_48",
            "higher_sensitivity", "lower_sensitivity",
            "small_area", "medium_area", "large_area",
        ]
    }
    direction = float(np.sign(selected["discovery_49"]))
    confirmation = panel.loc[panel["split"].eq("confirmation_48")]
    boot_lower, boot_upper, boot_positive = bootstrap_aligned_correlation(
        confirmation, selected_attribute, direction
    )
    current_table = pd.read_csv(CURRENT_CORRELATIONS)
    current_rho = float(current_table.loc[
        current_table["subset"].eq("all_97")
        & current_table["scale"].eq("upstream_aw")
        & current_table["attribute"].eq("vertical_share_current"),
        "spearman",
    ].iloc[0])
    same_discovery_confirmation_direction = (
        np.sign(selected["discovery_49"])
        == np.sign(selected["confirmation_48"])
    )
    subgroup_names = [
        "higher_sensitivity", "lower_sensitivity",
        "small_area", "medium_area", "large_area",
    ]
    all_subgroups_same_direction = all(
        np.sign(selected[name]) == direction
        for name in subgroup_names
    )
    target_subgroups_material = all(
        abs(selected[name]) >= 0.10
        and np.sign(selected[name]) == direction
        for name in ["higher_sensitivity", "small_area"]
    )
    absolute_gain = abs(selected["all_97"]) - abs(current_rho)
    candidate_authorized = bool(
        abs(selected["discovery_49"]) >= 0.15
        and abs(selected["confirmation_48"]) >= 0.15
        and same_discovery_confirmation_direction
        and absolute_gain >= 0.05
        and boot_lower > 0
        and target_subgroups_material
        and all_subgroups_same_direction
    )
    reach_output = OUTPUTS / "reach_texture_drainage_attributes.parquet"
    reaches.to_parquet(reach_output, index=False)
    panel_output = REPORT / "station_texture_drainage_panel.csv"
    corr_output = REPORT / "texture_drainage_bfi_correlations.csv"
    panel.to_csv(panel_output, index=False, encoding="utf-8-sig")
    correlations.to_csv(corr_output, index=False, encoding="utf-8-sig")
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "soil_values_finite": bool(np.isfinite(
            reaches.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "station_count_97": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "discovery_confirmation_49_48": (
            panel["split"].value_counts().to_dict()
            == {"discovery_49": 49, "confirmation_48": 48}
        ),
        "shijiao_present": "石角站" in set(panel["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(panel["station_name"])
        ),
        "no_parameter_calibration": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    decision = (
        "AUTHORIZE_TEXTURE_DRAINAGE_REORDERING_TEST"
        if candidate_authorized
        else "DO_NOT_AUTHORIZE_TEXTURE_DRAINAGE_PROXY"
    )
    next_action = (
        "RUN_ONE_CONFIRMED_TEXTURE_DRAINAGE_MODEL"
        if candidate_authorized
        else "ACQUIRE_MISSING_SAND_CLAY_BULK_DENSITY_AND_DRAINAGE_ATTRIBUTES"
    )
    gate = {
        "run_id": "20260729_35",
        "phase": "texture_drainage_attribute_audit",
        "created_utc": utc_now(),
        "checks": checks,
        "selected_discovery_attribute": selected_attribute,
        "evidence_metrics": {
            "selected_correlations": selected,
            "current_vertical_share_full_spearman": current_rho,
            "selected_absolute_gain_over_current": absolute_gain,
            "same_discovery_confirmation_direction": bool(
                same_discovery_confirmation_direction
            ),
            "confirmation_aligned_bootstrap_95": [
                boot_lower, boot_upper,
            ],
            "confirmation_aligned_bootstrap_positive_fraction": (
                boot_positive
            ),
            "target_subgroups_material": bool(target_subgroups_material),
            "all_subgroups_same_direction": bool(
                all_subgroups_same_direction
            ),
            "proxy_is_actual_drainage_or_ksat": False,
        },
        "candidate_authorized": candidate_authorized,
        "decision": decision,
        "authorized_next_action": next_action,
        "parameters_calibrated": False,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 土壤粗颗粒与排水代理属性复核

## 结论

`{decision}`

- 发现集选中：`{selected_attribute}`；
- 发现集/确认集 Spearman：{selected['discovery_49']:+.3f} /
  {selected['confirmation_48']:+.3f}；
- 全97站 Spearman：{selected['all_97']:+.3f}；
- 相对当前垂向份额的绝对相关增益：{absolute_gain:+.3f}；
- 确认集方向对齐 bootstrap 95%区间：
  [{boot_lower:+.3f}, {boot_upper:+.3f}]；
- 高敏感组/小面积组：{selected['higher_sensitivity']:+.3f} /
  {selected['small_area']:+.3f}。

该代理不等同于实际排水等级或饱和导水率。

下一步：`{next_action}`
"""
    technical_path = REPORT / "technical_report.md"
    technical_path.write_text(report, encoding="utf-8")
    sources = [
        PARENT_GATE, STATIC, STATION_PANEL, CURRENT_CORRELATIONS,
        CATCHMENTS, CONTRACT, LITERATURE,
    ] + raster_sources
    products = [
        reach_output, panel_output, corr_output, gate_path, technical_path,
        LOGS / "gdal_zonal_commands.log",
    ] + sorted(OUTPUTS.glob("*_reach_zonal.geojson"))
    manifest = {
        "generated_utc": utc_now(),
        "sources": [
            record(path, "texture_drainage_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "texture_drainage_product", "derived")
            for path in products
        ],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        manifest["sources"] + manifest["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False, encoding="utf-8-sig",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "selected_attribute": selected_attribute,
        "discovery_spearman": selected["discovery_49"],
        "confirmation_spearman": selected["confirmation_48"],
        "full_spearman": selected["all_97"],
        "absolute_gain_over_current": absolute_gain,
        "bootstrap_95": [boot_lower, boot_upper],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
