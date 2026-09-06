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
REPORT = RUN / "reports" / "texture_residual_audit"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_38" / "reports"
    / "soilgrids_acquisition" / "gate.json"
)
RAW = (
    ROOT / "5_Test" / "20260729_38" / "inputs"
    / "soilgrids_prb_buffer"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
STATION_ATTRIBUTES = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit" / "station_bfi_attribute_panel.csv"
)
CURRENT_STATION = (
    ROOT / "5_Test" / "20260729_36" / "reports"
    / "cfvo_reordered_recharge" / "station_current_vs_cfvo.csv"
)
CFVO_PANEL = (
    ROOT / "5_Test" / "20260729_35" / "reports"
    / "texture_drainage_attribute_audit"
    / "station_texture_drainage_panel.csv"
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
PROPERTIES = ["sand", "clay", "bdod"]
CANDIDATES = [
    "sand_top_0_30_pct", "clay_top_0_30_pct",
    "bdod_top_0_30_g_cm3", "sand_deep_30_200_pct",
    "clay_deep_30_200_pct", "bdod_deep_30_200_g_cm3",
    "drainage_rank_top", "drainage_rank_deep",
    "drainage_rank_full", "drainage_rank_deep_minus_top",
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
        "path": str(path), "role": role, "semantic_state": state,
        "bytes": int(stat.st_size), "sha256": sha256(path),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }


def gdal_path() -> Path:
    path = Path(sys.prefix) / "Library" / "bin" / "gdal.exe"
    if not path.exists():
        raise RuntimeError("GDAL is missing in sparrow environment")
    return path


def zonal(path: Path, name: str) -> pd.DataFrame:
    output = OUTPUTS / f"{name}_reach_zonal.geojson"
    result = subprocess.run([
        str(gdal_path()), "raster", "zonal-stats", "--quiet",
        "--overwrite", "--zones", str(CATCHMENTS),
        "--include-field", "reach_id", "--stat", "mean", "--stat", "count",
        str(path), str(output),
    ], check=True, text=True, stdout=subprocess.PIPE,
       stderr=subprocess.STDOUT)
    with (LOGS / "gdal_zonal.log").open("a", encoding="utf-8") as log:
        log.write(result.stdout or "")
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = []
    for feature in payload["features"]:
        prop = feature["properties"]
        mean_key = next(
            key for key in prop
            if key.lower() == "mean" or key.lower().endswith("_mean")
        )
        count_key = next(
            key for key in prop
            if key.lower() == "count" or key.lower().endswith("_count")
        )
        rows.append({
            "reach_id": int(prop["reach_id"]),
            name: float(prop[mean_key]),
            f"{name}_cells": int(prop[count_key]),
        })
    frame = pd.DataFrame(rows).sort_values("reach_id")
    if len(frame) != 230 or frame[f"{name}_cells"].le(0).any():
        raise RuntimeError(f"Incomplete zonal statistics for {name}")
    return frame


def reach_attributes() -> tuple[pd.DataFrame, list[Path]]:
    tables = []
    sources = []
    for property_name in PROPERTIES:
        for _, _, label in DEPTHS:
            safe = label.replace("-", "_")
            path = (
                RAW / property_name
                / f"{property_name}_{label}_mean_prb_buffer.tif"
            )
            sources.append(path)
            tables.append(zonal(path, f"{property_name}_{safe}_raw"))
    reaches = tables[0]
    for table in tables[1:]:
        reaches = reaches.merge(
            table, on="reach_id", validate="one_to_one"
        )
    thickness = np.asarray(
        [bottom - top for top, bottom, _ in DEPTHS], dtype=float
    )
    top = np.arange(6) < 3
    deep = ~top
    for property_name in PROPERTIES:
        columns = [
            f"{property_name}_{label.replace('-', '_')}_raw"
            for _, _, label in DEPTHS
        ]
        values = reaches[columns].to_numpy(float)
        scale = 0.01 if property_name == "bdod" else 0.1
        values *= scale
        unit = "g_cm3" if property_name == "bdod" else "pct"
        reaches[f"{property_name}_top_0_30_{unit}"] = np.average(
            values[:, top], axis=1, weights=thickness[top]
        )
        reaches[f"{property_name}_deep_30_200_{unit}"] = np.average(
            values[:, deep], axis=1, weights=thickness[deep]
        )
        reaches[f"{property_name}_full_0_200_{unit}"] = np.average(
            values, axis=1, weights=thickness
        )
    for scale_name, suffix in [
        ("top", "top_0_30"), ("deep", "deep_30_200"),
        ("full", "full_0_200"),
    ]:
        sand = reaches[f"sand_{suffix}_pct"].rank(
            method="average", pct=True
        )
        low_clay = (-reaches[f"clay_{suffix}_pct"]).rank(
            method="average", pct=True
        )
        low_density = (-reaches[f"bdod_{suffix}_g_cm3"]).rank(
            method="average", pct=True
        )
        reaches[f"drainage_rank_{scale_name}"] = (
            sand + low_clay + low_density
        ) / 3.0
    reaches["drainage_rank_deep_minus_top"] = (
        reaches["drainage_rank_deep"]
        - reaches["drainage_rank_top"]
    )
    return reaches, sources


def graph_from(static: pd.DataFrame) -> nx.DiGraph:
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
        raise RuntimeError("Frozen topology is not a DAG")
    return graph


def build_station_panel(
    reaches: pd.DataFrame, static: pd.DataFrame
) -> pd.DataFrame:
    observed = pd.read_csv(STATION_ATTRIBUTES)[[
        "station_name", "reach_id", "observed_bfi",
        "observed_bfi_sensitivity_width",
        "upstream_incremental_area_km2",
    ]]
    current = pd.read_csv(CURRENT_STATION)[[
        "station_name", "simulated_bfi_current_combined"
    ]]
    cfvo = pd.read_csv(CFVO_PANEL)[[
        "station_name", "upstream_aw_cfvo_top_0_30_pct"
    ]]
    stations = observed.merge(
        current, on="station_name", validate="one_to_one"
    ).merge(cfvo, on="station_name", validate="one_to_one")
    stations["bfi_residual_current"] = (
        stations["observed_bfi"]
        - stations["simulated_bfi_current_combined"]
    )
    graph = graph_from(static)
    indexed = reaches.set_index("reach_id")
    area = static.set_index("reach_id")["inc_area_km2"]
    rows = []
    for station in stations.itertuples():
        outlet = int(station.reach_id)
        ids = sorted(nx.ancestors(graph, outlet) | {outlet})
        weights = area.loc[ids].to_numpy(float)
        row = {
            "station_name": station.station_name,
            "reach_id": outlet,
            "observed_bfi": float(station.observed_bfi),
            "simulated_bfi_current_combined": float(
                station.simulated_bfi_current_combined
            ),
            "bfi_residual_current": float(station.bfi_residual_current),
            "observed_bfi_sensitivity_width": float(
                station.observed_bfi_sensitivity_width
            ),
            "upstream_area_km2": float(
                station.upstream_incremental_area_km2
            ),
            "upstream_aw_cfvo_top_0_30_pct": float(
                station.upstream_aw_cfvo_top_0_30_pct
            ),
        }
        subset = indexed.loc[ids]
        for attribute in CANDIDATES:
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


def correlations(panel: pd.DataFrame) -> pd.DataFrame:
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
            residual_rho = subset["bfi_residual_current"].corr(
                subset[f"upstream_aw_{attribute}"], method="spearman"
            )
            total_rho = subset["observed_bfi"].corr(
                subset[f"upstream_aw_{attribute}"], method="spearman"
            )
            rows.append({
                "subset": subset_name, "attribute": attribute,
                "n": len(subset), "residual_spearman": residual_rho,
                "absolute_residual_spearman": abs(residual_rho),
                "observed_bfi_spearman": total_rho,
            })
    return pd.DataFrame(rows)


def bootstrap(
    frame: pd.DataFrame, attribute: str, direction: float
) -> tuple[float, float, float]:
    x = frame[f"upstream_aw_{attribute}"].to_numpy(float)
    y = frame["bfi_residual_current"].to_numpy(float)
    rng = np.random.default_rng(2026072939)
    values = np.empty(4000)
    for index in range(len(values)):
        selected = rng.integers(0, len(frame), len(frame))
        values[index] = pd.Series(x[selected]).corr(
            pd.Series(y[selected]), method="spearman"
        ) * direction
    values = values[np.isfinite(values)]
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), float(np.mean(values > 0))


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    (LOGS / "gdal_zonal.log").write_text("", encoding="utf-8")
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "AUDIT_DIRECT_TEXTURE_AND_MULTIPLE_DRAINAGE_CANDIDATES"
    ):
        raise RuntimeError("Parent gate does not authorize residual audit")
    reaches, raster_sources = reach_attributes()
    static = pd.read_parquet(STATIC)
    panel = build_station_panel(reaches, static)
    if len(panel) != 97 or "石角站" not in set(panel["station_name"]):
        raise RuntimeError("Expected 97 stations including Shijiao")
    if FIXED_EXCLUSIONS & set(panel["station_name"]):
        raise RuntimeError("A fixed exclusion re-entered")
    corr = correlations(panel)
    discovery = corr.loc[
        corr["subset"].eq("discovery_49")
    ].sort_values(
        ["absolute_residual_spearman", "attribute"],
        ascending=[False, True],
    )
    selected_attribute = str(discovery.iloc[0]["attribute"])
    indexed = corr.set_index(["subset", "attribute"])
    subset_names = [
        "all_97", "discovery_49", "confirmation_48",
        "higher_sensitivity", "lower_sensitivity",
        "small_area", "medium_area", "large_area",
    ]
    selected = {
        name: float(indexed.loc[
            (name, selected_attribute), "residual_spearman"
        ])
        for name in subset_names
    }
    direction = float(np.sign(selected["discovery_49"]))
    confirmation = panel.loc[panel["split"].eq("confirmation_48")]
    boot_lower, boot_upper, boot_positive = bootstrap(
        confirmation, selected_attribute, direction
    )
    cfvo_residual = {}
    for name in subset_names:
        if name == "all_97":
            subset = panel
        elif name in {"discovery_49", "confirmation_48"}:
            subset = panel.loc[panel["split"].eq(name)]
        elif name in {"higher_sensitivity", "lower_sensitivity"}:
            subset = panel.loc[panel["sensitivity_group"].eq(name)]
        else:
            subset = panel.loc[panel["area_group"].eq(name)]
        cfvo_residual[name] = float(
            subset["bfi_residual_current"].corr(
                subset["upstream_aw_cfvo_top_0_30_pct"],
                method="spearman",
            )
        )
    same_discovery_confirmation = (
        np.sign(selected["discovery_49"])
        == np.sign(selected["confirmation_48"])
    )
    subgroups = [
        "higher_sensitivity", "lower_sensitivity",
        "small_area", "medium_area", "large_area",
    ]
    all_subgroups_same_direction = all(
        np.sign(selected[name]) == direction for name in subgroups
    )
    high_sensitivity_material = (
        abs(selected["higher_sensitivity"]) >= 0.15
        and np.sign(selected["higher_sensitivity"]) == direction
    )
    incremental_gain_full = (
        abs(selected["all_97"]) - abs(cfvo_residual["all_97"])
    )
    incremental_gain_high = (
        abs(selected["higher_sensitivity"])
        - abs(cfvo_residual["higher_sensitivity"])
    )
    cfvo_redundancy = abs(panel[
        f"upstream_aw_{selected_attribute}"
    ].corr(
        panel["upstream_aw_cfvo_top_0_30_pct"], method="spearman"
    ))
    candidate_authorized = bool(
        abs(selected["discovery_49"]) >= 0.15
        and abs(selected["confirmation_48"]) >= 0.15
        and same_discovery_confirmation
        and abs(selected["all_97"]) >= 0.20
        and boot_lower > 0
        and high_sensitivity_material
        and all_subgroups_same_direction
        and max(incremental_gain_full, incremental_gain_high) >= 0.05
        and cfvo_redundancy < 0.95
    )
    reach_path = OUTPUTS / "reach_texture_drainage_candidates.parquet"
    reaches.to_parquet(reach_path, index=False)
    panel_path = REPORT / "station_texture_residual_panel.csv"
    corr_path = REPORT / "texture_residual_correlations.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "station_count_97": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "discovery_confirmation_49_48": (
            panel["split"].value_counts().to_dict()
            == {"discovery_49": 49, "confirmation_48": 48}
        ),
        "all_values_finite": bool(
            np.isfinite(
                reaches.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                panel.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
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
        "AUTHORIZE_TEXTURE_RESIDUAL_REORDERING_TEST"
        if candidate_authorized
        else "DO_NOT_AUTHORIZE_STATIC_TEXTURE_RESIDUAL_CONTROL"
    )
    next_action = (
        "RUN_ONE_CONFIRMED_TEXTURE_RESIDUAL_MODEL"
        if candidate_authorized
        else "STOP_STATIC_SOIL_RANKING_AND_SUMMARIZE_SERIES"
    )
    gate = {
        "run_id": "20260729_39",
        "phase": "texture_residual_attribute_audit",
        "created_utc": utc_now(),
        "checks": checks,
        "selected_discovery_attribute": selected_attribute,
        "evidence_metrics": {
            "selected_residual_correlations": selected,
            "cfvo_residual_correlations": cfvo_residual,
            "confirmation_aligned_bootstrap_95": [
                boot_lower, boot_upper,
            ],
            "confirmation_bootstrap_positive_fraction": boot_positive,
            "same_discovery_confirmation_direction": bool(
                same_discovery_confirmation
            ),
            "all_subgroups_same_direction": bool(
                all_subgroups_same_direction
            ),
            "high_sensitivity_material": bool(high_sensitivity_material),
            "incremental_gain_over_cfvo_full": incremental_gain_full,
            "incremental_gain_over_cfvo_high_sensitivity": (
                incremental_gain_high
            ),
            "absolute_spearman_with_cfvo": cfvo_redundancy,
            "candidate_is_measured_ksat": False,
        },
        "candidate_authorized": candidate_authorized,
        "decision": decision,
        "authorized_next_action": next_action,
        "model_run": False,
        "parameters_calibrated": False,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "series_terminal": not candidate_authorized,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = REPORT / "technical_report.md"
    report_path.write_text(f"""# 质地与排水候选残差审计

## 结论

`{decision}`

- 发现集选中：`{selected_attribute}`；
- 发现/确认残差Spearman：
  {selected['discovery_49']:+.3f} / {selected['confirmation_48']:+.3f}；
- 全97站/高敏感组：
  {selected['all_97']:+.3f} / {selected['higher_sensitivity']:+.3f}；
- 确认集方向对齐bootstrap 95%区间：
  [{boot_lower:+.3f}, {boot_upper:+.3f}]；
- 相对CFVO残差相关增益（全样本/高敏感）：
  {incremental_gain_full:+.3f} / {incremental_gain_high:+.3f}；
- 与CFVO属性绝对Spearman：{cfvo_redundancy:.3f}。

下一步：`{next_action}`
""", encoding="utf-8")
    sources = [
        PARENT_GATE, STATIC, STATION_ATTRIBUTES, CURRENT_STATION,
        CFVO_PANEL, CATCHMENTS, CONTRACT, LITERATURE,
    ] + raster_sources
    products = [
        reach_path, panel_path, corr_path, gate_path, report_path,
        LOGS / "gdal_zonal.log",
    ] + sorted(OUTPUTS.glob("*_reach_zonal.geojson"))
    provenance = {
        "run_id": "20260729_39", "created_utc": utc_now(),
        "sources": [
            record(path, "texture_residual_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "texture_residual_product", "derived")
            for path in products
        ],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False, encoding="utf-8-sig",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "selected_attribute": selected_attribute,
        "discovery_residual_spearman": selected["discovery_49"],
        "confirmation_residual_spearman": selected["confirmation_48"],
        "full_residual_spearman": selected["all_97"],
        "high_sensitivity_residual_spearman": (
            selected["higher_sensitivity"]
        ),
        "bootstrap_95": [boot_lower, boot_upper],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
