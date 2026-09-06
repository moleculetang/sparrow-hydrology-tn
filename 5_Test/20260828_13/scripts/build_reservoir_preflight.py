"""Build the 20260828_13 reservoir inventory and observation-coverage audit.

This script performs no parameter fitting.  A nearest Reach is a spatial
diagnostic only; it is never promoted to an authoritative reservoir control
edge unless an explicit override says so.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_13"
DATASET = (
    ROOT
    / "0_hydro_sediment_data"
    / "reservoir"
    / "2010-2021年中国338座水库高分辨率水位与蓄水量变化数据集"
)
ATTRIBUTES = DATASET / "01 res_loc" / "01 res_loc" / "reservoir attributes.xlsx"
CATCHMENTS = ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
REACHES = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
OVERRIDES = STAGE / "inputs" / "manual_reservoir_topology_overrides.csv"
RESERVOIR_EVIDENCE = STAGE / "inputs" / "reservoir_specific_evidence.csv"
VERIFIED_RESERVOIR_EVIDENCE = STAGE / "inputs" / "reservoir_verified_evidence.csv"
METHOD_REGISTRY = STAGE / "inputs" / "method_literature_registry.csv"
DISCHARGE_REGISTRY = (
    ROOT
    / "1_Inputs"
    / "DischargeData"
    / "registry"
    / "discharge_station_year_registry.xlsx"
)
CANONICAL_DAILY = (
    ROOT
    / "5_Test"
    / "20260828_9"
    / "outputs"
    / "canonical_reach_daily_2006_2024.parquet"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def semicolon_ints(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(part) for part in str(value).split(";")]


def strict_bool(value: object) -> bool:
    """Parse an explicit CSV boolean and reject ambiguous authorization text."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected an explicit boolean, got {value!r}")


def reach_area_for_entity(
    reaches_by_id: pd.DataFrame, inflow_reaches: list[int], body_reaches: list[int]
) -> tuple[float | None, str]:
    """Return the model area represented at a proposed reservoir control point.

    A single terminal inflow Reach uses its routed contributing area.  A
    multi-arm entity sums the arms because neither arm includes the other.
    When no control edge is authorized (for example Baipenzhu), the body Reach
    area is retained only as a diagnostic and never promoted to an operator.
    """

    selected = inflow_reaches or body_reaches
    if not selected:
        return None, "unavailable"
    missing = sorted(set(selected).difference(reaches_by_id.index))
    if missing:
        raise ValueError(f"Registered Reach IDs are absent from topology: {missing}")
    areas = reaches_by_id.loc[selected, "tot_km2"].astype(float)
    if len(selected) == 1:
        return float(areas.iloc[0]), (
            "registered_terminal_reach_tot_km2"
            if inflow_reaches
            else "body_reach_tot_km2_diagnostic_only"
        )
    return float(areas.sum()), (
        "sum_of_registered_parallel_arm_tot_km2"
        if inflow_reaches
        else "sum_of_body_reach_tot_km2_diagnostic_only"
    )


def projected_position_on_reach(
    dam_geometry: object, reach_geometry: object
) -> tuple[float | None, float | None]:
    """Return line-order fraction and distance to the nearer Reach endpoint.

    The fraction follows stored geometry vertex order; it is intentionally not
    labelled upstream/downstream unless the line orientation is independently
    registered.  The endpoint distance is sufficient to flag an interior dam.
    """

    length = float(reach_geometry.length)
    if not np.isfinite(length) or length <= 0:
        return None, None
    fraction = float(reach_geometry.project(dam_geometry) / length)
    return fraction, float(min(fraction, 1.0 - fraction) * length)


def validate_authorized_topology(
    inventory: pd.DataFrame, reaches_by_id: pd.DataFrame
) -> None:
    """Reject duplicate or cyclic control edges among authorized reservoirs."""

    authorized = inventory[inventory["operator_authorized"].fillna(False)].copy()
    owners: dict[int, str] = {}
    graph: dict[str, str | None] = {}
    normalized: list[tuple[str, list[int], int]] = []
    for row in authorized.itertuples(index=False):
        entity = str(row.reservoir_entity_id)
        inflows = semicolon_ints(row.registered_operator_inflow_reaches)
        if not inflows or pd.isna(row.registered_unique_outflow_reach):
            raise ValueError(f"Authorized reservoir lacks a complete control edge: {entity}")
        outflow = int(row.registered_unique_outflow_reach)
        if outflow in inflows:
            raise ValueError(f"Reservoir routes to one of its own controlled Reaches: {entity}")
        for reach in inflows:
            actual_downstream = reaches_by_id.loc[reach, "down_rch"]
            if pd.isna(actual_downstream) or int(actual_downstream) != outflow:
                raise ValueError(
                    f"Authorized control edge {reach}->{outflow} disagrees with "
                    f"the locked topology ({actual_downstream}) for {entity}"
                )
            if reach in owners:
                raise ValueError(
                    f"Reach {reach} is controlled by both {owners[reach]} and {entity}"
                )
            owners[reach] = entity
        normalized.append((entity, inflows, outflow))
    for entity, _, outflow in normalized:
        graph[entity] = owners.get(outflow)
    for start in graph:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"Authorized reservoir graph contains a cycle at {current}")
            seen.add(current)
            current = graph.get(current)


def find_files_for_id(grand_id: int, section: str) -> list[Path]:
    directory = DATASET / section
    pattern = re.compile(rf"(?:^|_){grand_id}(?:\.|_)", re.IGNORECASE)
    return sorted(
        p
        for p in directory.rglob("*.csv")
        if not p.name.startswith("._") and pattern.search(p.name)
    )


def date_coverage(path: Path) -> tuple[str | None, str | None, int | None]:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return None, None, None
    date = None
    if "dates" in frame.columns:
        date = pd.to_datetime(frame["dates"], errors="coerce")
    elif {"year", "month", "day"}.issubset(frame.columns):
        date = pd.to_datetime(
            frame[["year", "month", "day"]].rename(
                columns={"year": "year", "month": "month", "day": "day"}
            ),
            errors="coerce",
        )
    elif {"year", "month"}.issubset(frame.columns):
        date = pd.to_datetime(
            {"year": frame["year"], "month": frame["month"], "day": 1},
            errors="coerce",
        )
    if date is None or date.notna().sum() == 0:
        return None, None, int(len(frame))
    return (
        date.min().date().isoformat(),
        date.max().date().isoformat(),
        int(date.notna().sum()),
    )


def choose_provisional(paths: list[Path], kind: str) -> Path | None:
    if not paths:
        return None
    if kind == "swa":
        exact = [p for p in paths if p.name.lower().startswith("swa_")]
        return exact[0] if exact else paths[0]
    if kind == "wse":
        enhanced = [p for p in paths if "Enhanced Measurement" in str(p)]
        return enhanced[0] if enhanced else paths[0]
    # Multiple RWSC algorithms exist.  This is only a coverage representative,
    # not a registered modeling choice.
    em = [p for p in paths if "WSE-SWA" in str(p) and "\\rwsc\\" in str(p)]
    return em[0] if em else paths[0]


def normalized_station(value: object) -> str:
    text = str(value)
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("站", "")
        .replace(" ", "")
        .lower()
    )


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    attributes = pd.read_excel(ATTRIBUTES)
    points = gpd.GeoDataFrame(
        attributes.copy(),
        geometry=gpd.points_from_xy(attributes["LONG"], attributes["LAT"]),
        crs="EPSG:4326",
    )
    catchments = gpd.read_file(CATCHMENTS)
    reaches = gpd.read_file(REACHES)
    if catchments.crs != reaches.crs:
        raise RuntimeError("Catchment and Reach CRS differ")
    if reaches["reach_id"].duplicated().any():
        raise RuntimeError("Reach topology contains duplicate reach_id values")
    reaches_by_id = reaches.set_index("reach_id", drop=False)
    domain = gpd.sjoin(
        points.to_crs(catchments.crs),
        catchments[["reach_id", "geometry"]],
        how="inner",
        predicate="within",
    ).rename(columns={"reach_id": "containing_incremental_catchment_reach"})
    if domain["GRAND_ID"].duplicated().any():
        raise RuntimeError("A dam point intersects multiple incremental catchments")

    reach_rows: list[dict[str, object]] = []
    for _, dam in domain.iterrows():
        distance = reaches.geometry.distance(dam.geometry)
        order = np.argsort(distance.to_numpy())[:5]
        for rank, pos in enumerate(order, start=1):
            reach = reaches.iloc[int(pos)]
            line_fraction, endpoint_distance = projected_position_on_reach(
                dam.geometry, reach.geometry
            )
            reach_rows.append(
                {
                    "grand_id": int(dam["GRAND_ID"]),
                    "dam_name": dam["DAM_NAME"],
                    "candidate_rank": rank,
                    "reach_id": int(reach["reach_id"]),
                    "reach_name": reach["src_id"],
                    "downstream_reach": (
                        None if pd.isna(reach["down_rch"]) else int(reach["down_rch"])
                    ),
                    "distance_m": float(distance.iloc[int(pos)]),
                    "dam_projected_fraction_from_geometry_start": line_fraction,
                    "dam_projected_distance_to_nearest_endpoint_m": endpoint_distance,
                    "authoritative": False,
                    "use": "spatial_review_only",
                }
            )
    reach_candidates = pd.DataFrame(reach_rows)

    overrides = pd.read_csv(OVERRIDES)
    required_override_columns = {
        "waterbody_type",
        "body_reaches",
        "operator_inflow_reaches",
        "unique_outflow_reach",
        "active_period_status",
        "operator_authorized",
        "fit_eligible_2010_2018",
        "extension_operator_eligible",
        "operator_start",
        "local_capture_fraction",
        "domain_storage_scale",
    }
    missing_override_columns = required_override_columns.difference(overrides.columns)
    if missing_override_columns:
        raise RuntimeError(
            f"Topology override registry is missing columns: {sorted(missing_override_columns)}"
        )
    if overrides["reservoir_name_en"].duplicated().any():
        raise RuntimeError("Topology override registry contains duplicate reservoir entities")
    override_by_id = {
        int(row.grand_id): row
        for row in overrides.itertuples(index=False)
        if pd.notna(row.grand_id)
    }
    inventory_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for _, dam in domain.sort_values("CAP_REP", ascending=False).iterrows():
        grand_id = int(dam["GRAND_ID"])
        nearest = reach_candidates[
            (reach_candidates["grand_id"] == grand_id)
            & (reach_candidates["candidate_rank"] == 1)
        ].iloc[0]
        override = override_by_id.get(grand_id)
        if override is None:
            body_reaches: list[int] = []
            inflow_reaches: list[int] = []
            unique_outflow = None
            topology_status = "candidate_nearest_reach_manual_review_required"
            topology_evidence = "GRanD_dam_point_and_nearest_reach_only"
            reservoir_name_zh = None
            waterbody_type = "artificial_reservoir_candidate"
            entity_status = "constructed_candidate"
            active_period_status = "not_registered"
            operator_authorized = False
            fit_eligible = False
            extension_eligible = False
            operator_start = None
            local_capture_fraction = 1.0
            domain_storage_scale = 1.0
            notes = "No authoritative control edge registered"
        else:
            body_reaches = semicolon_ints(override.body_reaches)
            inflow_reaches = semicolon_ints(override.operator_inflow_reaches)
            unique_outflow = (
                None
                if pd.isna(override.unique_outflow_reach)
                else int(override.unique_outflow_reach)
            )
            topology_status = override.topology_status
            topology_evidence = override.topology_evidence
            reservoir_name_zh = override.reservoir_name_zh
            waterbody_type = override.waterbody_type
            entity_status = override.entity_status
            active_period_status = override.active_period_status
            operator_authorized = strict_bool(override.operator_authorized)
            fit_eligible = strict_bool(override.fit_eligible_2010_2018)
            extension_eligible = strict_bool(override.extension_operator_eligible)
            operator_start = (
                None if pd.isna(override.operator_start) else str(override.operator_start)
            )
            local_capture_fraction = (
                1.0
                if pd.isna(override.local_capture_fraction)
                else float(override.local_capture_fraction)
            )
            domain_storage_scale = (
                1.0
                if pd.isna(override.domain_storage_scale)
                else float(override.domain_storage_scale)
            )
            notes = override.notes
            authoritative_mask = (
                (reach_candidates["grand_id"] == grand_id)
                & reach_candidates["reach_id"].isin(inflow_reaches)
            )
            if operator_authorized:
                reach_candidates.loc[authoritative_mask, "authoritative"] = True
                reach_candidates.loc[authoritative_mask, "use"] = (
                    "authorized_operator_inflow"
                )
            else:
                reach_candidates.loc[authoritative_mask, "use"] = (
                    "registered_but_blocked_operator_inflow"
                )

        nearest_distance = float(nearest["distance_m"])
        if nearest_distance <= 1000.0:
            nearest_quality = "high_within_1km"
        elif nearest_distance <= 5000.0:
            nearest_quality = "moderate_1_to_5km"
        else:
            nearest_quality = "low_beyond_5km_network_representation_gap"
        main_use = None if pd.isna(dam["MAIN_USE"]) else str(dam["MAIN_USE"]).strip()
        if not main_use or main_use.lower() == "nan":
            main_use = None
        model_area_km2, model_area_basis = reach_area_for_entity(
            reaches_by_id, inflow_reaches, body_reaches
        )
        grand_area_km2 = None if pd.isna(dam["CATCH_SKM"]) else float(dam["CATCH_SKM"])
        model_to_grand_area_ratio = (
            None
            if model_area_km2 is None or grand_area_km2 is None or grand_area_km2 <= 0
            else model_area_km2 / grand_area_km2
        )
        nearest_fraction = float(nearest["dam_projected_fraction_from_geometry_start"])
        nearest_endpoint_distance = float(
            nearest["dam_projected_distance_to_nearest_endpoint_m"]
        )
        interior_dam_split_required = bool(
            topology_status in {
                "reach_split_or_virtual_dam_required",
                "reach_split_or_subreach_partition_required",
            }
            or (
                not operator_authorized
                and nearest_distance <= 1000.0
                and nearest_endpoint_distance > 0.05 * float(
                    reaches_by_id.loc[int(nearest["reach_id"]), "geometry"].length
                )
                and grand_id == 5758
            )
        )
        operator_ready = bool(
            operator_authorized
            and fit_eligible
            and waterbody_type == "artificial_reservoir"
            and entity_status == "constructed"
            and active_period_status != "not_registered"
        )
        operator_ready_for_extension = bool(
            operator_authorized
            and extension_eligible
            and waterbody_type == "artificial_reservoir"
            and entity_status == "constructed"
            and active_period_status != "not_registered"
        )
        priority_tier = (
            "priority_13_artificial_reservoirs"
            if override is not None and waterbody_type == "artificial_reservoir"
            else "supplementary_21_candidate_dams"
        )

        inventory_rows.append(
            {
                "reservoir_entity_id": f"GRAND_{grand_id}",
                "grand_id": grand_id,
                "reservoir_name_en": dam["DAM_NAME"],
                "reservoir_name_zh": reservoir_name_zh,
                "priority_tier": priority_tier,
                "waterbody_type": waterbody_type,
                "entity_status": entity_status,
                "longitude": float(dam["LONG"]),
                "latitude": float(dam["LAT"]),
                "commissioning_year_grand": (
                    None if pd.isna(dam["YEAR"]) else int(dam["YEAR"])
                ),
                "dam_height_m": (
                    None if pd.isna(dam["DAM_HGT_M"]) else float(dam["DAM_HGT_M"])
                ),
                "surface_area_km2_grand": (
                    None if pd.isna(dam["AREA_SKM"]) else float(dam["AREA_SKM"])
                ),
                "capacity_million_m3_grand": (
                    None if pd.isna(dam["CAP_REP"]) else float(dam["CAP_REP"])
                ),
                "catchment_km2_grand": grand_area_km2,
                "model_control_area_km2": model_area_km2,
                "model_control_area_basis": model_area_basis,
                "model_to_grand_catchment_area_ratio": model_to_grand_area_ratio,
                "main_use_grand": main_use,
                "containing_incremental_catchment_reach": int(
                    dam["containing_incremental_catchment_reach"]
                ),
                "nearest_reach_id": int(nearest["reach_id"]),
                "nearest_reach_distance_m": nearest_distance,
                "nearest_reach_quality": nearest_quality,
                "dam_projected_fraction_from_nearest_reach_geometry_start": nearest_fraction,
                "dam_projected_distance_to_nearest_reach_endpoint_m": nearest_endpoint_distance,
                "interior_dam_split_required": interior_dam_split_required,
                "registered_body_reaches": ";".join(map(str, body_reaches)) or None,
                "registered_operator_inflow_reaches": ";".join(map(str, inflow_reaches)) or None,
                "registered_unique_outflow_reach": unique_outflow,
                "topology_evidence": topology_evidence,
                "topology_status": topology_status,
                "active_period_status": active_period_status,
                "operator_authorized": operator_authorized,
                "fit_eligible_2010_2018": fit_eligible,
                "extension_operator_eligible": extension_eligible,
                "operator_start": operator_start,
                "local_capture_fraction": local_capture_fraction,
                "domain_storage_scale": domain_storage_scale,
                "operator_ready_for_formal_fit": operator_ready,
                "operator_ready_for_extension": operator_ready_for_extension,
                "notes": notes,
            }
        )

        file_groups = {
            "wse": find_files_for_id(grand_id, "02 res_wse"),
            "swa": find_files_for_id(grand_id, "03 res_swa"),
            "rwsc": find_files_for_id(grand_id, "04 res_rwsc"),
        }
        coverage = {"grand_id": grand_id, "dam_name": dam["DAM_NAME"]}
        for kind, paths in file_groups.items():
            selected = choose_provisional(paths, kind)
            start, end, rows = (None, None, None) if selected is None else date_coverage(selected)
            coverage.update(
                {
                    f"{kind}_file_count": len(paths),
                    f"{kind}_available": bool(paths),
                    f"{kind}_representative_path": None if selected is None else str(selected),
                    f"{kind}_representative_start": start,
                    f"{kind}_representative_end": end,
                    f"{kind}_representative_rows": rows,
                    f"{kind}_selection_status": (
                        "absent"
                        if selected is None
                        else "coverage_representative_not_model_lock"
                    ),
                }
            )
        coverage_rows.append(coverage)

    inventory = pd.DataFrame(inventory_rows)

    # Add locally known entities missing from the 338-reservoir subset.  Their
    # physical attributes remain null until an auditable source is registered.
    for row in overrides[overrides["grand_id"].isna()].itertuples(index=False):
        body_reaches = semicolon_ints(row.body_reaches)
        inflow_reaches = semicolon_ints(row.operator_inflow_reaches)
        model_area_km2, model_area_basis = reach_area_for_entity(
            reaches_by_id, inflow_reaches, body_reaches
        )
        operator_authorized = strict_bool(row.operator_authorized)
        fit_eligible = strict_bool(row.fit_eligible_2010_2018)
        extension_eligible = strict_bool(row.extension_operator_eligible)
        inventory.loc[len(inventory)] = {
            "reservoir_entity_id": f"LOCAL_{row.reservoir_name_en.upper().replace(' ', '_')}",
            "grand_id": None,
            "reservoir_name_en": row.reservoir_name_en,
            "reservoir_name_zh": row.reservoir_name_zh,
            "priority_tier": (
                "priority_13_artificial_reservoirs"
                if row.waterbody_type == "artificial_reservoir"
                else "excluded_natural_lake"
            ),
            "waterbody_type": row.waterbody_type,
            "entity_status": row.entity_status,
            "longitude": 115.35 if row.reservoir_name_en == "Fengshuba" else None,
            "latitude": 24.416667 if row.reservoir_name_en == "Fengshuba" else None,
            "commissioning_year_grand": None,
            "dam_height_m": None,
            "surface_area_km2_grand": None,
            "capacity_million_m3_grand": None,
            "catchment_km2_grand": None,
            "model_control_area_km2": model_area_km2,
            "model_control_area_basis": model_area_basis,
            "model_to_grand_catchment_area_ratio": None,
            "main_use_grand": None,
            "containing_incremental_catchment_reach": None,
            "nearest_reach_id": None,
            "nearest_reach_distance_m": None,
            "nearest_reach_quality": None,
            "dam_projected_fraction_from_nearest_reach_geometry_start": None,
            "dam_projected_distance_to_nearest_reach_endpoint_m": None,
            "interior_dam_split_required": False,
            "registered_body_reaches": row.body_reaches,
            "registered_operator_inflow_reaches": row.operator_inflow_reaches,
            "registered_unique_outflow_reach": row.unique_outflow_reach,
            "topology_evidence": row.topology_evidence,
            "topology_status": row.topology_status,
            "active_period_status": row.active_period_status,
            "operator_authorized": operator_authorized,
            "fit_eligible_2010_2018": fit_eligible,
            "extension_operator_eligible": extension_eligible,
            "operator_start": None if pd.isna(row.operator_start) else str(row.operator_start),
            "local_capture_fraction": (
                1.0 if pd.isna(row.local_capture_fraction) else float(row.local_capture_fraction)
            ),
            "domain_storage_scale": (
                1.0 if pd.isna(row.domain_storage_scale) else float(row.domain_storage_scale)
            ),
            "operator_ready_for_formal_fit": bool(
                operator_authorized
                and fit_eligible
                and row.waterbody_type == "artificial_reservoir"
                and row.entity_status == "constructed"
                and row.active_period_status != "not_registered"
            ),
            "operator_ready_for_extension": bool(
                operator_authorized
                and extension_eligible
                and row.waterbody_type == "artificial_reservoir"
                and row.entity_status == "constructed"
                and row.active_period_status != "not_registered"
            ),
            "notes": row.notes,
        }

    coverage = pd.DataFrame(coverage_rows)
    validate_authorized_topology(inventory, reaches_by_id)
    priority_audit = inventory[
        inventory["priority_tier"].isin(
            ["priority_13_artificial_reservoirs", "excluded_natural_lake"]
        )
    ].copy()
    priority_artificial = priority_audit[
        priority_audit["priority_tier"] == "priority_13_artificial_reservoirs"
    ].copy()
    if (
        len(priority_artificial) != 13
    ):
        raise RuntimeError("Expected exactly 13 priority artificial reservoirs")

    station_summary = pd.read_excel(DISCHARGE_REGISTRY, sheet_name="station_summary")
    reservoir_station = station_summary[
        station_summary["station"].astype(str).str.contains(
            "水库|坝下|大坝|渠道|溢洪道|隧洞口", regex=True, na=False
        )
    ].copy()
    reservoir_station["normalized_station"] = reservoir_station["station"].map(
        normalized_station
    )
    reservoir_station["source_registry_has_complete_2010_2022"] = reservoir_station[
        "has_complete_2010_2022"
    ].astype(bool)
    missing_year_text = reservoir_station["missing_2010_2022_years"].fillna("").astype(str).str.strip()
    reservoir_station["audit_covers_2010_2022_all_years"] = (
        reservoir_station["complete_2010_2022_year_count"].fillna(0).astype(int) >= 13
    ) & missing_year_text.eq("")
    reservoir_station["registry_complete_flag_disagrees_with_audit"] = (
        reservoir_station["source_registry_has_complete_2010_2022"]
        != reservoir_station["audit_covers_2010_2022_all_years"]
    )
    reservoir_station["registered_use"] = "candidate_release_or_diversion_observation"
    reservoir_station["caution"] = (
        "station name alone cannot distinguish total release, power release, spill, or diversion"
    )

    inventory.to_parquet(outputs / "reservoir_entity_inventory.parquet", index=False)
    priority_audit.to_parquet(
        outputs / "priority_reservoir_topology_audit.parquet", index=False
    )
    reach_candidates.to_parquet(outputs / "dam_reach_candidate_mapping.parquet", index=False)
    coverage.to_parquet(outputs / "reservoir_remote_observation_coverage.parquet", index=False)
    reservoir_station.to_parquet(outputs / "reservoir_discharge_station_coverage.parquet", index=False)

    canonical_schema = pq.read_schema(CANONICAL_DAILY).names
    summary = {
        "stage": "20260828_13",
        "status": "PREFLIGHT_COMPLETE_FORMAL_FIT_NOT_AUTHORIZED",
        "counts": {
            "grand_dams_inside_230_incremental_catchments": int(domain["GRAND_ID"].nunique()),
            "local_special_entities_added": int(overrides["grand_id"].isna().sum()),
            "priority_artificial_reservoirs": int(
                (inventory["priority_tier"] == "priority_13_artificial_reservoirs").sum()
            ),
            "supplementary_candidate_dams": int(
                (inventory["priority_tier"] == "supplementary_21_candidate_dams").sum()
            ),
            "constructed_or_candidate_entities": int(
                inventory["entity_status"].astype(str).str.contains("constructed").sum()
            ),
            "natural_lakes_excluded": int((inventory["entity_status"] == "natural_lake").sum()),
            "confirmed_or_partly_confirmed_topologies": int(
                inventory["topology_status"].astype(str).str.contains("confirmed").sum()
            ),
            "manual_topology_reviews": int(
                inventory["topology_status"].astype(str).str.contains("review").sum()
            ),
            "operator_ready_entities": int(
                inventory["operator_ready_for_formal_fit"].fillna(False).sum()
            ),
            "priority_operator_authorized": int(
                priority_artificial["operator_authorized"].fillna(False).sum()
            ),
            "priority_operator_ready": int(
                priority_artificial["operator_ready_for_formal_fit"].fillna(False).sum()
            ),
            "priority_fit_ineligible_extension_only": int(
                (
                    priority_artificial["operator_authorized"].fillna(False)
                    & ~priority_artificial["operator_ready_for_formal_fit"].fillna(False)
                    & priority_artificial["operator_ready_for_extension"].fillna(False)
                ).sum()
            ),
            "priority_operator_unresolved": int(
                (~priority_artificial["operator_authorized"].fillna(False)).sum()
            ),
            "interior_dam_split_required": int(
                inventory["interior_dam_split_required"].fillna(False).sum()
            ),
            "catchment_area_review_required": int(
                inventory["topology_status"]
                .astype(str)
                .str.contains("catchment_area_review_required")
                .sum()
            ),
            "dam_points_beyond_5km_from_modeled_stream": int(
                inventory["nearest_reach_quality"]
                .astype(str)
                .str.contains("beyond_5km")
                .sum()
            ),
            "candidate_reservoir_related_discharge_station_labels": int(len(reservoir_station)),
            "reservoir_station_complete_flag_disagreements": int(
                reservoir_station["registry_complete_flag_disagrees_with_audit"].sum()
            ),
        },
        "remote_coverage": {
            "wse": int(coverage["wse_available"].sum()),
            "swa": int(coverage["swa_available"].sum()),
            "rwsc": int(coverage["rwsc_available"].sum()),
            "selection_warning": "representative paths audit coverage only; no RWSC method is locked",
        },
        "input_hashes": {
            "reservoir_attributes": sha256(ATTRIBUTES),
            "canonical_daily": sha256(CANONICAL_DAILY),
            "manual_topology_overrides": sha256(OVERRIDES),
            "reservoir_specific_evidence": sha256(RESERVOIR_EVIDENCE),
            "reservoir_verified_evidence": sha256(VERIFIED_RESERVOIR_EVIDENCE),
            "method_literature_registry": sha256(METHOD_REGISTRY),
        },
        "canonical_daily_schema": canonical_schema,
        "routing_contract": (
            "Reservoir scenarios must restart from local_fast_response_m3_s and "
            "local_slow_response_m3_s and reroute the complete network. Applying a "
            "reservoir operator to already accumulated routed flow is forbidden "
            "because it double-counts unregulated upstream water."
        ),
        "hard_boundary": (
            "Nearest-Reach results are not authoritative. The 32 in-domain GRanD "
            "points are candidates, not 32 authorized storage states. Formal fitting "
            "remains closed until every included reservoir has one entity, one release "
            "location, a valid activation period, non-duplicated cascade routing and "
            "registered parameter priors."
        ),
    }
    (reports / "reservoir_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 20260828_13 水库证据与拓扑预审计

状态：`PREFLIGHT_COMPLETE_FORMAL_FIT_NOT_AUTHORIZED`。

## 结论

- 338库数据中有 **{summary['counts']['grand_dams_inside_230_incremental_catchments']}** 座坝点落入当前230个增量汇水区。
- 旧拓扑还提供枫树坝、大藤峡和抚仙湖三个本地实体；抚仙湖是天然湖泊，已从普通调度水库算子中排除。
- 正式优先清单是 **{summary['counts']['priority_artificial_reservoirs']}** 座人工水库；其余 **{summary['counts']['supplementary_candidate_dams']}** 座GRanD坝点只是补充候选。现有空间资料不是“珠江全部水库名录”，小型水库的合并表示必须另行注册。
- 最近Reach仅用于人工复核。新丰江、枫树坝等跨多个水库命名Reach的实体只能保留一个库存状态和一个出流位置。
- 当前13座优先人工库均已通过控制边与启用期预授权；其中 **{summary['counts']['priority_operator_ready']}** 座可用于2010–2018开发拟合，**{summary['counts']['priority_fit_ineligible_extension_only']}** 座（大藤峡）仅可在其2020年启用后作为锁定扩展算子运行，未解决算子数为 **{summary['counts']['priority_operator_unresolved']}**。另有 **{summary['counts']['dam_points_beyond_5km_from_modeled_stream']}** 个补充候选GRanD坝点距现有简化河网超过5 km，不能靠最近线自动插入。

白盆珠坝点虽距Reach 17仅约597 m，但位于143.4 km长Reach的约39.3%位置，坝址GRanD汇水面积852 km²仅为该模型Reach约3063 km²的约28%。因此不能把整个Reach 17直接送入水库；正式建模前必须拆分河段或显式划分坝上/坝下局地产流。

龟石和澄碧河的模型控制面积/GRanD坝址面积分别约为0.56和1.74，继续标记 `catchment_area_review_required`；岩滩的Reach 153与154只能共享一个库存并向Reach 146释放，继续标记 `shared_storage_edge_review_required`。长洲、龙滩和大藤峡必须锁定模拟期内的实际启用日期，投运前严格旁通。

## 路由位置合同

水库情景必须从发布产品中的 `local_fast_response_m3_s` 与 `local_slow_response_m3_s` 重新沿完整拓扑汇流，并在唯一控制边更新一次库存。已经累计过上游水量的 `routed_*` 只能作为父模型对照，禁止再次经过水库；否则会把未调节上游流量重复加入。

## 数据覆盖

- 水位候选：{summary['remote_coverage']['wse']}/32；
- 月水面面积：{summary['remote_coverage']['swa']}/32；
- 月蓄水变化候选：{summary['remote_coverage']['rwsc']}/32；
- 水库、坝下、大坝、渠道等站名候选：{summary['counts']['candidate_reservoir_related_discharge_station_labels']}个标签。
- 源registry的“完整2010–2022”布尔值与按缺失年份重算结果不一致：{summary['counts']['reservoir_station_complete_flag_disagreements']}个标签；正式分类使用重算字段，不能再要求年份计数恰好等于13。

这些标签不能自动等同于水库总下泄；渠道、发电、溢洪和坝下断面必须逐站判别。遥感蓄水变化存在多种算法版本，本轮只审计覆盖，不锁定某个版本。

本地 `珠江水情水库.xlsx` 另含9座水库、2023-11-06至2026-06-18共6994行运行信息，但没有任何2006–2022记录，不能进入本轮拟合或严格留出。该表在2025-05-30发生未声明的列结构变化，并有53个可疑陈旧快照日期；修复后只能作为2023年以后锁模诊断。详细合同见 `pearl_river_reservoir_workbook_audit.json`。

## 文献与先验边界

用户给出的新丰江、枫树坝、白盆珠库容、水位、调节类型和汛期已经进入逐字段证据登记，但在独立来源核验前统一标为 `U_user_synthesis_unverified`，不能用于拟合。即使核验后，正常、汛限和死水位在缺少高程–库容曲线时也只能约束季节相位与高低顺序，不能线性转换为库存比例。逐库文献事实将转换成低维参数先验及宽度，不会变成固定的12个月下泄表。

2006–2009在发布父产品中明确标为状态预热期，只能用于库存初值收敛；正式可评分开发期为2010–2018。2019–2022在结构与参数锁定前不得读取，2023–2024只允许锁模后延伸。

## 正式建模前剩余硬门

1. 每座纳入模型的水库必须确定唯一实体、入流边集合和唯一出流Reach；
2. 大藤峡等新建水库必须按实际投运日期启用，不能影响投运前模拟；
3. 调度规程、汛限水位和功能只进入带证据等级的先验；
4. 缺少高程–库容曲线时，水位只约束季节相位与顺序，不能线性换算库容；
5. 关闭全部水库时，必须从局地快慢水逐行精确恢复canonical父模型；
6. 2019–2022观测不得用于参数、先验宽度或规则选择。
"""
    (reports / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
