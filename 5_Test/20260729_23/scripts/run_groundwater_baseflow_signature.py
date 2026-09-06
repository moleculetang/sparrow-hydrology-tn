from __future__ import annotations

import calendar
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from runtime_environment import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "groundwater_baseflow_gate"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
CONFIG = RUN / "config" / "groundwater_baseflow_contract.json"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_22" / "reports"
    / "et_evidence_audit" / "gate.json"
)
PARAMETERS = (
    ROOT / "5_Test" / "20260729_21" / "inputs"
    / "reach_attribute_parameter_map_gamma_0_5.parquet"
)
FORCING = (
    ROOT / "5_Test" / "20260729_9" / "inputs"
    / "reach_month_forcing_2006_2018.parquet"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
STATION_CLASSIFICATION = (
    ROOT / "5_Test" / "20260729_7" / "reports"
    / "q72_stable_process_diagnosis" / "stable_station_classification.csv"
)
DAILY_ROOTS = [
    ROOT / "1_Inputs" / "DischargeData" / "complete_2010_2022",
    ROOT / "1_Inputs" / "DischargeData" / "noncomplete_2010_2022",
]
MONTH_COLUMNS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
EPS = 1e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def prepare_network(static: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming = defaultdict(list)
    for row in static.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
    upstream_ids = {}
    for row in static.itertuples():
        reach = int(row.reach_id)
        upstream_ids[reach] = [
            item for item in incoming.get(int(row.fnode), []) if item != reach
        ]
        for item in upstream_ids[reach]:
            graph.add_edge(item, reach)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen reach graph is not a DAG")
    reaches = np.array(list(nx.topological_sort(graph)), dtype=int)
    position = {reach: index for index, reach in enumerate(reaches)}
    upstream = {
        position[reach]: [position[item] for item in upstream_ids[reach]]
        for reach in reaches
    }
    return reaches, position, upstream


def run_full_model(cfg: dict) -> pd.DataFrame:
    forcing = pd.read_parquet(FORCING)
    static = pd.read_parquet(STATIC)
    mapping = pd.read_parquet(PARAMETERS)
    start_year, end_year = cfg["actual_period"]
    forcing = forcing.loc[forcing["year"].between(start_year, end_year)].copy()
    if len(forcing) != cfg["expected_reach_months"]:
        raise RuntimeError("Forcing scope mismatch")
    if static["reach_id"].nunique() != cfg["expected_reaches"]:
        raise RuntimeError("Static reach scope mismatch")
    if mapping["reach_id"].nunique() != cfg["expected_reaches"]:
        raise RuntimeError("Parameter map scope mismatch")
    if not np.allclose(mapping["gamma_ET"].to_numpy(float), 0.5):
        raise RuntimeError("The frozen parameter map is not gamma_ET=0.5")

    reaches, position, upstream = prepare_network(static)
    static_i = static.set_index("reach_id").loc[reaches]
    mapped = mapping.set_index("reach_id").loc[reaches]
    forcing_i = forcing.set_index(["year", "month", "reach_id"]).sort_index()
    area_factor = static_i["inc_area_km2"].to_numpy(float) * 1000.0
    parameters = {
        name: mapped[name].to_numpy(float)
        for name in [
            "kappa_s", "gamma_ET", "k_perc", "p_perc",
            "k_int", "p_int", "k_g", "k_route", "k_deep",
        ]
    }
    capacity = (
        parameters["kappa_s"]
        * static_i["soil_storage_eff_mm"].to_numpy(float)
    )
    soil = 0.5 * capacity
    groundwater = np.full(len(reaches), 50.0)
    channel_total = np.zeros(len(reaches))
    channel_base = np.zeros(len(reaches))
    channel_quick = np.zeros(len(reaches))
    maximum_system_closure = 0.0
    maximum_component_closure = 0.0

    def block(year: int, month: int):
        values = forcing_i.loc[(year, month)].loc[reaches]
        return (
            values["P_mm"].to_numpy(float),
            values["PET_mm"].to_numpy(float),
        )

    def route(
        storage: np.ndarray, local_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        upstream_flow = np.zeros(len(reaches))
        outflow = np.zeros(len(reaches))
        storage_end = np.zeros(len(reaches))
        for index in range(len(reaches)):
            parents = upstream[index]
            if parents:
                upstream_flow[index] = outflow[parents].sum()
            available = storage[index] + local_input[index] + upstream_flow[index]
            outflow[index] = parameters["k_route"][index] * available
            storage_end[index] = available - outflow[index]
        return upstream_flow, outflow, storage_end

    def advance(p_mm: np.ndarray, pet_mm: np.ndarray):
        nonlocal soil, groundwater
        nonlocal channel_total, channel_base, channel_quick
        nonlocal maximum_system_closure, maximum_component_closure
        soil0 = soil.copy()
        groundwater0 = groundwater.copy()
        channel0 = channel_total.copy()
        available_soil = soil0 + p_mm
        aet_wetness = np.clip(available_soil / capacity, 0.0, 1.0)
        aet = np.minimum(
            available_soil,
            pet_mm * aet_wetness ** parameters["gamma_ET"],
        )
        after_et = available_soil - aet
        excess = np.maximum(after_et - capacity, 0.0)
        temporary = np.minimum(after_et, capacity)
        wetness = np.clip(temporary / capacity, 0.0, 1.0)
        recharge = (
            parameters["k_perc"]
            * wetness ** parameters["p_perc"]
            * temporary
        )
        interflow = (
            parameters["k_int"]
            * wetness ** parameters["p_int"]
            * temporary
        )
        soil1 = temporary - recharge - interflow
        baseflow = parameters["k_g"] * np.maximum(groundwater0, 0.0)
        deep = parameters["k_deep"] * np.maximum(groundwater0, 0.0)
        groundwater1 = groundwater0 + recharge - baseflow - deep
        local_quick = (excess + interflow) * area_factor
        local_base = baseflow * area_factor
        up_total, out_total, total1 = route(
            channel_total, local_quick + local_base
        )
        _, out_base, base1 = route(channel_base, local_base)
        _, out_quick, quick1 = route(channel_quick, local_quick)

        start_total = (soil0 + groundwater0) * area_factor + channel0
        external_input = p_mm * area_factor + up_total
        external_output = aet * area_factor + deep * area_factor + out_total
        end_total = (soil1 + groundwater1) * area_factor + total1
        residual = start_total + external_input - external_output - end_total
        scale = (
            np.abs(start_total) + np.abs(external_input)
            + np.abs(external_output) + np.abs(end_total) + EPS
        )
        maximum_system_closure = max(
            maximum_system_closure,
            float(np.max(np.abs(residual) / scale)),
        )
        component_residual = np.maximum(
            np.abs(total1 - base1 - quick1),
            np.abs(out_total - out_base - out_quick),
        )
        component_scale = np.maximum(
            np.maximum(np.abs(total1), np.abs(out_total)), 1.0
        )
        maximum_component_closure = max(
            maximum_component_closure,
            float(np.max(component_residual / component_scale)),
        )
        if (
            not np.isfinite(soil1).all()
            or not np.isfinite(groundwater1).all()
            or not np.isfinite(total1).all()
            or min(
                float(soil1.min()),
                float(groundwater1.min()),
                float(total1.min()),
                float(base1.min()),
                float(quick1.min()),
            ) < -1e-9
            or float((soil1 - capacity).max()) > 1e-9
        ):
            raise RuntimeError("Invalid model state")
        soil, groundwater = soil1, groundwater1
        channel_total, channel_base, channel_quick = total1, base1, quick1
        return {
            "aet": aet,
            "recharge": recharge,
            "baseflow": baseflow,
            "groundwater": groundwater1,
            "out_total": out_total,
            "out_base": out_base,
            "out_quick": out_quick,
        }

    spin_start, spin_end = cfg["spinup_period"]
    spin_times = [
        (year, month)
        for year in range(spin_start, spin_end + 1)
        for month in range(1, 13)
    ]
    for _ in range(cfg["spinup_cycles"]):
        for year, month in spin_times:
            advance(*block(year, month))

    rows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            p_mm, pet_mm = block(year, month)
            state = advance(p_mm, pet_mm)
            seconds = calendar.monthrange(year, month)[1] * 86400.0
            for index, reach in enumerate(reaches):
                total = state["out_total"][index]
                routed_base = state["out_base"][index]
                rows.append({
                    "reach_id": int(reach),
                    "year": year,
                    "month": month,
                    "P_mm": float(p_mm[index]),
                    "PET_mm": float(pet_mm[index]),
                    "AET_mm": float(state["aet"][index]),
                    "groundwater_recharge_mm": float(state["recharge"][index]),
                    "local_baseflow_mm": float(state["baseflow"][index]),
                    "groundwater_storage_end_mm": float(
                        state["groundwater"][index]
                    ),
                    "channel_outflow_m3": float(total),
                    "routed_baseflow_outflow_m3": float(routed_base),
                    "routed_quickflow_outflow_m3": float(
                        state["out_quick"][index]
                    ),
                    "channel_outflow_cfs": float(
                        total / seconds / 0.028316846592
                    ),
                    "routed_baseflow_fraction": float(
                        routed_base / total if total > 0 else np.nan
                    ),
                })
    result = pd.DataFrame(rows)
    result.attrs["maximum_system_relative_closure"] = maximum_system_closure
    result.attrs[
        "maximum_linear_component_relative_closure"
    ] = maximum_component_closure
    return result


def eckhardt_filter(q: np.ndarray, alpha: float, bfi_max: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    base = np.zeros_like(q)
    if len(q) == 0:
        return base
    base[0] = min(q[0], bfi_max * q[0])
    denominator = 1.0 - alpha * bfi_max
    for index in range(1, len(q)):
        estimate = (
            (1.0 - bfi_max) * alpha * base[index - 1]
            + (1.0 - alpha) * bfi_max * q[index]
        ) / denominator
        base[index] = min(q[index], max(0.0, estimate))
    return base


def find_daily_file(station: str, year: int) -> Path | None:
    candidates = [
        root / str(year) / f"{station}.csv" for root in DAILY_ROOTS
    ]
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        raise RuntimeError(
            f"Duplicate active daily source for {station} {year}: {existing}"
        )
    return existing[0] if existing else None


def read_station_daily(
    station: str, start_year: int, end_year: int
) -> tuple[pd.DataFrame, list[Path], list[dict]]:
    rows = []
    used_files = []
    year_audit = []
    for year in range(start_year, end_year + 1):
        path = find_daily_file(station, year)
        if path is None:
            year_audit.append({
                "station_name": station,
                "year": year,
                "source_file": "",
                "valid_days": 0,
            })
            continue
        data = pd.read_csv(path)
        data.columns = [str(item).strip().lower() for item in data.columns]
        if "day" not in data or not set(MONTH_COLUMNS).issubset(data.columns):
            raise RuntimeError(f"Unexpected daily schema: {path}")
        used_files.append(path)
        valid_days = 0
        for month, column in enumerate(MONTH_COLUMNS, start=1):
            maximum_day = calendar.monthrange(year, month)[1]
            part = data.loc[
                pd.to_numeric(data["day"], errors="coerce").between(
                    1, maximum_day
                ),
                ["day", column],
            ].copy()
            part["day"] = pd.to_numeric(part["day"], errors="coerce")
            part["q_cfs"] = pd.to_numeric(part[column], errors="coerce")
            part = part.loc[
                part["day"].notna()
                & np.isfinite(part["q_cfs"])
                & part["q_cfs"].gt(0)
            ]
            valid_days += len(part)
            for item in part.itertuples():
                rows.append({
                    "station_name": station,
                    "date": pd.Timestamp(year, month, int(item.day)),
                    "year": year,
                    "month": month,
                    "q_cfs": float(item.q_cfs),
                })
        year_audit.append({
            "station_name": station,
            "year": year,
            "source_file": str(path),
            "valid_days": valid_days,
        })
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = daily.sort_values("date").drop_duplicates("date")
    return daily, used_files, year_audit


def circular_month_difference(first: int, second: int) -> int:
    raw = abs(int(first) - int(second))
    return min(raw, 12 - raw)


def safe_corr(first: pd.Series, second: pd.Series) -> float:
    valid = first.notna() & second.notna()
    if valid.sum() < 4:
        return np.nan
    first = first[valid].astype(float)
    second = second[valid].astype(float)
    if first.std(ddof=0) <= 0 or second.std(ddof=0) <= 0:
        return np.nan
    return float(first.corr(second))


def lowflow_persistence(values: pd.DataFrame, column: str) -> float:
    work = values[["date", column]].dropna().sort_values("date").copy()
    if len(work) < 24:
        return np.nan
    threshold = work[column].quantile(0.25)
    current = work.iloc[:-1].copy()
    following = work.iloc[1:].copy()
    adjacent = (
        following["date"].to_numpy(dtype="datetime64[M]")
        - current["date"].to_numpy(dtype="datetime64[M]")
    ).astype(int) == 1
    current_low = current[column].to_numpy(float) <= threshold
    following_low = following[column].to_numpy(float) <= threshold
    denominator = int(np.sum(adjacent & current_low))
    if denominator == 0:
        return np.nan
    return float(
        np.sum(adjacent & current_low & following_low) / denominator
    )


def station_signatures(
    cfg: dict, model: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    classification = pd.read_csv(STATION_CLASSIFICATION)
    classification["reservoir_related"] = bool_series(
        classification["reservoir_related"]
    )
    fixed = set(cfg["fixed_excluded_stations"])
    natural = classification.loc[
        ~classification["reservoir_related"],
        ["q_site", "reach_id"],
    ].drop_duplicates()
    if not fixed.isdisjoint(set(natural["q_site"])):
        raise RuntimeError("Fixed exclusion leaked into natural station set")

    primary = cfg["eckhardt_primary"]
    sensitivity = [
        (float(alpha), float(bfi_max))
        for alpha in cfg["eckhardt_sensitivity"]["alpha"]
        for bfi_max in cfg["eckhardt_sensitivity"]["bfi_max"]
    ]
    model_indexed = model.set_index(["reach_id", "year", "month"]).sort_index()
    coverage_rows = []
    station_rows = []
    monthly_rows = []
    used_files = []
    year_audits = []
    for station, reach_id in natural.itertuples(index=False):
        daily, files, audit = read_station_daily(
            station, *cfg["actual_period"]
        )
        used_files.extend(files)
        year_audits.extend(audit)
        if daily.empty:
            valid_years = []
        else:
            counts = daily.groupby("year").size()
            valid_years = counts.loc[
                counts.ge(cfg["minimum_valid_days_per_year"])
            ].index.astype(int).tolist()
        eligible = (
            len(valid_years) >= cfg["minimum_valid_years_per_station"]
        )
        coverage_rows.append({
            "station_name": station,
            "reach_id": int(reach_id),
            "reservoir_related": False,
            "valid_year_count": len(valid_years),
            "valid_years": ";".join(map(str, valid_years)),
            "valid_day_count": int(
                daily.loc[daily["year"].isin(valid_years)].shape[0]
                if not daily.empty else 0
            ),
            "eligible": eligible,
        })
        if not eligible:
            continue
        accepted = daily.loc[daily["year"].isin(valid_years)].copy()
        primary_parts = []
        sensitivity_totals = {
            pair: [0.0, 0.0] for pair in sensitivity
        }
        for year, group in accepted.groupby("year"):
            group = group.sort_values("date").copy()
            q = group["q_cfs"].to_numpy(float)
            group["baseflow_cfs"] = eckhardt_filter(
                q, float(primary["alpha"]), float(primary["bfi_max"])
            )
            primary_parts.append(group)
            for pair in sensitivity:
                separated = eckhardt_filter(q, *pair)
                sensitivity_totals[pair][0] += float(separated.sum())
                sensitivity_totals[pair][1] += float(q.sum())
        accepted = pd.concat(primary_parts, ignore_index=True)
        observed_bfi = float(
            accepted["baseflow_cfs"].sum() / accepted["q_cfs"].sum()
        )
        sensitivity_bfi = [
            numerator / denominator
            for numerator, denominator in sensitivity_totals.values()
            if denominator > 0
        ]
        observed_month = (
            accepted.groupby(["year", "month"], as_index=False)
            .agg(
                observed_q_cfs=("q_cfs", "mean"),
                observed_q_sum=("q_cfs", "sum"),
                observed_baseflow_sum=("baseflow_cfs", "sum"),
                valid_days=("q_cfs", "size"),
            )
        )
        observed_month["observed_baseflow_fraction"] = (
            observed_month["observed_baseflow_sum"]
            / observed_month["observed_q_sum"]
        )
        model_station = (
            model_indexed.loc[int(reach_id)]
            .reset_index()
            .loc[lambda frame: frame["year"].isin(valid_years)]
        )
        comparison = observed_month.merge(
            model_station[
                [
                    "year", "month", "channel_outflow_cfs",
                    "routed_baseflow_fraction",
                ]
            ],
            on=["year", "month"],
            how="inner",
            validate="one_to_one",
        )
        comparison["station_name"] = station
        comparison["reach_id"] = int(reach_id)
        monthly_rows.extend(comparison.to_dict("records"))
        simulated_bfi = float(
            model_station["routed_baseflow_outflow_m3"].sum()
            / model_station["channel_outflow_m3"].sum()
        )
        observed_clim = comparison.groupby("month")[
            "observed_baseflow_fraction"
        ].mean()
        simulated_clim = comparison.groupby("month")[
            "routed_baseflow_fraction"
        ].mean()
        seasonal_corr = safe_corr(observed_clim, simulated_clim)
        observed_peak = int(observed_clim.idxmax())
        simulated_peak = int(simulated_clim.idxmax())
        persistence_input = comparison.copy()
        persistence_input["date"] = pd.to_datetime(
            dict(
                year=persistence_input["year"],
                month=persistence_input["month"],
                day=1,
            )
        )
        observed_persistence = lowflow_persistence(
            persistence_input, "observed_q_cfs"
        )
        simulated_persistence = lowflow_persistence(
            persistence_input, "channel_outflow_cfs"
        )
        station_rows.append({
            "station_name": station,
            "reach_id": int(reach_id),
            "valid_year_count": len(valid_years),
            "observed_bfi": observed_bfi,
            "observed_bfi_sensitivity_min": min(sensitivity_bfi),
            "observed_bfi_sensitivity_max": max(sensitivity_bfi),
            "observed_bfi_sensitivity_width": (
                max(sensitivity_bfi) - min(sensitivity_bfi)
            ),
            "simulated_bfi": simulated_bfi,
            "bfi_difference": simulated_bfi - observed_bfi,
            "absolute_bfi_difference": abs(simulated_bfi - observed_bfi),
            "seasonal_baseflow_fraction_correlation": seasonal_corr,
            "observed_peak_month": observed_peak,
            "simulated_peak_month": simulated_peak,
            "peak_month_circular_difference": circular_month_difference(
                observed_peak, simulated_peak
            ),
            "observed_lowflow_persistence": observed_persistence,
            "simulated_lowflow_persistence": simulated_persistence,
            "lowflow_persistence_difference": (
                simulated_persistence - observed_persistence
            ),
            "absolute_lowflow_persistence_difference": abs(
                simulated_persistence - observed_persistence
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    station = pd.DataFrame(station_rows)
    monthly = pd.DataFrame(monthly_rows)
    pd.DataFrame(year_audits).to_csv(
        REPORT / "station_year_daily_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return coverage, station, monthly, sorted(set(used_files))


def main() -> None:
    for directory in [REPORT, OUTPUTS, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize baseflow diagnosis")

    model = run_full_model(cfg)
    maximum_system_closure = float(
        model.attrs["maximum_system_relative_closure"]
    )
    maximum_component_closure = float(
        model.attrs["maximum_linear_component_relative_closure"]
    )
    coverage, station, monthly, used_daily_files = station_signatures(
        cfg, model
    )
    model_path = OUTPUTS / "q78_nat_groundwater_reach_month.parquet"
    model.to_parquet(model_path, index=False)
    coverage.to_csv(
        REPORT / "daily_data_coverage.csv", index=False, encoding="utf-8-sig"
    )
    station.to_csv(
        REPORT / "station_signature_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly.to_parquet(
        OUTPUTS / "station_month_signature_comparison.parquet", index=False
    )

    protected = cfg["protected_station"]
    data_checks = {
        "parent_authorization": True,
        "at_least_60_eligible_natural_stations": int(
            coverage["eligible"].sum()
        ) >= cfg["minimum_eligible_natural_stations"],
        "protected_shijiao_eligible": bool(
            coverage.loc[
                coverage["station_name"].eq(protected), "eligible"
            ].any()
        ),
        "fixed_exclusions_absent": set(
            cfg["fixed_excluded_stations"]
        ).isdisjoint(set(coverage["station_name"])),
        "reservoir_context_absent": not bool(
            coverage["reservoir_related"].any()
        ),
    }
    data_passed = all(data_checks.values())
    threshold = cfg["thresholds"]
    if data_passed:
        spatial_spearman = float(
            station["observed_bfi"].corr(
                station["simulated_bfi"], method="spearman"
            )
        )
        metrics = {
            "eligible_natural_station_count": int(len(station)),
            "median_absolute_bfi_difference": float(
                station["absolute_bfi_difference"].median()
            ),
            "fraction_absolute_bfi_difference_le_0_25": float(
                station["absolute_bfi_difference"].le(0.25).mean()
            ),
            "spatial_bfi_spearman": spatial_spearman,
            "median_seasonal_correlation": float(
                station[
                    "seasonal_baseflow_fraction_correlation"
                ].median()
            ),
            "median_peak_month_circular_difference": float(
                station["peak_month_circular_difference"].median()
            ),
            "median_lowflow_persistence_absolute_difference": float(
                station[
                    "absolute_lowflow_persistence_difference"
                ].median()
            ),
            "median_observed_bfi_sensitivity_width": float(
                station["observed_bfi_sensitivity_width"].median()
            ),
            "maximum_system_relative_closure": maximum_system_closure,
            "maximum_linear_component_relative_closure": (
                maximum_component_closure
            ),
        }
        science_checks = {
            "median_absolute_bfi_difference_le_0_20": (
                metrics["median_absolute_bfi_difference"]
                <= threshold["median_absolute_bfi_difference_max"]
            ),
            "fraction_absolute_bfi_difference_le_0_25_ge_0_60": (
                metrics["fraction_absolute_bfi_difference_le_0_25"]
                >= threshold[
                    "fraction_absolute_bfi_difference_le_0_25_min"
                ]
            ),
            "spatial_bfi_spearman_ge_0_30": (
                metrics["spatial_bfi_spearman"]
                >= threshold["spatial_bfi_spearman_min"]
            ),
            "median_seasonal_correlation_ge_0_50": (
                metrics["median_seasonal_correlation"]
                >= threshold["median_seasonal_correlation_min"]
            ),
            "median_peak_month_difference_le_2": (
                metrics["median_peak_month_circular_difference"]
                <= threshold[
                    "median_peak_month_circular_difference_max"
                ]
            ),
            "median_lowflow_persistence_difference_le_0_20": (
                metrics[
                    "median_lowflow_persistence_absolute_difference"
                ]
                <= threshold[
                    "median_lowflow_persistence_absolute_difference_max"
                ]
            ),
            "strict_system_closure": maximum_system_closure < 1e-8,
            "strict_linear_component_closure": maximum_component_closure < 1e-8,
        }
    else:
        metrics = {
            "eligible_natural_station_count": int(
                coverage["eligible"].sum()
            ),
            "maximum_system_relative_closure": maximum_system_closure,
            "maximum_linear_component_relative_closure": (
                maximum_component_closure
            ),
        }
        science_checks = {}
    science_checks = {
        key: bool(value) for key, value in science_checks.items()
    }
    data_checks = {key: bool(value) for key, value in data_checks.items()}
    science_passed = data_passed and all(science_checks.values())
    if not data_passed:
        decision = "DATA_SUFFICIENCY_FAILED"
        next_action = cfg["next_action_if_data_fail"]
    elif science_passed:
        decision = "GROUNDWATER_BASEFLOW_SIGNATURE_ACCEPTED"
        next_action = cfg["next_action_if_pass"]
    else:
        decision = "GROUNDWATER_BASEFLOW_SIGNATURE_FAILED"
        next_action = cfg["next_action_if_fail"]
    gate = {
        "run_id": cfg["run_id"],
        "phase": "groundwater_baseflow_signature_gate",
        "pml_primary_aet_reference": True,
        "era5_decision_authority": False,
        "data_checks": data_checks,
        "science_checks": science_checks,
        "metrics": metrics,
        "data_passed": data_passed,
        "science_passed": science_passed,
        "decision": decision,
        "authorized_next_action": next_action,
        "used_year_min": cfg["actual_period"][0],
        "used_year_max": cfg["actual_period"][1],
        "confirmation_years_used": False,
        "station_discharge_used_for_calibration": False,
        "management_fluxes_read": False,
        "monthly_recession_branch_reopened": False,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failed = [key for key, value in science_checks.items() if not value]
    report = [
        "# Q78-NAT 地下水/baseflow signature 门禁",
        "",
        f"- 数据门禁：{'PASS' if data_passed else 'FAIL'}",
        f"- 科学门禁：{'PASS' if science_passed else 'FAIL'}",
        f"- 结论：`{decision}`",
        f"- 下一动作：`{next_action}`",
        f"- 合格自然站：{metrics['eligible_natural_station_count']}",
        f"- PML 为主要 AET 基准：True",
        f"- ERA5 具有方向裁决权：False",
        "",
    ]
    if data_passed:
        report.extend([
            "## 核心数值",
            "",
            (
                "- BFI 绝对差中位数："
                f"{metrics['median_absolute_bfi_difference']:.3f}"
            ),
            (
                "- BFI 绝对差≤0.25的站比例："
                f"{metrics['fraction_absolute_bfi_difference_le_0_25']:.1%}"
            ),
            (
                "- 站际 BFI Spearman："
                f"{metrics['spatial_bfi_spearman']:.3f}"
            ),
            (
                "- 月基流比例气候态相关中位数："
                f"{metrics['median_seasonal_correlation']:.3f}"
            ),
            (
                "- 峰值月份环形相位差中位数："
                f"{metrics['median_peak_month_circular_difference']:.1f} 月"
            ),
            (
                "- 低流持续概率绝对差中位数："
                f"{metrics['median_lowflow_persistence_absolute_difference']:.3f}"
            ),
            (
                "- 观测 BFI 滤波敏感性宽度中位数："
                f"{metrics['median_observed_bfi_sensitivity_width']:.3f}"
            ),
            "",
            "## 未通过项",
            "",
            *([f"- `{item}`" for item in failed] or ["- 无"]),
            "",
        ])
    report.extend([
        "## 边界",
        "",
        "- 站点日流量只生成独立 signature，没有参与参数率定。",
        "- 水库/人类调控背景站未进入科学门禁。",
        "- 2019–2022、管理通量均未读取。",
        "- `_46` 的月尺度退水分支没有重新开启。",
    ])
    (REPORT / "technical_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    daily_records = [
        file_record(path, "daily_discharge_signature_source", "reported")
        for path in used_daily_files
    ]
    pd.DataFrame(daily_records).to_csv(
        MANIFEST / "used_daily_discharge_files.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_files = [
        CONFIG, PARENT_GATE, PARAMETERS, FORCING, STATIC,
        STATION_CLASSIFICATION, RUN / "experiment_contract.md",
        RUN / "literature_basis.md",
        RUN / "scripts" / "run_groundwater_baseflow_signature.py",
    ]
    product_files = [
        model_path,
        OUTPUTS / "station_month_signature_comparison.parquet",
        REPORT / "daily_data_coverage.csv",
        REPORT / "station_year_daily_coverage.csv",
        REPORT / "station_signature_metrics.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            file_record(path, "groundwater_baseflow_source", "reported")
            for path in source_files
        ],
        "daily_source_count": len(daily_records),
        "products": [
            file_record(path, "groundwater_baseflow_product", "derived")
            for path in product_files
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
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
