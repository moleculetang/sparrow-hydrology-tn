from __future__ import annotations

import calendar
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "distributed_storage_preferential_recharge"
OUTPUTS = RUN / "outputs"
INPUTS = RUN / "inputs"
MANIFEST = RUN / "inputs_manifest"
CONFIG = RUN / "config" / "candidate_contract.json"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_25" / "reports"
    / "soil_storage_semantics" / "gate.json"
)
CAPACITY_RASTER = (
    ROOT / "0_reach_topology" / "data" / "processed" / "soil_prb"
    / "soil_storage_eff_mm_prb_buffer.tif"
)
CATCHMENTS = (
    ROOT / "5_Test" / "20260729_25" / "outputs"
    / "reach_catchments_esri54009.geojson"
)
FORCING = (
    ROOT / "5_Test" / "20260729_9" / "inputs"
    / "reach_month_forcing_2006_2018.parquet"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
PARAMETERS = (
    ROOT / "5_Test" / "20260729_21" / "inputs"
    / "reach_attribute_parameter_map_gamma_0_5.parquet"
)
BASE_MODEL = (
    ROOT / "5_Test" / "20260729_23" / "outputs"
    / "q78_nat_groundwater_reach_month.parquet"
)
BASE_STATION_MONTH = (
    ROOT / "5_Test" / "20260729_23" / "outputs"
    / "station_month_signature_comparison.parquet"
)
BASE_STATION_METRICS = (
    ROOT / "5_Test" / "20260729_23" / "reports"
    / "groundwater_baseflow_gate" / "station_signature_metrics.csv"
)
BASE_GATE = (
    ROOT / "5_Test" / "20260729_23" / "reports"
    / "groundwater_baseflow_gate" / "gate.json"
)
EPS = 1e-30


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
    path = Path(sys.prefix) / "Library" / "bin" / "gdal.exe"
    if not path.exists():
        raise RuntimeError(f"Missing GDAL in sparrow environment: {path}")
    return path


def build_capacity_classes(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_geojson = INPUTS / "capacity_zonal_stats.geojson"
    subprocess.run(
        [
            str(gdal_path()), "raster", "zonal-stats", "--quiet",
            "--overwrite", "--zones", str(CATCHMENTS),
            "--include-field", "reach_id", "--stat", "mean",
            "--stat", "stdev", "--stat", "min", "--stat", "max",
            "--stat", "count",
            str(CAPACITY_RASTER), str(stats_geojson),
        ],
        check=True,
    )
    payload = json.loads(stats_geojson.read_text(encoding="utf-8"))
    stats_rows: list[dict] = []
    for feature in payload["features"]:
        prop = feature["properties"]
        stats_rows.append(
            {
                "reach_id": int(prop["reach_id"]),
                "capacity_mean_mm": float(prop["mean"]),
                "capacity_stdev_mm": float(prop["stdev"]),
                "capacity_min_mm": float(prop["min"]),
                "capacity_max_mm": float(prop["max"]),
                "valid_pixel_count": int(prop["count"]),
            }
        )
    stats = pd.DataFrame(stats_rows).sort_values("reach_id")
    if len(stats) != cfg["expected_reaches"]:
        raise RuntimeError("Capacity zonal statistics do not cover 230 reaches")

    class_count = int(cfg["capacity_class_count"])
    probabilities = (np.arange(class_count, dtype=float) + 0.5) / class_count
    class_rows: list[dict] = []
    for row in stats.itertuples(index=False):
        low = float(row.capacity_min_mm)
        high = float(row.capacity_max_mm)
        mean = float(row.capacity_mean_mm)
        stdev = float(row.capacity_stdev_mm)
        width = high - low
        fit = "beta_moment"
        if width <= 1e-9 or stdev <= 1e-9:
            values = np.full(class_count, mean)
            fit = "degenerate"
        else:
            normalized_mean = np.clip((mean - low) / width, 1e-6, 1 - 1e-6)
            normalized_variance = (stdev / width) ** 2
            maximum_variance = normalized_mean * (1 - normalized_mean)
            normalized_variance = min(
                max(normalized_variance, 1e-9), maximum_variance * 0.999
            )
            common = maximum_variance / normalized_variance - 1.0
            alpha = normalized_mean * common
            beta = (1.0 - normalized_mean) * common
            values = low + width * beta_distribution.ppf(
                probabilities, alpha, beta
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"Invalid beta capacity fit at reach {row.reach_id}")
            for _ in range(8):
                values = np.clip(values + (mean - values.mean()), low, high)
        relative_error = abs(values.mean() - mean) / max(abs(mean), EPS)
        if relative_error > cfg["capacity_mean_relative_tolerance"]:
            raise RuntimeError(
                f"Capacity discretization mean error at reach {row.reach_id}: "
                f"{relative_error}"
            )
        for class_id, value in enumerate(values):
            class_rows.append(
                {
                    "reach_id": int(row.reach_id),
                    "capacity_class": class_id,
                    "area_weight": 1.0 / class_count,
                    "capacity_mm": float(value),
                    "fit_method": fit,
                    "mean_relative_error": float(relative_error),
                }
            )
    classes = pd.DataFrame(class_rows)
    stats.to_csv(
        INPUTS / "capacity_zonal_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classes.to_parquet(INPUTS / "capacity_classes.parquet", index=False)
    return stats, classes


def prepare_network(static: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming: defaultdict[int, list[int]] = defaultdict(list)
    for row in static.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
    upstream_ids: dict[int, list[int]] = {}
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
    return reaches, upstream


def run_candidate(
    cfg: dict, capacity_classes: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    forcing = pd.read_parquet(FORCING)
    static = pd.read_parquet(STATIC)
    mapping = pd.read_parquet(PARAMETERS)
    start_year, end_year = cfg["actual_period"]
    forcing = forcing.loc[forcing["year"].between(start_year, end_year)].copy()
    if len(forcing) != cfg["expected_reach_months"]:
        raise RuntimeError("Forcing scope mismatch")
    reaches, upstream = prepare_network(static)
    static_i = static.set_index("reach_id").loc[reaches]
    mapped = mapping.set_index("reach_id").loc[reaches]
    forcing_i = forcing.set_index(["year", "month", "reach_id"]).sort_index()
    class_count = cfg["capacity_class_count"]
    capacities = (
        capacity_classes.pivot(
            index="reach_id",
            columns="capacity_class",
            values="capacity_mm",
        )
        .loc[reaches, range(class_count)]
        .to_numpy(float)
    )
    area_factor = static_i["inc_area_km2"].to_numpy(float) * 1000.0
    parameters = {
        name: mapped[name].to_numpy(float)
        for name in [
            "gamma_ET", "k_perc", "p_perc", "k_int", "p_int",
            "k_g", "k_route", "k_deep",
        ]
    }
    soil = 0.5 * capacities
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
            values["AET_diagnostic_mm"].to_numpy(float),
        )

    def route(storage: np.ndarray, local_input: np.ndarray):
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

        prior_wetness_by_class = np.clip(soil0 / capacities, 0.0, 1.0)
        prior_wetness = prior_wetness_by_class.mean(axis=1)
        vertical_share = parameters["k_perc"] / (
            parameters["k_perc"] + parameters["k_int"]
        )
        preferential_fraction = np.clip(
            vertical_share
            * prior_wetness ** parameters["p_perc"],
            0.0,
            1.0,
        )
        preferential_recharge = p_mm * preferential_fraction
        matrix_precipitation = p_mm - preferential_recharge

        available = soil0 + matrix_precipitation[:, None]
        aet_wetness = np.clip(available / capacities, 0.0, 1.0)
        aet_by_class = np.minimum(
            available,
            pet_mm[:, None]
            * aet_wetness ** parameters["gamma_ET"][:, None],
        )
        after_et = available - aet_by_class
        excess_by_class = np.maximum(after_et - capacities, 0.0)
        temporary = np.minimum(after_et, capacities)
        wetness = np.clip(temporary / capacities, 0.0, 1.0)
        matrix_recharge_by_class = (
            parameters["k_perc"][:, None]
            * wetness ** parameters["p_perc"][:, None]
            * temporary
        )
        interflow_by_class = (
            parameters["k_int"][:, None]
            * wetness ** parameters["p_int"][:, None]
            * temporary
        )
        soil1 = temporary - matrix_recharge_by_class - interflow_by_class
        aet = aet_by_class.mean(axis=1)
        matrix_recharge = matrix_recharge_by_class.mean(axis=1)
        interflow = interflow_by_class.mean(axis=1)
        excess = excess_by_class.mean(axis=1)
        recharge = preferential_recharge + matrix_recharge

        baseflow = parameters["k_g"] * np.maximum(groundwater0, 0.0)
        deep = parameters["k_deep"] * np.maximum(groundwater0, 0.0)
        groundwater1 = groundwater0 + recharge - baseflow - deep
        local_quick = (excess + interflow) * area_factor
        local_base = baseflow * area_factor
        upstream_total, out_total, total1 = route(
            channel_total, local_quick + local_base
        )
        _, out_base, base1 = route(channel_base, local_base)
        _, out_quick, quick1 = route(channel_quick, local_quick)

        soil0_mean = soil0.mean(axis=1)
        soil1_mean = soil1.mean(axis=1)
        start_total = (soil0_mean + groundwater0) * area_factor + channel0
        external_input = p_mm * area_factor + upstream_total
        external_output = aet * area_factor + deep * area_factor + out_total
        end_total = (soil1_mean + groundwater1) * area_factor + total1
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
            or float((soil1 - capacities).max()) > 1e-9
        ):
            raise RuntimeError("Invalid candidate model state")
        soil, groundwater = soil1, groundwater1
        channel_total, channel_base, channel_quick = total1, base1, quick1
        return {
            "aet": aet,
            "preferential_fraction": preferential_fraction,
            "preferential_recharge": preferential_recharge,
            "matrix_recharge": matrix_recharge,
            "recharge": recharge,
            "interflow": interflow,
            "excess": excess,
            "baseflow": baseflow,
            "groundwater": groundwater1,
            "soil_mean": soil1_mean,
            "out_total": out_total,
            "out_base": out_base,
            "out_quick": out_quick,
        }

    spin_times = [
        (year, month)
        for year in range(cfg["spinup_period"][0], cfg["spinup_period"][1] + 1)
        for month in range(1, 13)
    ]
    for _ in range(cfg["spinup_cycles"]):
        for year, month in spin_times:
            p_mm, pet_mm, _ = block(year, month)
            advance(p_mm, pet_mm)

    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            p_mm, pet_mm, pml_aet = block(year, month)
            state = advance(p_mm, pet_mm)
            seconds = calendar.monthrange(year, month)[1] * 86400.0
            for index, reach in enumerate(reaches):
                total = state["out_total"][index]
                routed_base = state["out_base"][index]
                rows.append(
                    {
                        "reach_id": int(reach),
                        "year": year,
                        "month": month,
                        "P_mm": float(p_mm[index]),
                        "PET_mm": float(pet_mm[index]),
                        "AET_mm": float(state["aet"][index]),
                        "AET_diagnostic_mm": float(pml_aet[index]),
                        "preferential_fraction": float(
                            state["preferential_fraction"][index]
                        ),
                        "preferential_recharge_mm": float(
                            state["preferential_recharge"][index]
                        ),
                        "matrix_recharge_mm": float(
                            state["matrix_recharge"][index]
                        ),
                        "groundwater_recharge_mm": float(
                            state["recharge"][index]
                        ),
                        "interflow_mm": float(state["interflow"][index]),
                        "excess_mm": float(state["excess"][index]),
                        "local_baseflow_mm": float(state["baseflow"][index]),
                        "soil_storage_mean_end_mm": float(
                            state["soil_mean"][index]
                        ),
                        "groundwater_storage_end_mm": float(
                            state["groundwater"][index]
                        ),
                        "inc_area_km2": float(area_factor[index] / 1000.0),
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
                    }
                )
    model = pd.DataFrame(rows)
    diagnostics = {
        "maximum_system_relative_closure": maximum_system_closure,
        "maximum_linear_component_relative_closure": maximum_component_closure,
    }
    return model, diagnostics


def safe_corr(first: pd.Series, second: pd.Series, method: str = "pearson") -> float:
    valid = first.notna() & second.notna()
    if valid.sum() < 4:
        return np.nan
    x = first.loc[valid].astype(float)
    y = second.loc[valid].astype(float)
    if x.std(ddof=0) <= 0 or y.std(ddof=0) <= 0:
        return np.nan
    return float(x.corr(y, method=method))


def circular_month_difference(first: int, second: int) -> int:
    raw = abs(int(first) - int(second))
    return min(raw, 12 - raw)


def lowflow_persistence(frame: pd.DataFrame, column: str) -> float:
    work = frame[["date", column]].dropna().sort_values("date")
    if len(work) < 24:
        return np.nan
    threshold = work[column].quantile(0.25)
    current = work.iloc[:-1]
    following = work.iloc[1:]
    adjacent = (
        following["date"].to_numpy(dtype="datetime64[M]")
        - current["date"].to_numpy(dtype="datetime64[M]")
    ).astype(int) == 1
    current_low = current[column].to_numpy(float) <= threshold
    following_low = following[column].to_numpy(float) <= threshold
    denominator = int(np.sum(adjacent & current_low))
    if denominator == 0:
        return np.nan
    return float(np.sum(adjacent & current_low & following_low) / denominator)


def discharge_scores(observed: np.ndarray, simulated: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated) & (observed >= 0)
    observed = observed[valid]
    simulated = simulated[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    nse = 1.0 - np.sum((simulated - observed) ** 2) / denominator
    log_observed = np.log1p(observed)
    log_simulated = np.log1p(np.maximum(simulated, 0.0))
    log_denominator = np.sum((log_observed - log_observed.mean()) ** 2)
    log_nse = (
        1.0
        - np.sum((log_simulated - log_observed) ** 2) / log_denominator
    )
    correlation = np.corrcoef(observed, simulated)[0, 1]
    alpha = simulated.std(ddof=0) / observed.std(ddof=0)
    beta = simulated.mean() / observed.mean()
    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
    )
    return {
        "nse": float(nse),
        "log_nse": float(log_nse),
        "kge": float(kge),
        "relative_bias": float(beta - 1.0),
        "correlation": float(correlation),
    }


def compare_stations(model: pd.DataFrame, cfg: dict):
    baseline_month = pd.read_parquet(BASE_STATION_MONTH)
    baseline_station = pd.read_csv(BASE_STATION_METRICS)
    candidate_columns = [
        "reach_id", "year", "month", "channel_outflow_m3",
        "routed_baseflow_outflow_m3", "channel_outflow_cfs",
        "routed_baseflow_fraction",
    ]
    joined = baseline_month.merge(
        model[candidate_columns],
        on=["reach_id", "year", "month"],
        how="inner",
        validate="many_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    rows: list[dict] = []
    for station, group in joined.groupby("station_name"):
        group = group.sort_values(["year", "month"]).copy()
        reach_id = int(group["reach_id"].iloc[0])
        baseline_row = baseline_station.loc[
            baseline_station["station_name"].eq(station)
        ].iloc[0]
        candidate_bfi = float(
            group["routed_baseflow_outflow_m3"].sum()
            / group["channel_outflow_m3"].sum()
        )
        observed_clim = group.groupby("month")[
            "observed_baseflow_fraction"
        ].mean()
        candidate_clim = group.groupby("month")[
            "routed_baseflow_fraction_candidate"
        ].mean()
        candidate_peak = int(candidate_clim.idxmax())
        observed_peak = int(observed_clim.idxmax())
        group["date"] = pd.to_datetime(
            dict(year=group["year"], month=group["month"], day=1)
        )
        candidate_scores = discharge_scores(
            group["observed_q_cfs"].to_numpy(float),
            group["channel_outflow_cfs_candidate"].to_numpy(float),
        )
        baseline_scores = discharge_scores(
            group["observed_q_cfs"].to_numpy(float),
            group["channel_outflow_cfs_baseline"].to_numpy(float),
        )
        observed_bfi = float(baseline_row["observed_bfi"])
        rows.append(
            {
                "station_name": station,
                "reach_id": reach_id,
                "observed_bfi": observed_bfi,
                "baseline_bfi": float(baseline_row["simulated_bfi"]),
                "candidate_bfi": candidate_bfi,
                "baseline_absolute_bfi_difference": float(
                    baseline_row["absolute_bfi_difference"]
                ),
                "candidate_absolute_bfi_difference": abs(
                    candidate_bfi - observed_bfi
                ),
                "candidate_seasonal_correlation": safe_corr(
                    observed_clim, candidate_clim
                ),
                "candidate_peak_month_difference": (
                    circular_month_difference(observed_peak, candidate_peak)
                ),
                "candidate_lowflow_persistence": lowflow_persistence(
                    group, "channel_outflow_cfs_candidate"
                ),
                "observed_lowflow_persistence": float(
                    baseline_row["observed_lowflow_persistence"]
                ),
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_scores.items()
                },
                **{
                    f"baseline_{key}": value
                    for key, value in baseline_scores.items()
                },
            }
        )
    station = pd.DataFrame(rows)
    fixed = set(cfg["fixed_excluded_stations"])
    if not fixed.isdisjoint(set(station["station_name"])):
        raise RuntimeError("Fixed excluded station leaked into candidate comparison")
    if cfg["protected_station"] not in set(station["station_name"]):
        raise RuntimeError("Protected Shijiao station is absent")
    return joined, station


def aet_metrics(model: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    def summarize(frame: pd.DataFrame, aet_column: str) -> dict:
        area = frame["inc_area_km2"].to_numpy(float) * 1000.0
        candidate = frame[aet_column].to_numpy(float)
        reference = frame["AET_diagnostic_mm"].to_numpy(float)
        domain_bias = float(
            np.sum((candidate - reference) * area)
            / np.sum(reference * area)
        )
        reach_rows = []
        for _, group in frame.groupby("reach_id"):
            bias = (
                group[aet_column].sum() - group["AET_diagnostic_mm"].sum()
            ) / group["AET_diagnostic_mm"].sum()
            reach_rows.append(abs(float(bias)))
        return {
            "domain_relative_bias": domain_bias,
            "reach_absolute_bias_median": float(np.median(reach_rows)),
        }

    candidate_metrics = summarize(model, "AET_mm")
    base = baseline.merge(
        model[
            ["reach_id", "year", "month", "AET_diagnostic_mm", "inc_area_km2"]
        ],
        on=["reach_id", "year", "month"],
        validate="one_to_one",
    )
    baseline_metrics = summarize(base, "AET_mm")
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
    }


def main() -> None:
    for directory in [REPORT, OUTPUTS, INPUTS, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize this candidate")

    capacity_stats, capacity_classes = build_capacity_classes(cfg)
    model, closure = run_candidate(cfg, capacity_classes)
    if len(model) != cfg["expected_reach_months"]:
        raise RuntimeError("Candidate model row count mismatch")
    baseline_model = pd.read_parquet(BASE_MODEL)
    station_month, station = compare_stations(model, cfg)
    aet = aet_metrics(model, baseline_model)

    model.to_parquet(
        OUTPUTS / "distributed_storage_preferential_recharge_reach_month.parquet",
        index=False,
    )
    station_month.to_parquet(
        OUTPUTS / "station_month_candidate_vs_baseline.parquet",
        index=False,
    )
    station.to_csv(
        REPORT / "station_candidate_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_gate = json.loads(BASE_GATE.read_text(encoding="utf-8"))
    baseline_median_abs_bfi = float(
        station["baseline_absolute_bfi_difference"].median()
    )
    candidate_median_abs_bfi = float(
        station["candidate_absolute_bfi_difference"].median()
    )
    baseline_spatial_bfi = safe_corr(
        station["observed_bfi"], station["baseline_bfi"], method="spearman"
    )
    candidate_spatial_bfi = safe_corr(
        station["observed_bfi"], station["candidate_bfi"], method="spearman"
    )
    area = model["inc_area_km2"].to_numpy(float) * 1000.0
    recharge_volume = float(
        np.sum(model["groundwater_recharge_mm"].to_numpy(float) * area)
    )
    quick_volume = float(
        np.sum(
            (
                model["interflow_mm"].to_numpy(float)
                + model["excess_mm"].to_numpy(float)
            )
            * area
        )
    )
    preferential_volume = float(
        np.sum(model["preferential_recharge_mm"].to_numpy(float) * area)
    )
    metrics = {
        "station_count": int(len(station)),
        "baseline_median_absolute_bfi_difference": baseline_median_abs_bfi,
        "candidate_median_absolute_bfi_difference": candidate_median_abs_bfi,
        "median_absolute_bfi_improvement": (
            baseline_median_abs_bfi - candidate_median_abs_bfi
        ),
        "baseline_spatial_bfi_spearman": baseline_spatial_bfi,
        "candidate_spatial_bfi_spearman": candidate_spatial_bfi,
        "spatial_bfi_spearman_delta": (
            candidate_spatial_bfi - baseline_spatial_bfi
        ),
        "baseline_median_nse": float(station["baseline_nse"].median()),
        "candidate_median_nse": float(station["candidate_nse"].median()),
        "median_nse_delta": float(
            (station["candidate_nse"] - station["baseline_nse"]).median()
        ),
        "baseline_median_log_nse": float(
            station["baseline_log_nse"].median()
        ),
        "candidate_median_log_nse": float(
            station["candidate_log_nse"].median()
        ),
        "median_log_nse_delta": float(
            (
                station["candidate_log_nse"]
                - station["baseline_log_nse"]
            ).median()
        ),
        "baseline_median_kge": float(station["baseline_kge"].median()),
        "candidate_median_kge": float(station["candidate_kge"].median()),
        "candidate_median_seasonal_correlation": float(
            station["candidate_seasonal_correlation"].median()
        ),
        "candidate_median_peak_month_difference": float(
            station["candidate_peak_month_difference"].median()
        ),
        "baseline_aet_domain_relative_bias": float(
            aet["baseline"]["domain_relative_bias"]
        ),
        "candidate_aet_domain_relative_bias": float(
            aet["candidate"]["domain_relative_bias"]
        ),
        "candidate_recharge_fraction_of_generation": (
            recharge_volume / (recharge_volume + quick_volume)
        ),
        "candidate_preferential_fraction_of_recharge": (
            preferential_volume / recharge_volume
        ),
        "mean_preferential_precipitation_fraction": float(
            model["preferential_fraction"].mean()
        ),
        "maximum_capacity_class_mean_relative_error": float(
            capacity_classes["mean_relative_error"].max()
        ),
        **closure,
    }
    thresholds = cfg["comparison_thresholds"]
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["environment_name"] == "sparrow"
            and RUNTIME_IDENTITY["conda_default_env"] == "sparrow"
            and RUNTIME_IDENTITY["sys_prefix"]
            == RUNTIME_IDENTITY["expected_prefix"]
        ),
        "all_230_reaches_have_capacity_distribution": (
            capacity_stats["reach_id"].nunique() == cfg["expected_reaches"]
        ),
        "ten_capacity_classes_per_reach": bool(
            capacity_classes.groupby("reach_id").size().eq(
                cfg["capacity_class_count"]
            ).all()
        ),
        "capacity_class_means_match_raster": (
            metrics["maximum_capacity_class_mean_relative_error"]
            <= cfg["capacity_mean_relative_tolerance"]
        ),
        "same_97_station_sample": len(station) == 97,
        "protected_shijiao_present": (
            cfg["protected_station"] in set(station["station_name"])
        ),
        "fixed_exclusions_absent": set(
            cfg["fixed_excluded_stations"]
        ).isdisjoint(set(station["station_name"])),
        "preferential_path_is_active": preferential_volume > 0,
        "strict_system_closure": (
            metrics["maximum_system_relative_closure"]
            < thresholds["maximum_system_relative_closure_max"]
        ),
        "strict_component_closure": (
            metrics["maximum_linear_component_relative_closure"] < 1e-8
        ),
        "bfi_error_improves_by_at_least_0_10": (
            metrics["median_absolute_bfi_improvement"]
            >= thresholds["median_absolute_bfi_improvement_min"]
        ),
        "median_nse_not_materially_worse": (
            metrics["median_nse_delta"]
            >= thresholds["median_nse_delta_min"]
        ),
        "median_log_nse_not_materially_worse": (
            metrics["median_log_nse_delta"]
            >= thresholds["median_log_nse_delta_min"]
        ),
        "pml_aet_bias_not_materially_worse": (
            abs(metrics["candidate_aet_domain_relative_bias"])
            <= abs(metrics["baseline_aet_domain_relative_bias"])
            + thresholds["aet_absolute_bias_extra_max"]
        ),
        "spatial_bfi_pattern_not_materially_worse": (
            metrics["spatial_bfi_spearman_delta"]
            >= thresholds["spatial_bfi_spearman_delta_min"]
        ),
        "confirmation_years_not_used": True,
        "management_fluxes_not_read": True,
        "station_discharge_not_used_for_calibration": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    comparison_names = [
        "bfi_error_improves_by_at_least_0_10",
        "median_nse_not_materially_worse",
        "median_log_nse_not_materially_worse",
        "pml_aet_bias_not_materially_worse",
        "spatial_bfi_pattern_not_materially_worse",
    ]
    hard_names = [
        key for key in checks if key not in comparison_names
    ]
    hard_passed = all(checks[key] for key in hard_names)
    comparison_passed = all(checks[key] for key in comparison_names)
    bfi_improved = checks["bfi_error_improves_by_at_least_0_10"]
    if hard_passed and comparison_passed:
        decision = "DISTRIBUTED_STORAGE_PREF_RECHARGE_CANDIDATE_PASSED"
        next_action = cfg["next_action_if_pass"]
    elif hard_passed and bfi_improved:
        decision = "CANDIDATE_IMPROVED_BFI_BUT_FAILED_GUARDRAIL"
        next_action = cfg["next_action_if_tradeoff"]
    else:
        decision = "DISTRIBUTED_STORAGE_PREF_RECHARGE_CANDIDATE_FAILED"
        next_action = cfg["next_action_if_fail"]
    gate = {
        "run_id": cfg["run_id"],
        "phase": "distributed_storage_preferential_recharge_test",
        "checks": checks,
        "metrics": metrics,
        "hard_passed": hard_passed,
        "comparison_passed": comparison_passed,
        "decision": decision,
        "authorized_next_action": next_action,
        "baseline_gate_metrics": baseline_gate["metrics"],
        "pml_primary_aet_reference": True,
        "era5_decision_authority": False,
        "capacity_multiplier_calibrated": False,
        "new_free_parameter_calibrated": False,
        "confirmation_years_used": False,
        "management_fluxes_read": False,
        "station_discharge_used_for_calibration": False,
        "runtime_identity": RUNTIME_IDENTITY,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [key for key, value in checks.items() if not value]
    report_lines = [
        "# 分布式容量 + 优先补给单候选测试",
        "",
        f"- 结论：`{decision}`",
        f"- 下一动作：`{next_action}`",
        f"- 独立自然站：{len(station)}（石角保留）",
        "",
        "## 相对 `_23` 的效果",
        "",
        (
            f"- BFI 绝对误差中位数：{baseline_median_abs_bfi:.3f} → "
            f"{candidate_median_abs_bfi:.3f}"
        ),
        (
            f"- BFI 误差改善："
            f"{metrics['median_absolute_bfi_improvement']:+.3f}"
        ),
        (
            f"- 站际 BFI Spearman：{baseline_spatial_bfi:.3f} → "
            f"{candidate_spatial_bfi:.3f}"
        ),
        (
            f"- 月流量 NSE 中位数："
            f"{metrics['baseline_median_nse']:.3f} → "
            f"{metrics['candidate_median_nse']:.3f}"
        ),
        (
            f"- 月流量 log-NSE 中位数："
            f"{metrics['baseline_median_log_nse']:.3f} → "
            f"{metrics['candidate_median_log_nse']:.3f}"
        ),
        (
            f"- PML AET 全域相对偏差："
            f"{metrics['baseline_aet_domain_relative_bias']:.1%} → "
            f"{metrics['candidate_aet_domain_relative_bias']:.1%}"
        ),
        (
            f"- 地下水补给占局地产流："
            f"{metrics['candidate_recharge_fraction_of_generation']:.1%}"
        ),
        (
            f"- 优先补给占总补给："
            f"{metrics['candidate_preferential_fraction_of_recharge']:.1%}"
        ),
        "",
        "## 未通过项",
        "",
        *([f"- `{item}`" for item in failed] or ["- 无"]),
        "",
        "## 边界",
        "",
        "- 没有率定容量倍数或新自由参数。",
        "- 没有改变 `k_g`、`k_route` 或既有参数图。",
        "- 站点流量只用于冻结后的比较门禁。",
        "- 2019–2022 与管理通量未读取。",
    ]
    (REPORT / "technical_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    sources = [
        CONFIG, PARENT_GATE, CAPACITY_RASTER, CATCHMENTS, FORCING,
        STATIC, PARAMETERS, BASE_MODEL, BASE_STATION_MONTH,
        BASE_STATION_METRICS, BASE_GATE, RUN / "experiment_contract.md",
        RUN / "literature_basis.md",
        RUN / "scripts" / "run_distributed_storage_preferential_recharge.py",
    ]
    products = [
        INPUTS / "capacity_zonal_stats.geojson",
        INPUTS / "capacity_zonal_stats.csv",
        INPUTS / "capacity_classes.parquet",
        OUTPUTS / "distributed_storage_preferential_recharge_reach_month.parquet",
        OUTPUTS / "station_month_candidate_vs_baseline.parquet",
        REPORT / "station_candidate_vs_baseline.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_identity": RUNTIME_IDENTITY,
        "sources": [
            record(path, "candidate_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "candidate_product", "derived")
            for path in products
        ],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(provenance["sources"] + provenance["products"]).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
