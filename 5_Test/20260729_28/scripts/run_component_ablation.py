from __future__ import annotations

import calendar
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "component_ablation"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_27" / "reports"
    / "tradeoff_and_unit_audit" / "gate.json"
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
CAPACITY_CLASSES = (
    ROOT / "5_Test" / "20260729_26" / "inputs"
    / "capacity_classes.parquet"
)
BASELINE_MODEL = (
    ROOT / "5_Test" / "20260729_23" / "outputs"
    / "q78_nat_groundwater_reach_month.parquet"
)
COMBINED_MODEL = (
    ROOT / "5_Test" / "20260729_26" / "outputs"
    / "distributed_storage_preferential_recharge_reach_month.parquet"
)
OBSERVED_MONTH = (
    ROOT / "5_Test" / "20260729_27" / "outputs"
    / "station_month_corrected_m3s.parquet"
)
OBSERVED_STATION = (
    ROOT / "5_Test" / "20260729_27" / "reports"
    / "tradeoff_and_unit_audit" / "station_corrected_unit_tradeoff.csv"
)
EPS = 1e-30
CFS_PER_M3S = 35.3146667215


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


def prepare_network(static: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming: defaultdict[int, list[int]] = defaultdict(list)
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
        raise RuntimeError("Frozen graph is not a DAG")
    reaches = np.array(list(nx.topological_sort(graph)), dtype=int)
    position = {reach: index for index, reach in enumerate(reaches)}
    upstream = {
        position[reach]: [position[item] for item in upstream_ids[reach]]
        for reach in reaches
    }
    return reaches, upstream


def run_scenario(
    name: str,
    capacity_mode: str,
    preferential_enabled: bool,
) -> tuple[pd.DataFrame, dict]:
    forcing = pd.read_parquet(FORCING)
    static = pd.read_parquet(STATIC)
    mapping = pd.read_parquet(PARAMETERS)
    reaches, upstream = prepare_network(static)
    static_i = static.set_index("reach_id").loc[reaches]
    mapped = mapping.set_index("reach_id").loc[reaches]
    forcing_i = forcing.set_index(["year", "month", "reach_id"]).sort_index()
    if capacity_mode == "distributed":
        classes = pd.read_parquet(CAPACITY_CLASSES)
        capacities = (
            classes.pivot(
                index="reach_id",
                columns="capacity_class",
                values="capacity_mm",
            ).loc[reaches].to_numpy(float)
        )
    elif capacity_mode == "homogeneous":
        capacities = static_i["soil_storage_eff_mm"].to_numpy(float)[:, None]
    else:
        raise ValueError(capacity_mode)
    area = static_i["inc_area_km2"].to_numpy(float) * 1000.0
    parameters = {
        key: mapped[key].to_numpy(float)
        for key in [
            "gamma_ET", "k_perc", "p_perc", "k_int", "p_int",
            "k_g", "k_route", "k_deep",
        ]
    }
    soil = 0.5 * capacities
    groundwater = np.full(len(reaches), 50.0)
    channel_total = np.zeros(len(reaches))
    channel_base = np.zeros(len(reaches))
    channel_quick = np.zeros(len(reaches))
    max_system = 0.0
    max_component = 0.0

    def block(year: int, month: int):
        values = forcing_i.loc[(year, month)].loc[reaches]
        return (
            values["P_mm"].to_numpy(float),
            values["PET_mm"].to_numpy(float),
            values["AET_diagnostic_mm"].to_numpy(float),
        )

    def route(storage: np.ndarray, local: np.ndarray):
        upstream_flow = np.zeros(len(reaches))
        outflow = np.zeros(len(reaches))
        end = np.zeros(len(reaches))
        for index in range(len(reaches)):
            parents = upstream[index]
            if parents:
                upstream_flow[index] = outflow[parents].sum()
            available = storage[index] + local[index] + upstream_flow[index]
            outflow[index] = parameters["k_route"][index] * available
            end[index] = available - outflow[index]
        return upstream_flow, outflow, end

    def advance(p_mm: np.ndarray, pet_mm: np.ndarray):
        nonlocal soil, groundwater
        nonlocal channel_total, channel_base, channel_quick
        nonlocal max_system, max_component
        soil0 = soil.copy()
        groundwater0 = groundwater.copy()
        channel0 = channel_total.copy()
        prior_wetness = np.clip(soil0 / capacities, 0, 1).mean(axis=1)
        if preferential_enabled:
            vertical_share = parameters["k_perc"] / (
                parameters["k_perc"] + parameters["k_int"]
            )
            pref_fraction = np.clip(
                vertical_share
                * prior_wetness ** parameters["p_perc"],
                0,
                1,
            )
        else:
            pref_fraction = np.zeros(len(reaches))
        pref_recharge = p_mm * pref_fraction
        matrix_p = p_mm - pref_recharge
        available = soil0 + matrix_p[:, None]
        aet_wetness = np.clip(available / capacities, 0, 1)
        aet_classes = np.minimum(
            available,
            pet_mm[:, None]
            * aet_wetness ** parameters["gamma_ET"][:, None],
        )
        after_et = available - aet_classes
        excess_classes = np.maximum(after_et - capacities, 0)
        temporary = np.minimum(after_et, capacities)
        wetness = np.clip(temporary / capacities, 0, 1)
        matrix_recharge_classes = (
            parameters["k_perc"][:, None]
            * wetness ** parameters["p_perc"][:, None]
            * temporary
        )
        interflow_classes = (
            parameters["k_int"][:, None]
            * wetness ** parameters["p_int"][:, None]
            * temporary
        )
        soil1 = temporary - matrix_recharge_classes - interflow_classes
        aet = aet_classes.mean(axis=1)
        matrix_recharge = matrix_recharge_classes.mean(axis=1)
        interflow = interflow_classes.mean(axis=1)
        excess = excess_classes.mean(axis=1)
        recharge = pref_recharge + matrix_recharge
        baseflow = parameters["k_g"] * np.maximum(groundwater0, 0)
        deep = parameters["k_deep"] * np.maximum(groundwater0, 0)
        groundwater1 = groundwater0 + recharge - baseflow - deep
        local_quick = (interflow + excess) * area
        local_base = baseflow * area
        upstream_total, out_total, total1 = route(
            channel_total, local_quick + local_base
        )
        _, out_base, base1 = route(channel_base, local_base)
        _, out_quick, quick1 = route(channel_quick, local_quick)
        start = (soil0.mean(axis=1) + groundwater0) * area + channel0
        inputs = p_mm * area + upstream_total
        outputs = aet * area + deep * area + out_total
        end = (soil1.mean(axis=1) + groundwater1) * area + total1
        residual = start + inputs - outputs - end
        scale = np.abs(start) + np.abs(inputs) + np.abs(outputs) + np.abs(end) + EPS
        max_system = max(
            max_system, float(np.max(np.abs(residual) / scale))
        )
        component_residual = np.maximum(
            np.abs(total1 - base1 - quick1),
            np.abs(out_total - out_base - out_quick),
        )
        component_scale = np.maximum(
            np.maximum(np.abs(total1), np.abs(out_total)), 1
        )
        max_component = max(
            max_component,
            float(np.max(component_residual / component_scale)),
        )
        if (
            not np.isfinite(soil1).all()
            or not np.isfinite(groundwater1).all()
            or min(
                float(soil1.min()),
                float(groundwater1.min()),
                float(total1.min()),
            ) < -1e-9
            or float((soil1 - capacities).max()) > 1e-9
        ):
            raise RuntimeError(f"Invalid state in {name}")
        soil, groundwater = soil1, groundwater1
        channel_total, channel_base, channel_quick = total1, base1, quick1
        return {
            "aet": aet,
            "pml": None,
            "pref_fraction": pref_fraction,
            "pref_recharge": pref_recharge,
            "matrix_recharge": matrix_recharge,
            "recharge": recharge,
            "interflow": interflow,
            "excess": excess,
            "out_total": out_total,
            "out_base": out_base,
            "out_quick": out_quick,
        }

    spin_times = [
        (year, month)
        for year in range(2006, 2012)
        for month in range(1, 13)
    ]
    for _ in range(10):
        for year, month in spin_times:
            p_mm, pet_mm, _ = block(year, month)
            advance(p_mm, pet_mm)
    rows = []
    for year in range(2006, 2019):
        for month in range(1, 13):
            p_mm, pet_mm, pml = block(year, month)
            state = advance(p_mm, pet_mm)
            seconds = calendar.monthrange(year, month)[1] * 86400.0
            for index, reach in enumerate(reaches):
                total = state["out_total"][index]
                base = state["out_base"][index]
                rows.append(
                    {
                        "scenario": name,
                        "reach_id": int(reach),
                        "year": year,
                        "month": month,
                        "P_mm": float(p_mm[index]),
                        "AET_mm": float(state["aet"][index]),
                        "AET_diagnostic_mm": float(pml[index]),
                        "preferential_fraction": float(
                            state["pref_fraction"][index]
                        ),
                        "preferential_recharge_mm": float(
                            state["pref_recharge"][index]
                        ),
                        "matrix_recharge_mm": float(
                            state["matrix_recharge"][index]
                        ),
                        "groundwater_recharge_mm": float(
                            state["recharge"][index]
                        ),
                        "interflow_mm": float(state["interflow"][index]),
                        "excess_mm": float(state["excess"][index]),
                        "inc_area_km2": float(area[index] / 1000.0),
                        "channel_outflow_m3": float(total),
                        "routed_baseflow_outflow_m3": float(base),
                        "channel_outflow_m3s": float(total / seconds),
                        "routed_baseflow_fraction": float(
                            base / total if total > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows), {
        "maximum_system_relative_closure": max_system,
        "maximum_component_relative_closure": max_component,
    }


def scores(observed: np.ndarray, simulated: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    denominator = np.sum((observed - observed.mean()) ** 2)
    nse = 1 - np.sum((simulated - observed) ** 2) / denominator
    log_observed = np.log1p(observed)
    log_simulated = np.log1p(np.maximum(simulated, 0))
    log_nse = 1 - np.sum(
        (log_simulated - log_observed) ** 2
    ) / np.sum((log_observed - log_observed.mean()) ** 2)
    correlation = float(np.corrcoef(observed, simulated)[0, 1])
    alpha = simulated.std(ddof=0) / observed.std(ddof=0)
    beta = simulated.mean() / observed.mean()
    kge = 1 - np.sqrt(
        (correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    )
    return {
        "nse": float(nse),
        "log_nse": float(log_nse),
        "kge": float(kge),
        "relative_bias": float(beta - 1),
    }


def normalize_existing(model: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = model.copy()
    frame["scenario"] = scenario
    frame["channel_outflow_m3s"] = frame["channel_outflow_cfs"] / CFS_PER_M3S
    if "AET_diagnostic_mm" not in frame:
        forcing = pd.read_parquet(FORCING)[
            ["reach_id", "year", "month", "AET_diagnostic_mm"]
        ]
        frame = frame.merge(
            forcing,
            on=["reach_id", "year", "month"],
            validate="one_to_one",
        )
    if "inc_area_km2" not in frame:
        static = pd.read_parquet(STATIC)[["reach_id", "inc_area_km2"]]
        frame = frame.merge(static, on="reach_id", validate="many_to_one")
    return frame


def evaluate_scenario(
    scenario: str,
    model: pd.DataFrame,
    observed_month: pd.DataFrame,
    observed_station: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    joined = observed_month[
        ["station_name", "reach_id", "year", "month", "observed_q_m3s"]
    ].merge(
        model[
            [
                "reach_id", "year", "month", "channel_outflow_m3s",
                "channel_outflow_m3", "routed_baseflow_outflow_m3",
            ]
        ],
        on=["reach_id", "year", "month"],
        validate="many_to_one",
    )
    rows = []
    for station_name, group in joined.groupby("station_name"):
        obs = group["observed_q_m3s"].to_numpy(float)
        sim = group["channel_outflow_m3s"].to_numpy(float)
        station_score = scores(obs, sim)
        candidate_bfi = float(
            group["routed_baseflow_outflow_m3"].sum()
            / group["channel_outflow_m3"].sum()
        )
        observed_bfi = float(
            observed_station.loc[
                observed_station["station_name"].eq(station_name),
                "observed_bfi",
            ].iloc[0]
        )
        rows.append(
            {
                "scenario": scenario,
                "station_name": station_name,
                "reach_id": int(group["reach_id"].iloc[0]),
                "observed_bfi": observed_bfi,
                "simulated_bfi": candidate_bfi,
                "absolute_bfi_error": abs(candidate_bfi - observed_bfi),
                **station_score,
            }
        )
    station = pd.DataFrame(rows)
    spatial = float(
        station["observed_bfi"].corr(
            station["simulated_bfi"], method="spearman"
        )
    )
    area = model["inc_area_km2"].to_numpy(float) * 1000.0
    aet_bias = float(
        np.sum(
            (
                model["AET_mm"].to_numpy(float)
                - model["AET_diagnostic_mm"].to_numpy(float)
            )
            * area
        )
        / np.sum(model["AET_diagnostic_mm"].to_numpy(float) * area)
    )
    metrics = {
        "scenario": scenario,
        "median_bfi": float(station["simulated_bfi"].median()),
        "median_absolute_bfi_error": float(
            station["absolute_bfi_error"].median()
        ),
        "spatial_bfi_spearman": spatial,
        "median_nse": float(station["nse"].median()),
        "median_log_nse": float(station["log_nse"].median()),
        "median_kge": float(station["kge"].median()),
        "median_relative_bias": float(station["relative_bias"].median()),
        "pml_aet_domain_relative_bias": aet_bias,
    }
    if {
        "groundwater_recharge_mm",
        "interflow_mm",
        "excess_mm",
        "preferential_recharge_mm",
    }.issubset(model.columns):
        recharge = float(
            np.sum(model["groundwater_recharge_mm"].to_numpy(float) * area)
        )
        quick = float(
            np.sum(
                (
                    model["interflow_mm"].to_numpy(float)
                    + model["excess_mm"].to_numpy(float)
                )
                * area
            )
        )
        preferential = float(
            np.sum(model["preferential_recharge_mm"].to_numpy(float) * area)
        )
        metrics["recharge_fraction_of_generation"] = recharge / (
            recharge + quick
        )
        metrics["preferential_share_of_recharge"] = (
            preferential / recharge if recharge > 0 else 0.0
        )
    else:
        metrics["recharge_fraction_of_generation"] = np.nan
        metrics["preferential_share_of_recharge"] = 0.0
    return metrics, station


def main() -> None:
    for directory in [REPORT, OUTPUTS, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    required = "RUN_PARAMETER_FREE_CAPACITY_VS_PREFERENCE_COMPONENT_ABLATION"
    if parent["authorized_next_action"] != required:
        raise RuntimeError("Parent gate does not authorize component ablation")

    distributed, closure_distributed = run_scenario(
        "distributed_only", "distributed", False
    )
    preference, closure_preference = run_scenario(
        "preference_only", "homogeneous", True
    )
    distributed.to_parquet(
        OUTPUTS / "distributed_only_reach_month.parquet", index=False
    )
    preference.to_parquet(
        OUTPUTS / "preference_only_reach_month.parquet", index=False
    )
    baseline = normalize_existing(
        pd.read_parquet(BASELINE_MODEL), "baseline"
    )
    combined_raw = pd.read_parquet(COMBINED_MODEL)
    combined = combined_raw.copy()
    combined["scenario"] = "combined"
    combined["channel_outflow_m3s"] = (
        combined["channel_outflow_cfs"] / CFS_PER_M3S
    )

    observed_month = pd.read_parquet(OBSERVED_MONTH)
    observed_station = pd.read_csv(OBSERVED_STATION)
    scenario_models = {
        "baseline": baseline,
        "distributed_only": distributed,
        "preference_only": preference,
        "combined": combined,
    }
    metric_rows = []
    station_rows = []
    for scenario, model in scenario_models.items():
        metrics, stations = evaluate_scenario(
            scenario, model, observed_month, observed_station
        )
        metric_rows.append(metrics)
        station_rows.append(stations)
    metrics = pd.DataFrame(metric_rows).set_index("scenario")
    station = pd.concat(station_rows, ignore_index=True)
    metrics.reset_index().to_csv(
        REPORT / "scenario_metrics.csv", index=False, encoding="utf-8-sig"
    )
    station.to_csv(
        REPORT / "station_scenario_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    directions = {
        "median_bfi": 1.0,
        "median_absolute_bfi_error": -1.0,
        "spatial_bfi_spearman": 1.0,
        "median_nse": 1.0,
        "median_log_nse": 1.0,
        "median_kge": 1.0,
        "pml_aet_domain_relative_bias": -1.0,
    }
    effects = []
    for metric, direction in directions.items():
        base = float(metrics.loc["baseline", metric])
        dist = float(metrics.loc["distributed_only", metric])
        pref = float(metrics.loc["preference_only", metric])
        combined_value = float(metrics.loc["combined", metric])
        effects.append(
            {
                "metric": metric,
                "higher_is_better": direction > 0,
                "baseline": base,
                "distributed_only": dist,
                "preference_only": pref,
                "combined": combined_value,
                "distribution_main_effect": dist - base,
                "preference_main_effect": pref - base,
                "interaction_effect": combined_value - dist - pref + base,
            }
        )
    effects = pd.DataFrame(effects)
    effects.to_csv(
        REPORT / "factorial_effects.csv", index=False, encoding="utf-8-sig"
    )

    bfi_gain_combined = (
        metrics.loc["combined", "median_bfi"]
        - metrics.loc["baseline", "median_bfi"]
    )
    bfi_gain_preference = (
        metrics.loc["preference_only", "median_bfi"]
        - metrics.loc["baseline", "median_bfi"]
    )
    preference_gain_share = (
        float(bfi_gain_preference / bfi_gain_combined)
        if abs(bfi_gain_combined) > EPS else np.nan
    )
    summary = {
        "preference_share_of_combined_bfi_gain": preference_gain_share,
        "distribution_only_nse_delta": float(
            metrics.loc["distributed_only", "median_nse"]
            - metrics.loc["baseline", "median_nse"]
        ),
        "distribution_only_log_nse_delta": float(
            metrics.loc["distributed_only", "median_log_nse"]
            - metrics.loc["baseline", "median_log_nse"]
        ),
        "distribution_only_spatial_bfi_delta": float(
            metrics.loc["distributed_only", "spatial_bfi_spearman"]
            - metrics.loc["baseline", "spatial_bfi_spearman"]
        ),
        "preference_only_bfi_error_improvement": float(
            metrics.loc["baseline", "median_absolute_bfi_error"]
            - metrics.loc["preference_only", "median_absolute_bfi_error"]
        ),
        "preference_only_spatial_bfi_delta": float(
            metrics.loc["preference_only", "spatial_bfi_spearman"]
            - metrics.loc["baseline", "spatial_bfi_spearman"]
        ),
        "distributed_closure": closure_distributed,
        "preference_closure": closure_preference,
    }
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["environment_name"] == "sparrow"
            and RUNTIME_IDENTITY["conda_default_env"] == "sparrow"
            and RUNTIME_IDENTITY["sys_prefix"]
            == RUNTIME_IDENTITY["expected_prefix"]
        ),
        "four_scenarios_present": len(metrics) == 4,
        "same_97_stations_each": bool(
            station.groupby("scenario").size().eq(97).all()
        ),
        "shijiao_present_each": bool(
            station.loc[station["station_name"].eq("石角站")]
            .groupby("scenario").size().eq(1).all()
        ),
        "fixed_exclusions_absent": {
            "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
        }.isdisjoint(set(station["station_name"])),
        "new_scenarios_strict_closure": (
            closure_distributed["maximum_system_relative_closure"] < 1e-8
            and closure_distributed[
                "maximum_component_relative_closure"
            ] < 1e-8
            and closure_preference["maximum_system_relative_closure"] < 1e-8
            and closure_preference[
                "maximum_component_relative_closure"
            ] < 1e-8
        ),
        "preference_is_primary_bfi_gain_driver": (
            preference_gain_share >= 0.75
        ),
        "distribution_safe_for_flow": (
            summary["distribution_only_nse_delta"] >= -0.05
            and summary["distribution_only_log_nse_delta"] >= -0.05
        ),
        "distribution_safe_for_spatial_bfi": (
            summary["distribution_only_spatial_bfi_delta"] >= -0.05
        ),
        "current_preference_formula_improves_bfi_error": (
            summary["preference_only_bfi_error_improvement"] >= 0.10
        ),
        "current_preference_formula_preserves_spatial_bfi": (
            summary["preference_only_spatial_bfi_delta"] >= -0.05
        ),
        "no_parameter_calibration": True,
        "correct_observed_unit_m3s": True,
        "no_confirmation_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    retain_distribution = (
        checks["distribution_safe_for_flow"]
        and checks["distribution_safe_for_spatial_bfi"]
    )
    retain_preference_formula = (
        checks["current_preference_formula_improves_bfi_error"]
        and checks["current_preference_formula_preserves_spatial_bfi"]
    )
    if retain_distribution and not retain_preference_formula:
        decision = "RETAIN_CAPACITY_DISTRIBUTION_REJECT_CURRENT_PREF_SPATIAL_FORMULA"
        next_action = (
            "LITERATURE_GUIDED_REDESIGN_OF_SPATIALLY_CONSTRAINED_RECHARGE_PATH"
        )
    elif not retain_distribution and not retain_preference_formula:
        decision = "REJECT_BOTH_COMPONENTS_IN_CURRENT_FORM"
        next_action = "REASSESS_STORAGE_AND_RECHARGE_STRUCTURE_FROM_LITERATURE"
    elif retain_distribution and retain_preference_formula:
        decision = "RETAIN_BOTH_COMPONENTS_FOR_NEXT_SIGNATURE_GATE"
        next_action = "TEST_RETAINED_COMPONENTS_AGAINST_FULL_SIGNATURE_GATE"
    else:
        decision = "RETAIN_PREF_ONLY_REJECT_DISTRIBUTED_CAPACITY"
        next_action = "TEST_PREF_ONLY_AGAINST_FULL_SIGNATURE_GATE"
    gate = {
        "run_id": "20260729_28",
        "phase": "capacity_preference_component_ablation",
        "checks": checks,
        "scenario_metrics": metrics.reset_index().to_dict("records"),
        "factorial_summary": summary,
        "retain_capacity_distribution": retain_distribution,
        "retain_current_preference_formula": retain_preference_formula,
        "decision": decision,
        "authorized_next_action": next_action,
        "parameters_calibrated": False,
        "observed_discharge_unit": "m3/s",
        "confirmation_years_used": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 容量分布 × 优先补给组件消融",
        "",
        f"- 结论：`{decision}`",
        f"- 下一动作：`{next_action}`",
        "",
        "## 四情景核心指标",
        "",
        "| 情景 | BFI | BFI误差 | BFI Spearman | NSE | log-NSE | 流量偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, row in metrics.iterrows():
        lines.append(
            f"| {scenario} | {row['median_bfi']:.3f} | "
            f"{row['median_absolute_bfi_error']:.3f} | "
            f"{row['spatial_bfi_spearman']:.3f} | "
            f"{row['median_nse']:.3f} | {row['median_log_nse']:.3f} | "
            f"{row['median_relative_bias']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 归因",
            "",
            (
                "- 仅优先补给解释的组合 BFI 增益比例："
                f"{preference_gain_share:.1%}"
            ),
            (
                "- 容量分布单独 NSE / log-NSE / BFI Spearman 变化："
                f"{summary['distribution_only_nse_delta']:+.3f} / "
                f"{summary['distribution_only_log_nse_delta']:+.3f} / "
                f"{summary['distribution_only_spatial_bfi_delta']:+.3f}"
            ),
            (
                "- 优先补给单独 BFI 误差改善 / BFI Spearman 变化："
                f"{summary['preference_only_bfi_error_improvement']:+.3f} / "
                f"{summary['preference_only_spatial_bfi_delta']:+.3f}"
            ),
            "",
            "- 本轮没有参数率定，所有比较使用观测 `m³/s`。",
        ]
    )
    (REPORT / "technical_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    sources = [
        PARENT_GATE, FORCING, STATIC, PARAMETERS, CAPACITY_CLASSES,
        BASELINE_MODEL, COMBINED_MODEL, OBSERVED_MONTH, OBSERVED_STATION,
        RUN / "experiment_contract.md",
        RUN / "scripts" / "run_component_ablation.py",
    ]
    products = [
        OUTPUTS / "distributed_only_reach_month.parquet",
        OUTPUTS / "preference_only_reach_month.parquet",
        REPORT / "scenario_metrics.csv",
        REPORT / "station_scenario_metrics.csv",
        REPORT / "factorial_effects.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_identity": RUNTIME_IDENTITY,
        "sources": [
            record(path, "ablation_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "ablation_product", "derived")
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
