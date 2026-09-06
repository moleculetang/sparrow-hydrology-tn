"""Test whether the locked R2 state forgets plausible 2006 initial stocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_22"
S20 = ROOT / "5_Test" / "20260828_20"
S21 = ROOT / "5_Test" / "20260828_21"
S14 = ROOT / "5_Test" / "20260828_14"
S15 = ROOT / "5_Test" / "20260828_15"
for folder in [S20 / "scripts", S15 / "scripts", S14 / "scripts"]:
    sys.path.insert(0, str(folder))

from reservoir_development_core import (  # noqa: E402
    PARENT,
    SECONDS_PER_DAY,
    build_runtimes,
    load_static_context,
    metric_summary,
    prepare_reservoir_metadata,
    station_monthly_prediction,
)
from reservoir_network_router import build_reach_graph, route_network_steps  # noqa: E402
from reservoir_operator import ReservoirState  # noqa: E402


TAU = 180.0
ALPHA = 0.25
DRAWDOWN = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def captured_component_fraction(
    context: dict[str, object], metadata: dict[str, dict[str, object]]
) -> dict[str, float]:
    parent = pd.read_parquet(
        PARENT,
        columns=["date", "reach_id", "routed_fast_response_m3_s", "routed_slow_response_m3_s"],
        filters=[("date", "<=", pd.Timestamp("2015-12-31"))],
    ).sort_values(["date", "reach_id"], kind="mergesort")
    dates = pd.Index(parent["date"].drop_duplicates())
    fast = parent["routed_fast_response_m3_s"].to_numpy(float).reshape(len(dates), 230)
    slow = parent["routed_slow_response_m3_s"].to_numpy(float).reshape(len(dates), 230)
    mask = np.asarray((pd.to_datetime(dates).year >= 2010) & (pd.to_datetime(dates).year <= 2015))
    reach_index = {reach: index for index, reach in enumerate(context["reach_ids"])}
    result = {}
    for entity, item in metadata.items():
        controls = tuple(item["control_reaches"])
        fast_mean = sum(float(fast[mask, reach_index[reach]].mean()) for reach in controls)
        slow_mean = sum(float(slow[mask, reach_index[reach]].mean()) for reach in controls)
        result[entity] = fast_mean / max(fast_mean + slow_mean, 1.0e-12)
    return result


def simulate(
    context: dict[str, object],
    metadata: dict[str, dict[str, object]],
    fast_fraction: dict[str, float],
    initial_mode: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    runtimes = build_runtimes(metadata, "R2", TAU, ALPHA, DRAWDOWN)
    for runtime in runtimes:
        if runtime.active_start > pd.Timestamp("2006-01-01") or initial_mode == "empty":
            continue
        if initial_mode == "target_centered":
            storage = runtime.base_rule.target_storage_m3
        elif initial_mode == "high":
            storage = 0.9 * runtime.base_rule.capacity_m3
        else:
            raise ValueError(initial_mode)
        fraction = fast_fraction[runtime.entity_id]
        recovery_days = 1.0 / runtime.base_rule.storage_recovery_per_step
        runtime.state = ReservoirState(
            storage_m3=storage,
            fast_water_m3=storage * fraction,
            slow_water_m3=storage * (1.0 - fraction),
            direct_water_m3=0.0,
            age_moment_m3_day=storage * recovery_days,
        )
    graph = build_reach_graph(context["topology"], context["reach_ids"])
    routed = route_network_steps(
        context["local_fast_rate"] * SECONDS_PER_DAY,
        context["local_slow_rate"] * SECONDS_PER_DAY,
        graph,
        runtimes,
        dates=context["dates"],
        values_are_volumes_per_step=True,
        keep_reservoir_diagnostics=False,
    )
    total = (routed.routed_fast + routed.routed_slow + routed.routed_direct) / SECONDS_PER_DAY
    prediction = station_monthly_prediction(context, total)
    final = {}
    for runtime in runtimes:
        state = runtime.state
        final[runtime.entity_id] = {
            "storage_fraction": state.storage_m3 / runtime.base_rule.capacity_m3,
            "fast_fraction": state.fast_water_m3 / state.storage_m3 if state.storage_m3 > 0 else 0.0,
            "mean_storage_age_day": state.age_moment_m3_day / state.storage_m3 if state.storage_m3 > 0 else 0.0,
        }
    return prediction, final


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    lock = json.loads((S21 / "locks" / "boundary_complete_development_lock.json").read_text(encoding="utf-8"))
    selected = lock["selected_candidates"]["R2"]
    if (selected["tau_day"], selected["inflow_response"], selected["drawdown_fraction"]) != (TAU, ALPHA, DRAWDOWN):
        raise RuntimeError("Stage-21 selected R2 candidate changed")

    context = load_static_context("2018-12-31")
    metadata = prepare_reservoir_metadata(context)
    fast_fraction = captured_component_fraction(context, metadata)
    predictions = {}
    finals = {}
    metrics = {}
    for mode in ["empty", "target_centered", "high"]:
        prediction, final = simulate(context, metadata, fast_fraction, mode)
        predictions[mode] = prediction
        finals[mode] = final
        metrics[mode] = metric_summary(context["obs_monthly"], prediction, (2016, 2018))

    selector = (
        (predictions["empty"].index.get_level_values("year") >= 2016)
        & (predictions["empty"].index.get_level_values("year") <= 2018)
    )
    base = predictions["empty"].loc[selector].to_numpy(float)
    comparisons = {}
    gates = {}
    detail_rows = []
    for mode in ["target_centered", "high"]:
        candidate = predictions[mode].loc[selector].to_numpy(float)
        log_difference = np.abs(np.log1p(candidate) - np.log1p(base))
        storage_difference = max(
            abs(finals[mode][entity]["storage_fraction"] - finals["empty"][entity]["storage_fraction"])
            for entity in metadata
        )
        fast_difference = max(
            abs(finals[mode][entity]["fast_fraction"] - finals["empty"][entity]["fast_fraction"])
            for entity in metadata
        )
        age_difference = max(
            abs(finals[mode][entity]["mean_storage_age_day"] - finals["empty"][entity]["mean_storage_age_day"])
            for entity in metadata
        )
        comparison = {
            "pooled_log_RMSE_change": metrics[mode]["pooled_log_RMSE"] - metrics["empty"]["pooled_log_RMSE"],
            "p99_absolute_station_month_log_prediction_change": float(np.quantile(log_difference, 0.99)),
            "maximum_end_2018_storage_fraction_change": storage_difference,
            "maximum_end_2018_fast_fraction_change": fast_difference,
            "maximum_end_2018_mean_storage_age_change_day": age_difference,
        }
        comparisons[mode] = comparison
        gates[mode] = {
            "pooled_log_RMSE": abs(comparison["pooled_log_RMSE_change"]) <= 0.005,
            "prediction_p99": comparison["p99_absolute_station_month_log_prediction_change"] <= 0.02,
            "storage_fraction": storage_difference <= 0.02,
            "fast_fraction": fast_difference <= 0.02,
            "mean_storage_age": age_difference <= 30.0,
        }
        gates[mode]["all"] = all(gates[mode].values())
        for entity in metadata:
            detail_rows.append(
                {
                    "initial_mode": mode,
                    "reservoir_entity_id": entity,
                    **{f"empty_{key}": value for key, value in finals["empty"][entity].items()},
                    **{f"sensitivity_{key}": value for key, value in finals[mode][entity].items()},
                }
            )
    pd.DataFrame(detail_rows).to_parquet(outputs / "initial_state_reservoir_details.parquet", index=False)
    report = {
        "stage": "20260828_22",
        "status": "PASS_INITIAL_STATE_CONVERGED" if all(value["all"] for value in gates.values()) else "INITIAL_STATE_CONFOUNDED",
        "selected_candidate": selected,
        "selection_metrics": metrics,
        "comparisons": comparisons,
        "gates": gates,
        "held_out_2019_2022_opened": False,
        "tn_observations_opened": False,
        "hashes": {
            "contract": sha256(STAGE / "experiment_contract.json"),
            "stage21_lock": sha256(S21 / "locks" / "boundary_complete_development_lock.json"),
            "runner_code": sha256(Path(__file__)),
        },
    }
    (reports / "initial_state_sensitivity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS_INITIAL_STATE_CONVERGED":
        raise SystemExit("INITIAL_STATE_CONFOUNDED")


if __name__ == "__main__":
    main()
