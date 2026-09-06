"""Deterministic tests for the real 230-Reach reservoir routing layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_network_router import (
    ReservoirRuntime,
    build_reach_graph,
    route_network_steps,
    validate_reservoir_runtimes,
)

import sys


OPERATOR_DIR = Path(r"E:\SPARROW\5_Test\20260828_14\scripts")
if str(OPERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(OPERATOR_DIR))

from reservoir_operator import ReservoirRule, ReservoirState  # noqa: E402


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_15"
TOPOLOGY = (
    ROOT
    / "5_Test"
    / "20260814_1"
    / "inputs"
    / "topology"
    / "topology_edges.csv"
)


def make_rule(*, enabled: bool, release_all: bool = False) -> ReservoirRule:
    return ReservoirRule(
        capacity_m3=1.0e9 if enabled else 0.0,
        dead_storage_m3=0.0,
        target_storage_m3=0.0,
        baseline_release_m3_step=0.0,
        climatological_inflow_m3_step=0.0,
        inflow_response=1.0 if release_all else 0.0,
        storage_recovery_per_step=0.0,
        minimum_release_m3_step=0.0,
        maximum_controlled_release_m3_step=1.0e9 if enabled else 0.0,
        enabled=enabled,
    )


def yantan_runtime(*, enabled: bool, release_all: bool = False) -> ReservoirRuntime:
    return ReservoirRuntime(
        entity_id="GRAND_5718",
        control_reach_ids=(153, 154),
        outflow_reach_id=146,
        base_rule=make_rule(enabled=enabled, release_all=release_all),
        state=ReservoirState.empty(),
    )


def baipenzhu_runtime(*, enabled: bool, release_all: bool = False) -> ReservoirRuntime:
    return ReservoirRuntime(
        entity_id="GRAND_5758",
        control_reach_ids=(17,),
        outflow_reach_id=22,
        base_rule=make_rule(enabled=enabled, release_all=release_all),
        state=ReservoirState.empty(),
        local_capture_fraction=856.0 / 3062.8544538422084,
    )


def parent_route(local: np.ndarray, graph) -> np.ndarray:
    routed = local.copy()
    index = {reach: position for position, reach in enumerate(graph.reach_ids)}
    for reach in graph.topological_order:
        if reach in graph.downstream:
            target, fraction = graph.downstream[reach]
            routed[:, index[target]] += fraction * routed[:, index[reach]]
    return routed


def real_graph():
    frame = pd.read_csv(TOPOLOGY)
    return build_reach_graph(frame, list(range(1, 231)))


def test_real_yantan_shared_edges() -> None:
    graph = real_graph()
    assert graph.downstream[153] == (146, 1.0)
    assert graph.downstream[154] == (146, 1.0)
    validate_reservoir_runtimes(graph, [yantan_runtime(enabled=False)])


def test_real_yantan_named_arms_and_control_area() -> None:
    frame = pd.read_csv(TOPOLOGY).set_index("reach_id")
    assert str(frame.loc[153, "src_id"]).startswith("岩滩水库")
    assert str(frame.loc[154, "src_id"]).startswith("岩滩水库")
    assert int(float(frame.loc[153, "downstream_reach"])) == 146
    assert int(float(frame.loc[154, "downstream_reach"])) == 146
    represented_area = float(frame.loc[153, "tot_area_km2"]) + float(
        frame.loc[154, "tot_area_km2"]
    )
    grand_control_area = 101893.0
    assert 0.9 <= represented_area / grand_control_area <= 1.0


def test_disabled_shared_reservoir_bitwise_parent() -> None:
    graph = real_graph()
    rng = np.random.default_rng(20260830)
    fast = rng.uniform(0.0, 1000.0, size=(4, 230))
    slow = rng.uniform(0.0, 500.0, size=(4, 230))
    expected_fast = parent_route(fast, graph)
    expected_slow = parent_route(slow, graph)
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [yantan_runtime(enabled=False)],
        values_are_volumes_per_step=False,
        keep_reservoir_diagnostics=False,
    )
    assert np.array_equal(routed.routed_fast, expected_fast)
    assert np.array_equal(routed.routed_slow, expected_slow)
    assert np.count_nonzero(routed.routed_direct) == 0
    assert np.count_nonzero(routed.routed_age_moment) == 0


def test_enabled_shared_reservoir_single_release_and_tracers() -> None:
    graph = real_graph()
    index = {reach: position for position, reach in enumerate(graph.reach_ids)}
    fast = np.zeros((1, 230), dtype=np.float64)
    slow = np.zeros_like(fast)
    direct = np.zeros_like(fast)
    age = np.zeros_like(fast)
    fast[0, index[153]], fast[0, index[154]] = 100.0, 30.0
    slow[0, index[153]], slow[0, index[154]] = 40.0, 20.0
    direct[0, index[153]], direct[0, index[154]] = 10.0, 5.0
    age[0, index[153]], age[0, index[154]] = 500.0, 250.0
    runtime = yantan_runtime(enabled=True, release_all=True)
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [runtime],
        dates=[pd.Timestamp("2010-01-01")],
        local_direct=direct,
        local_age_moment=age,
        values_are_volumes_per_step=True,
    )
    assert routed.routed_fast[0, index[146]] == 130.0
    assert routed.routed_slow[0, index[146]] == 60.0
    assert routed.routed_direct[0, index[146]] == 15.0
    assert routed.routed_age_moment[0, index[146]] == 750.0
    assert runtime.state.storage_m3 == 0.0
    rows = [
        row
        for row in routed.reservoir_diagnostics
        if row["reservoir_entity_id"] == "GRAND_5718"
    ]
    assert len(rows) == 1
    assert rows[0]["operator_calls_this_step"] == 1
    assert rows[0]["control_reach_ids"] == [153, 154]
    assert rows[0]["release_origin_fast"] == 130.0
    assert rows[0]["release_origin_slow"] == 60.0
    assert rows[0]["release_origin_direct"] == 15.0


def test_enabled_shared_reservoir_retains_one_conservative_stock() -> None:
    graph = real_graph()
    index = {reach: position for position, reach in enumerate(graph.reach_ids)}
    fast = np.zeros((1, 230), dtype=np.float64)
    slow = np.zeros_like(fast)
    direct = np.zeros_like(fast)
    age = np.zeros_like(fast)
    fast[0, index[153]], fast[0, index[154]] = 3.0, 7.0
    slow[0, index[153]], slow[0, index[154]] = 11.0, 13.0
    direct[0, index[153]], direct[0, index[154]] = 17.0, 19.0
    age[0, index[153]], age[0, index[154]] = 23.0, 29.0
    runtime = yantan_runtime(enabled=True, release_all=False)
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [runtime],
        dates=[pd.Timestamp("2010-01-01")],
        local_direct=direct,
        local_age_moment=age,
        values_are_volumes_per_step=True,
    )
    assert runtime.state.fast_water_m3 == 10.0
    assert runtime.state.slow_water_m3 == 24.0
    assert runtime.state.direct_water_m3 == 36.0
    assert runtime.state.storage_m3 == 70.0
    assert runtime.state.age_moment_m3_day == 52.0
    assert routed.routed_fast[0, index[146]] == 0.0
    assert routed.routed_slow[0, index[146]] == 0.0
    assert routed.routed_direct[0, index[146]] == 0.0
    assert len(routed.reservoir_diagnostics) == 1


def test_duplicate_control_reach_is_rejected() -> None:
    graph = real_graph()
    duplicate = ReservoirRuntime(
        entity_id="DUPLICATE",
        control_reach_ids=(153,),
        outflow_reach_id=146,
        base_rule=make_rule(enabled=False),
    )
    try:
        validate_reservoir_runtimes(
            graph,
            [yantan_runtime(enabled=False), duplicate],
        )
    except ValueError as error:
        assert "Multiple reservoir states" in str(error)
    else:
        raise AssertionError("Duplicate control Reach was accepted")


def test_mismatched_shared_outflow_is_rejected() -> None:
    graph = real_graph()
    invalid = ReservoirRuntime(
        entity_id="GRAND_5718",
        control_reach_ids=(153, 154),
        outflow_reach_id=145,
        base_rule=make_rule(enabled=False),
    )
    try:
        validate_reservoir_runtimes(graph, [invalid])
    except ValueError as error:
        assert "is not a real Reach edge" in str(error)
    else:
        raise AssertionError("Mismatched shared-reservoir outflow was accepted")


def test_disabled_partial_capture_is_bitwise_parent() -> None:
    graph = real_graph()
    rng = np.random.default_rng(5758)
    fast = rng.uniform(0.0, 1000.0, size=(3, 230))
    slow = rng.uniform(0.0, 500.0, size=(3, 230))
    expected_fast = parent_route(fast, graph)
    expected_slow = parent_route(slow, graph)
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [baipenzhu_runtime(enabled=False)],
        values_are_volumes_per_step=False,
        keep_reservoir_diagnostics=False,
    )
    assert np.array_equal(routed.routed_fast, expected_fast)
    assert np.array_equal(routed.routed_slow, expected_slow)


def test_baipenzhu_partial_local_capture_and_bypass_close() -> None:
    graph = real_graph()
    index = {reach: position for position, reach in enumerate(graph.reach_ids)}
    fast = np.zeros((1, 230), dtype=np.float64)
    slow = np.zeros_like(fast)
    direct = np.zeros_like(fast)
    age = np.zeros_like(fast)
    fast[0, index[17]] = 100.0
    slow[0, index[17]] = 40.0
    direct[0, index[17]] = 10.0
    age[0, index[17]] = 500.0
    runtime = baipenzhu_runtime(enabled=True, release_all=False)
    fraction = runtime.local_capture_fraction
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [runtime],
        dates=[pd.Timestamp("2010-01-01")],
        local_direct=direct,
        local_age_moment=age,
        values_are_volumes_per_step=True,
    )
    assert np.isclose(runtime.state.fast_water_m3, 100.0 * fraction)
    assert np.isclose(runtime.state.slow_water_m3, 40.0 * fraction)
    assert np.isclose(runtime.state.direct_water_m3, 10.0 * fraction)
    assert np.isclose(runtime.state.age_moment_m3_day, 500.0 * fraction)
    assert np.isclose(routed.routed_fast[0, index[22]], 100.0 * (1.0 - fraction))
    assert np.isclose(routed.routed_slow[0, index[22]], 40.0 * (1.0 - fraction))
    assert np.isclose(routed.routed_direct[0, index[22]], 10.0 * (1.0 - fraction))
    assert np.isclose(
        routed.routed_age_moment[0, index[22]], 500.0 * (1.0 - fraction)
    )
    row = routed.reservoir_diagnostics[0]
    assert np.isclose(row["local_capture_fraction"], fraction)
    assert np.isclose(row["captured_inflow"] + row["bypass_inflow"], 150.0)


def test_partial_capture_retains_all_upstream_water() -> None:
    graph = real_graph()
    index = {reach: position for position, reach in enumerate(graph.reach_ids)}
    fast = np.zeros((1, 230), dtype=np.float64)
    slow = np.zeros_like(fast)
    fast[0, index[91]] = 20.0
    fast[0, index[86]] = 100.0
    runtime = ReservoirRuntime(
        entity_id="GRAND_5703_TEST",
        control_reach_ids=(86,),
        outflow_reach_id=50,
        base_rule=make_rule(enabled=True, release_all=False),
        state=ReservoirState.empty(),
        local_capture_fraction=0.25,
    )
    routed = route_network_steps(
        fast,
        slow,
        graph,
        [runtime],
        dates=[pd.Timestamp("2010-01-01")],
        values_are_volumes_per_step=True,
    )
    assert np.isclose(runtime.state.fast_water_m3, 45.0)
    assert np.isclose(routed.routed_fast[0, index[50]], 75.0)


def test_invalid_partial_capture_is_rejected() -> None:
    for value in (0.0, -0.1, 1.1, float("nan")):
        try:
            ReservoirRuntime(
                entity_id="INVALID_PARTIAL",
                control_reach_ids=(17,),
                outflow_reach_id=22,
                base_rule=make_rule(enabled=False),
                local_capture_fraction=value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid local-capture fraction {value} was accepted")


TESTS = [
    test_real_yantan_shared_edges,
    test_real_yantan_named_arms_and_control_area,
    test_disabled_shared_reservoir_bitwise_parent,
    test_enabled_shared_reservoir_single_release_and_tracers,
    test_enabled_shared_reservoir_retains_one_conservative_stock,
    test_duplicate_control_reach_is_rejected,
    test_mismatched_shared_outflow_is_rejected,
    test_disabled_partial_capture_is_bitwise_parent,
    test_baipenzhu_partial_local_capture_and_bypass_close,
    test_partial_capture_retains_all_upstream_water,
    test_invalid_partial_capture_is_rejected,
]


def main() -> None:
    results = []
    for test in TESTS:
        test()
        results.append({"test": test.__name__, "status": "PASS"})
    report = {
        "stage": "20260828_15",
        "status": "PASS",
        "tests": results,
        "real_topology_source": str(TOPOLOGY),
        "shared_reservoir_entity": "GRAND_5718",
        "shared_control_reaches": [153, 154],
        "unique_outflow_reach": 146,
        "empirical_observations_read": False,
        "parameters_fitted": False,
    }
    output = STAGE / "reports" / "reservoir_network_router_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
