"""Run deterministic synthetic tests for the reservoir operator."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


STAGE = Path(r"E:\SPARROW\5_Test\20260828_14")
sys.path.insert(0, str(STAGE / "scripts"))

from reservoir_operator import (  # noqa: E402
    ReservoirFluxes,
    ReservoirRule,
    ReservoirState,
    generate_calendar_seasonal_target_storage_curve,
    generate_seasonal_target_storage_curve,
    step_reservoir,
    validate_operator_registry,
)


def rule(**updates: float | bool) -> ReservoirRule:
    values = {
        "capacity_m3": 100.0,
        "dead_storage_m3": 0.0,
        "target_storage_m3": 50.0,
        "baseline_release_m3_step": 10.0,
        "climatological_inflow_m3_step": 10.0,
        "inflow_response": 0.2,
        "storage_recovery_per_step": 0.1,
        "minimum_release_m3_step": 0.0,
        "maximum_controlled_release_m3_step": 100.0,
        "enabled": True,
    }
    values.update(updates)
    return ReservoirRule(**values)


def close(a: float, b: float, tol: float = 1.0e-9) -> None:
    assert abs(a - b) <= tol, (a, b)


def test_parent_exact_passthrough() -> None:
    out = step_reservoir(
        ReservoirState.empty(),
        rule(enabled=False, capacity_m3=0.0, target_storage_m3=0.0),
        ReservoirFluxes(7.0, 3.0, 2.0, inflow_age_moment_m3_day=60.0),
    )
    close(out.total_release_m3, 12.0)
    close(out.release_origin_fast_m3, 7.0)
    close(out.release_origin_slow_m3, 3.0)
    close(out.release_origin_direct_m3, 2.0)
    close(out.total_release_age_moment_m3_day, 60.0)
    close(out.mean_release_age_day or -1.0, 5.0)
    close(out.state_next.storage_m3, 0.0)


def test_parent_rejects_reservoir_only_fluxes() -> None:
    try:
        step_reservoir(
            ReservoirState.empty(),
            rule(enabled=False, capacity_m3=0.0, target_storage_m3=0.0),
            ReservoirFluxes(7.0, 3.0, precipitation_m3=1.0),
        )
    except ValueError as exc:
        assert "P/E/W" in str(exc)
    else:
        raise AssertionError("Disabled exact-parent operator accepted reservoir precipitation")


def test_flux_rejects_ghost_inflow_age() -> None:
    try:
        step_reservoir(
            ReservoirState.empty(),
            rule(),
            ReservoirFluxes(0.0, 0.0, inflow_age_moment_m3_day=10.0),
        )
    except ValueError as exc:
        assert "nonzero age moment" in str(exc)
    else:
        raise AssertionError("Zero inflow carried a nonzero upstream age moment")


def test_full_mass_balance_with_losses() -> None:
    state = ReservoirState(60.0, 18.0, 36.0, 6.0, 240.0)
    out = step_reservoir(
        state,
        rule(),
        ReservoirFluxes(
            20.0,
            5.0,
            1.0,
            precipitation_m3=4.0,
            evaporation_demand_m3=3.0,
            withdrawal_demand_m3=7.0,
        ),
    )
    close(out.mass_balance_error_m3, 0.0)
    close(out.age_moment_balance_error_m3_day, 0.0)
    close(
        state.storage_m3 + 26.0 + 4.0,
        out.state_next.storage_m3
        + out.total_release_m3
        + out.evaporation_actual_m3
        + out.withdrawal_actual_m3,
    )


def test_spill_and_capacity() -> None:
    out = step_reservoir(
        ReservoirState(95.0, 30.0, 60.0, 5.0, 0.0),
        rule(baseline_release_m3_step=0.0, storage_recovery_per_step=0.0),
        ReservoirFluxes(20.0, 10.0),
    )
    assert out.spill_m3 > 0
    close(out.state_next.storage_m3, 100.0)
    close(
        out.controlled_release_age_moment_m3_day
        + out.spill_age_moment_m3_day,
        out.total_release_age_moment_m3_day,
    )
    close(
        out.controlled_release_origin_fast_m3 + out.spill_origin_fast_m3,
        out.release_origin_fast_m3,
    )


def test_fast_slow_direct_tracer_closure() -> None:
    out = step_reservoir(
        ReservoirState(50.0, 10.0, 35.0, 5.0, 0.0),
        rule(),
        ReservoirFluxes(10.0, 20.0, 5.0, precipitation_m3=5.0),
    )
    close(
        out.state_next.fast_water_m3
        + out.state_next.slow_water_m3
        + out.state_next.direct_water_m3,
        out.state_next.storage_m3,
    )
    close(
        out.release_origin_fast_m3
        + out.release_origin_slow_m3
        + out.release_origin_direct_m3,
        out.total_release_m3,
    )
    close(out.fast_balance_error_m3, 0.0)
    close(out.slow_balance_error_m3, 0.0)
    close(out.direct_balance_error_m3, 0.0)


def test_release_uses_storage_mixture_not_current_inflow_fraction() -> None:
    out = step_reservoir(
        ReservoirState(90.0, 0.0, 90.0, 0.0, 0.0),
        rule(
            capacity_m3=200.0,
            target_storage_m3=90.0,
            baseline_release_m3_step=10.0,
            climatological_inflow_m3_step=10.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
            maximum_controlled_release_m3_step=10.0,
        ),
        ReservoirFluxes(10.0, 0.0),
    )
    close(out.release_origin_fast_m3 / out.total_release_m3, 0.1)
    close(out.release_origin_slow_m3 / out.total_release_m3, 0.9)


def test_withdrawal_respects_intake_floor() -> None:
    out = step_reservoir(
        ReservoirState(60.0, 20.0, 40.0, 0.0, 0.0),
        rule(
            dead_storage_m3=20.0,
            target_storage_m3=40.0,
            withdrawal_floor_m3=50.0,
            baseline_release_m3_step=0.0,
            storage_recovery_per_step=0.0,
        ),
        ReservoirFluxes(0.0, 0.0, withdrawal_demand_m3=30.0),
    )
    close(out.withdrawal_actual_m3, 10.0)
    close(out.state_next.storage_m3, 50.0)


def test_age_moment_update() -> None:
    out = step_reservoir(
        ReservoirState(100.0, 50.0, 50.0, 0.0, 1000.0),
        rule(
            capacity_m3=200.0,
            target_storage_m3=90.0,
            baseline_release_m3_step=10.0,
            climatological_inflow_m3_step=0.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
            maximum_controlled_release_m3_step=10.0,
        ),
        ReservoirFluxes(0.0, 0.0),
    )
    close(out.mean_release_age_day or -1.0, 11.0)
    close(out.state_next.age_moment_m3_day, 990.0)
    close(out.controlled_release_age_moment_m3_day, 110.0)
    close(out.age_moment_balance_error_m3_day, 0.0)


def test_age_moment_outputs_with_all_removal_paths() -> None:
    out = step_reservoir(
        ReservoirState(90.0, 30.0, 50.0, 10.0, 900.0),
        rule(
            capacity_m3=80.0,
            dead_storage_m3=10.0,
            target_storage_m3=60.0,
            baseline_release_m3_step=8.0,
            climatological_inflow_m3_step=20.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
            maximum_controlled_release_m3_step=8.0,
            withdrawal_floor_m3=10.0,
        ),
        ReservoirFluxes(
            10.0,
            5.0,
            5.0,
            precipitation_m3=2.0,
            evaporation_demand_m3=3.0,
            withdrawal_demand_m3=4.0,
            inflow_age_moment_m3_day=80.0,
        ),
    )
    assert out.controlled_release_age_moment_m3_day > 0
    assert out.spill_age_moment_m3_day > 0
    assert out.evaporation_age_moment_m3_day > 0
    assert out.withdrawal_age_moment_m3_day > 0
    close(
        900.0 + 90.0 + 80.0,
        out.state_next.age_moment_m3_day
        + out.total_release_age_moment_m3_day
        + out.evaporation_age_moment_m3_day
        + out.withdrawal_age_moment_m3_day,
    )
    close(out.age_moment_balance_error_m3_day, 0.0)


def test_series_mass_conservation() -> None:
    state = ReservoirState.empty()
    total_in = 0.0
    total_out = 0.0
    for day in range(365):
        fast = 6.0 + (day % 7)
        slow = 3.0
        total_in += fast + slow
        out = step_reservoir(state, rule(), ReservoirFluxes(fast, slow))
        total_out += out.total_release_m3
        state = out.state_next
    close(total_in, total_out + state.storage_m3, tol=1.0e-7)


def test_cascade_no_double_count() -> None:
    upstream = step_reservoir(
        ReservoirState(50.0, 20.0, 20.0, 10.0, 150.0),
        rule(
            capacity_m3=70.0,
            target_storage_m3=50.0,
            baseline_release_m3_step=30.0,
            climatological_inflow_m3_step=105.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
            withdrawal_floor_m3=10.0,
        ),
        ReservoirFluxes(
            60.0,
            40.0,
            5.0,
            precipitation_m3=2.0,
            evaporation_demand_m3=1.0,
            withdrawal_demand_m3=4.0,
        ),
    )
    downstream = step_reservoir(
        ReservoirState(20.0, 5.0, 10.0, 5.0, 40.0),
        rule(
            capacity_m3=80.0,
            target_storage_m3=30.0,
            baseline_release_m3_step=20.0,
            climatological_inflow_m3_step=upstream.total_release_m3 + 5.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
            withdrawal_floor_m3=5.0,
        ),
        ReservoirFluxes(
            upstream.release_origin_fast_m3 + 3.0,
            upstream.release_origin_slow_m3 + 2.0,
            upstream.release_origin_direct_m3,
            precipitation_m3=3.0,
            evaporation_demand_m3=1.0,
            withdrawal_demand_m3=2.0,
            inflow_age_moment_m3_day=upstream.total_release_age_moment_m3_day,
        ),
    )
    external_initial_and_input = 50.0 + 105.0 + 2.0 + 20.0 + 5.0 + 3.0
    external_removals_and_final = (
        upstream.evaporation_actual_m3
        + upstream.withdrawal_actual_m3
        + downstream.evaporation_actual_m3
        + downstream.withdrawal_actual_m3
        + upstream.state_next.storage_m3
        + downstream.state_next.storage_m3
        + downstream.total_release_m3
    )
    close(
        external_initial_and_input,
        external_removals_and_final,
    )
    external_initial_age_and_aging = 150.0 + 50.0 + 40.0 + 20.0
    external_age_removals_and_final = (
        upstream.evaporation_age_moment_m3_day
        + upstream.withdrawal_age_moment_m3_day
        + downstream.evaporation_age_moment_m3_day
        + downstream.withdrawal_age_moment_m3_day
        + downstream.total_release_age_moment_m3_day
        + upstream.state_next.age_moment_m3_day
        + downstream.state_next.age_moment_m3_day
    )
    close(external_initial_age_and_aging, external_age_removals_and_final)
    assert downstream.mean_release_age_day is not None
    assert downstream.mean_release_age_day > 0


def test_parallel_branches_merge_once() -> None:
    left = step_reservoir(
        ReservoirState.empty(),
        rule(enabled=False, capacity_m3=0.0, target_storage_m3=0.0),
        ReservoirFluxes(12.0, 3.0),
    )
    right = step_reservoir(
        ReservoirState.empty(),
        rule(enabled=False, capacity_m3=0.0, target_storage_m3=0.0),
        ReservoirFluxes(5.0, 10.0),
    )
    shared = step_reservoir(
        ReservoirState.empty(),
        rule(
            capacity_m3=40.0,
            target_storage_m3=20.0,
            baseline_release_m3_step=10.0,
            climatological_inflow_m3_step=30.0,
            inflow_response=0.0,
            storage_recovery_per_step=0.0,
        ),
        ReservoirFluxes(
            left.release_origin_fast_m3 + right.release_origin_fast_m3,
            left.release_origin_slow_m3 + right.release_origin_slow_m3,
            left.release_origin_direct_m3 + right.release_origin_direct_m3,
        ),
    )
    close(30.0, shared.state_next.storage_m3 + shared.total_release_m3)


def test_operator_registry_uniqueness_and_acyclicity() -> None:
    order = validate_operator_registry(
        [
            {
                "reservoir_entity_id": "upper",
                "operator_inflow_reaches": [10, 11],
                "unique_outflow_reach": 20,
            },
            {
                "reservoir_entity_id": "lower",
                "operator_inflow_reaches": [20],
                "unique_outflow_reach": 30,
            },
        ],
        {10: 20, 11: 20, 20: 30, 30: None},
    )
    assert order == ["upper", "lower"]
    invalid = [
        {
            "reservoir_entity_id": "duplicate_a",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 20,
        },
        {
            "reservoir_entity_id": "duplicate_b",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 30,
        },
    ]
    try:
        validate_operator_registry(invalid, {10: 20, 20: None, 30: None})
    except ValueError as exc:
        assert "controlled by both" in str(exc)
    else:
        raise AssertionError("Duplicate reservoir control edge was accepted")


def test_operator_registry_real_network_and_spaced_cascade() -> None:
    records = [
        {
            "reservoir_entity_id": "upper",
            "operator_inflow_reaches": [10, 11],
            "unique_outflow_reach": 12,
        },
        {
            "reservoir_entity_id": "lower",
            "operator_inflow_reaches": [21],
            "unique_outflow_reach": 30,
        },
    ]
    downstream = {10: 12, 11: 12, 12: 20, 20: 21, 21: 30, 30: None}
    assert validate_operator_registry(records, downstream) == ["upper", "lower"]

    immediate_records = [
        {
            "reservoir_entity_id": "upper",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 20,
        },
        {
            "reservoir_entity_id": "lower",
            "operator_inflow_reaches": [20],
            "unique_outflow_reach": 30,
        },
    ]
    assert validate_operator_registry(
        immediate_records, {10: 20, 20: 30, 30: None}
    ) == ["upper", "lower"]


def test_operator_registry_rejects_skipped_and_mismatched_control_edges() -> None:
    skipped_reach = [
        {
            "reservoir_entity_id": "skipped_reach",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 20,
        }
    ]
    try:
        validate_operator_registry(skipped_reach, {10: 12, 12: 20, 20: None})
    except ValueError as exc:
        assert "disagrees with locked topology" in str(exc)
    else:
        raise AssertionError("A control edge spanning an ordinary Reach was accepted")

    mismatched = [
        {
            "reservoir_entity_id": "wrong_outlet",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 30,
        }
    ]
    try:
        validate_operator_registry(mismatched, {10: 20, 20: 30, 30: None})
    except ValueError as exc:
        assert "disagrees with locked topology" in str(exc)
    else:
        raise AssertionError("Incorrect immediate control edge was accepted")


def test_operator_registry_requires_real_downstream_map() -> None:
    records = [
        {
            "reservoir_entity_id": "requires_map",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 20,
        }
    ]
    try:
        validate_operator_registry(records)
    except ValueError as exc:
        assert "real downstream_map is required" in str(exc)
    else:
        raise AssertionError("Registry validation ran without the locked Reach topology")


def test_operator_registry_rejects_real_reach_cycle() -> None:
    records = [
        {
            "reservoir_entity_id": "cycle_test",
            "operator_inflow_reaches": [10],
            "unique_outflow_reach": 30,
        }
    ]
    try:
        validate_operator_registry(records, {10: 20, 20: 10, 30: None})
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("Cyclic real Reach network was accepted")


def test_low_dimensional_seasonal_target_curve() -> None:
    curve = generate_seasonal_target_storage_curve(
        capacity_m3=1000.0,
        dead_storage_m3=200.0,
        baseline_active_fraction=0.70,
        annual_harmonic_amplitude_fraction=0.05,
        annual_peak_day=30,
        flood_start_day=150,
        flood_end_day=240,
        flood_drawdown_fraction=0.20,
        transition_days=15.0,
    )
    assert len(curve) == 365
    assert min(curve) >= 200.0
    assert max(curve) <= 1000.0
    assert sum(curve) / len(curve) < 200.0 + 800.0 * 0.70
    flood_mean = sum(curve[159:220]) / len(curve[159:220])
    nonflood_mean = sum(curve[59:120]) / len(curve[59:120])
    assert flood_mean < nonflood_mean
    repeated = generate_seasonal_target_storage_curve(
        capacity_m3=1000.0,
        dead_storage_m3=200.0,
        baseline_active_fraction=0.70,
        annual_harmonic_amplitude_fraction=0.05,
        annual_peak_day=30,
        flood_start_day=150,
        flood_end_day=240,
        flood_drawdown_fraction=0.20,
        transition_days=15.0,
    )
    assert curve == repeated


def test_seasonal_target_curve_rejects_unphysical_parameters() -> None:
    try:
        generate_seasonal_target_storage_curve(
            capacity_m3=1000.0,
            dead_storage_m3=200.0,
            baseline_active_fraction=0.10,
            annual_harmonic_amplitude_fraction=0.20,
            annual_peak_day=30,
            flood_start_day=150,
            flood_end_day=240,
            flood_drawdown_fraction=0.40,
        )
    except ValueError as exc:
        assert "physical storage bounds" in str(exc)
    else:
        raise AssertionError("Unphysical seasonal target parameters were accepted")


def test_seasonal_target_curve_supports_leap_year_wraparound_window() -> None:
    curve = generate_seasonal_target_storage_curve(
        capacity_m3=1000.0,
        dead_storage_m3=100.0,
        baseline_active_fraction=0.70,
        annual_harmonic_amplitude_fraction=0.0,
        annual_peak_day=1,
        flood_start_day=330,
        flood_end_day=60,
        flood_drawdown_fraction=0.20,
        transition_days=10.0,
        days_in_year=366,
    )
    assert len(curve) == 366
    assert curve[0] < curve[199]
    assert curve[350] < curve[199]


def test_calendar_target_curve_preserves_month_day_phase_across_leap_year() -> None:
    kwargs = {
        "capacity_m3": 1000.0,
        "dead_storage_m3": 100.0,
        "baseline_active_fraction": 0.70,
        "annual_harmonic_amplitude_fraction": 0.0,
        "annual_peak_month_day": (1, 15),
        "flood_start_month_day": (5, 1),
        "flood_end_month_day": (8, 31),
        "flood_drawdown_fraction": 0.20,
        "transition_days": 10.0,
    }
    common = generate_calendar_seasonal_target_storage_curve(year=2023, **kwargs)
    leap = generate_calendar_seasonal_target_storage_curve(year=2024, **kwargs)
    assert len(common) == 365
    assert len(leap) == 366

    def value_on(curve: tuple[float, ...], year: int, month: int, day: int) -> float:
        index = date(year, month, day).timetuple().tm_yday - 1
        return curve[index]

    for month, day in [(4, 25), (5, 1), (6, 15), (8, 31), (9, 5), (10, 1)]:
        close(
            value_on(common, 2023, month, day),
            value_on(leap, 2024, month, day),
        )


def test_large_reservoir_scale_relative_closure() -> None:
    state = ReservoirState(1.0e10, 2.0e9, 7.0e9, 1.0e9, 3.0e12)
    out = step_reservoir(
        state,
        rule(
            capacity_m3=1.4e10,
            dead_storage_m3=2.0e9,
            target_storage_m3=1.0e10,
            baseline_release_m3_step=1.0e8,
            climatological_inflow_m3_step=1.0e8,
            maximum_controlled_release_m3_step=5.0e8,
        ),
        ReservoirFluxes(8.0e7, 2.0e7, precipitation_m3=1.0e6),
    )
    assert out.mass_balance_relative_error <= 1.0e-12
    assert out.fast_balance_relative_error <= 1.0e-12
    assert out.slow_balance_relative_error <= 1.0e-12
    assert out.direct_balance_relative_error <= 1.0e-12
    assert out.age_moment_balance_relative_error <= 1.0e-12
    assert out.state_tracer_relative_error <= 1.0e-12
    assert out.release_tracer_relative_error <= 1.0e-12


TESTS = [
    test_parent_exact_passthrough,
    test_parent_rejects_reservoir_only_fluxes,
    test_flux_rejects_ghost_inflow_age,
    test_full_mass_balance_with_losses,
    test_spill_and_capacity,
    test_fast_slow_direct_tracer_closure,
    test_release_uses_storage_mixture_not_current_inflow_fraction,
    test_withdrawal_respects_intake_floor,
    test_age_moment_update,
    test_age_moment_outputs_with_all_removal_paths,
    test_series_mass_conservation,
    test_cascade_no_double_count,
    test_parallel_branches_merge_once,
    test_operator_registry_uniqueness_and_acyclicity,
    test_operator_registry_real_network_and_spaced_cascade,
    test_operator_registry_rejects_skipped_and_mismatched_control_edges,
    test_operator_registry_requires_real_downstream_map,
    test_operator_registry_rejects_real_reach_cycle,
    test_low_dimensional_seasonal_target_curve,
    test_seasonal_target_curve_rejects_unphysical_parameters,
    test_seasonal_target_curve_supports_leap_year_wraparound_window,
    test_calendar_target_curve_preserves_month_day_phase_across_leap_year,
    test_large_reservoir_scale_relative_closure,
]


def main() -> None:
    results = []
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # pragma: no cover - explicit test reporting
            results.append({"test": test.__name__, "status": "FAIL", "error": repr(exc)})
        else:
            results.append({"test": test.__name__, "status": "PASS"})
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    report = {
        "stage": "20260828_14",
        "status": status,
        "absolute_tolerance_m3": 1.0e-6,
        "relative_tolerance": 1.0e-12,
        "tests": results,
        "empirical_observations_read": False,
        "parameters_fitted": False,
    }
    out_dir = STAGE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
