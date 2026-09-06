"""Reusable conservative Reach-network router with explicit reservoir nodes.

The same router is intended for R0, R1 and R2.  Values supplied to
``route_network_steps`` must be extensive per-step quantities whenever an
enabled reservoir is present.  In exact-parent mode every reservoir rule is
disabled; the identity operator may then be applied directly to discharge
rates without a multiply/divide round trip, preserving bitwise equality with
the frozen parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence
import sys

import numpy as np
import pandas as pd


OPERATOR_DIR = Path(r"E:\SPARROW\5_Test\20260828_14\scripts")
if str(OPERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(OPERATOR_DIR))

from reservoir_operator import (  # noqa: E402
    ReservoirFluxes,
    ReservoirRule,
    ReservoirState,
    ReservoirStepResult,
    step_reservoir,
)


@dataclass(frozen=True)
class ReachRoutingGraph:
    reach_ids: tuple[int, ...]
    topological_order: tuple[int, ...]
    downstream: dict[int, tuple[int, float]]

    def validate(self) -> None:
        nodes = set(self.reach_ids)
        if len(nodes) != len(self.reach_ids):
            raise ValueError("Reach IDs must be unique")
        if set(self.topological_order) != nodes:
            raise ValueError("Topological order must contain every Reach exactly once")
        position = {reach: index for index, reach in enumerate(self.topological_order)}
        for reach, (target, fraction) in self.downstream.items():
            if reach not in nodes or target not in nodes:
                raise ValueError("A routing edge points outside the registered Reach set")
            if not 0 < fraction <= 1:
                raise ValueError(f"Invalid routing fraction for Reach {reach}: {fraction}")
            if position[reach] >= position[target]:
                raise ValueError("Reach graph is cyclic or topological order is invalid")


RuleProvider = Callable[[int, pd.Timestamp | None, ReservoirRule], ReservoirRule]


@dataclass
class ReservoirRuntime:
    entity_id: str
    control_reach_ids: tuple[int, ...]
    outflow_reach_id: int
    base_rule: ReservoirRule
    state: ReservoirState = field(default_factory=ReservoirState.empty)
    active_start: pd.Timestamp | None = None
    active_end: pd.Timestamp | None = None
    rule_provider: RuleProvider | None = None
    local_capture_fraction: float = 1.0

    def __post_init__(self) -> None:
        normalized = tuple(int(value) for value in self.control_reach_ids)
        if not normalized:
            raise ValueError("A reservoir runtime needs at least one control Reach")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Reservoir {self.entity_id} repeats a control Reach")
        self.control_reach_ids = normalized
        if not np.isfinite(self.local_capture_fraction):
            raise ValueError("Reservoir local-capture fraction must be finite")
        if not 0 < self.local_capture_fraction <= 1:
            raise ValueError(
                "Reservoir local-capture fraction must be in the interval (0, 1]"
            )
        if len(normalized) > 1 and self.local_capture_fraction != 1.0:
            raise ValueError(
                "A shared multi-arm reservoir cannot also use a partial local-capture "
                "fraction; register explicit arm-specific virtual controls instead"
            )

    def rule_for_step(self, step: int, timestamp: pd.Timestamp | None) -> ReservoirRule:
        active = True
        if timestamp is not None and self.active_start is not None:
            active = active and timestamp >= self.active_start
        if timestamp is not None and self.active_end is not None:
            active = active and timestamp <= self.active_end
        rule = self.base_rule
        if self.rule_provider is not None:
            rule = self.rule_provider(step, timestamp, rule)
        if not active:
            if self.state.storage_m3 > 1e-6:
                raise RuntimeError(
                    f"Inactive reservoir {self.entity_id} cannot retain storage; "
                    "an explicit impoundment/decommission transition is required"
                )
            rule = replace(rule, enabled=False)
        return rule


@dataclass(frozen=True)
class NetworkRoutingResult:
    routed_fast: np.ndarray
    routed_slow: np.ndarray
    routed_direct: np.ndarray
    routed_age_moment: np.ndarray
    reservoir_diagnostics: tuple[dict[str, object], ...]


def build_reach_graph(topology: pd.DataFrame, reach_ids: Sequence[int]) -> ReachRoutingGraph:
    nodes = set(int(value) for value in reach_ids)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topology.itertuples(index=False):
        reach = int(row.reach_id)
        if reach not in nodes or pd.isna(row.downstream_reach):
            continue
        target = int(row.downstream_reach)
        if target not in nodes:
            raise ValueError(f"Reach {reach} points outside the registered domain")
        fraction = float(row.frac)
        if reach in downstream:
            raise ValueError(f"Reach {reach} has more than one downstream edge")
        downstream[reach] = (target, fraction)

    indegree = {reach: 0 for reach in nodes}
    for target, _ in downstream.values():
        indegree[target] += 1
    ready = sorted(reach for reach, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        reach = ready.pop(0)
        order.append(reach)
        if reach in downstream:
            target = downstream[reach][0]
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    graph = ReachRoutingGraph(
        reach_ids=tuple(sorted(nodes)),
        topological_order=tuple(order),
        downstream=downstream,
    )
    graph.validate()
    return graph


def validate_reservoir_runtimes(
    graph: ReachRoutingGraph, reservoirs: Sequence[ReservoirRuntime]
) -> dict[int, ReservoirRuntime]:
    by_control: dict[int, ReservoirRuntime] = {}
    entity_ids: set[str] = set()
    for runtime in reservoirs:
        if runtime.entity_id in entity_ids:
            raise ValueError(f"Duplicate reservoir entity {runtime.entity_id}")
        entity_ids.add(runtime.entity_id)
        for control_reach_id in runtime.control_reach_ids:
            if control_reach_id in by_control:
                raise ValueError(
                    f"Multiple reservoir states are assigned to Reach {control_reach_id}"
                )
            expected = graph.downstream.get(control_reach_id)
            if expected is None or expected[0] != runtime.outflow_reach_id:
                raise ValueError(
                    f"Reservoir {runtime.entity_id} control edge "
                    f"{control_reach_id}->{runtime.outflow_reach_id} is not a real Reach edge"
                )
            if expected[1] != 1.0:
                raise ValueError(
                    f"Reservoir {runtime.entity_id} control Reach {control_reach_id} has "
                    f"routing fraction {expected[1]}; split-flow reservoir controls require "
                    "an explicit diversion operator"
                )
            by_control[control_reach_id] = runtime
        positions = {reach: index for index, reach in enumerate(graph.topological_order)}
        if any(
            positions[control_reach_id] >= positions[runtime.outflow_reach_id]
            for control_reach_id in runtime.control_reach_ids
        ):
            raise ValueError(
                f"Reservoir {runtime.entity_id} must be evaluated upstream of its outflow Reach"
            )
        runtime.base_rule.validate()
        runtime.state.validate()
    return by_control


def route_network_steps(
    local_fast: np.ndarray,
    local_slow: np.ndarray,
    graph: ReachRoutingGraph,
    reservoirs: Sequence[ReservoirRuntime],
    *,
    dates: Sequence[date | datetime | pd.Timestamp] | None = None,
    local_direct: np.ndarray | None = None,
    local_age_moment: np.ndarray | None = None,
    step_days: float = 1.0,
    values_are_volumes_per_step: bool = False,
    keep_reservoir_diagnostics: bool = True,
) -> NetworkRoutingResult:
    """Route local source-class quantities and update each reservoir once.

    Arrays have shape ``(n_steps, n_reaches)`` in ``graph.reach_ids`` order.
    When any active rule is enabled, inputs must be volumes per step.  Disabled
    identity routing can operate on rates, which avoids numerical changes to
    the exact parent comparator.
    """

    fast = np.asarray(local_fast, dtype=np.float64).copy()
    slow = np.asarray(local_slow, dtype=np.float64).copy()
    if fast.shape != slow.shape or fast.ndim != 2:
        raise ValueError("Fast and slow arrays must be equal-shape two-dimensional arrays")
    if fast.shape[1] != len(graph.reach_ids):
        raise ValueError("Array Reach dimension does not match the graph")
    if not np.isfinite(fast).all() or not np.isfinite(slow).all():
        raise ValueError("Local routing arrays contain non-finite values")
    if (fast < 0).any() or (slow < 0).any():
        raise ValueError("Local routing arrays must be nonnegative")

    direct = (
        np.zeros_like(fast)
        if local_direct is None
        else np.asarray(local_direct, dtype=np.float64).copy()
    )
    age = (
        np.zeros_like(fast)
        if local_age_moment is None
        else np.asarray(local_age_moment, dtype=np.float64).copy()
    )
    if direct.shape != fast.shape or age.shape != fast.shape:
        raise ValueError("Direct and age-moment arrays must match fast/slow shape")
    if (direct < 0).any() or (age < 0).any():
        raise ValueError("Direct water and age moments must be nonnegative")

    # Preserve the true incremental contributions.  During routing, ``fast``
    # and peers accumulate upstream water in place.  An interior-dam virtual
    # control must retain every upstream contribution but only the registered
    # fraction of the control Reach's own incremental contribution; the
    # remainder is generated below the dam and bypasses storage.
    local_fast_source = fast.copy()
    local_slow_source = slow.copy()
    local_direct_source = direct.copy()
    local_age_source = age.copy()

    timestamps: list[pd.Timestamp | None]
    if dates is None:
        timestamps = [None] * fast.shape[0]
    else:
        if len(dates) != fast.shape[0]:
            raise ValueError("Dates length does not match the step dimension")
        timestamps = [pd.Timestamp(value) for value in dates]

    graph.validate()
    runtimes = validate_reservoir_runtimes(graph, reservoirs)
    if not values_are_volumes_per_step:
        for runtime in reservoirs:
            if runtime.base_rule.enabled:
                raise ValueError(
                    "Enabled reservoirs require extensive volumes per step; "
                    "discharge-rate routing is allowed only for exact disabled-parent mode"
                )

    reach_index = {reach: index for index, reach in enumerate(graph.reach_ids)}
    diagnostics: list[dict[str, object]] = []
    for step, timestamp in enumerate(timestamps):
        rules_by_entity = {
            runtime.entity_id: runtime.rule_for_step(step, timestamp)
            for runtime in reservoirs
        }
        shared_pending: dict[str, dict[str, object]] = {}
        for reach in graph.topological_order:
            position = reach_index[reach]
            runtime = runtimes.get(reach)
            if runtime is not None:
                rule = rules_by_entity[runtime.entity_id]
                if rule.enabled and not values_are_volumes_per_step:
                    raise ValueError("An active enabled reservoir received discharge rates")

                # A disabled shared reservoir is evaluated as an identity on
                # each real arm.  This preserves the exact addition order of
                # the frozen parent and therefore bitwise parent degeneracy.
                # Once enabled, all arms are buffered, combined, and advanced
                # through one physical storage state exactly once.
                if len(runtime.control_reach_ids) > 1 and rule.enabled:
                    pending = shared_pending.setdefault(
                        runtime.entity_id,
                        {
                            "fast": 0.0,
                            "slow": 0.0,
                            "direct": 0.0,
                            "age": 0.0,
                            "seen": set(),
                        },
                    )
                    pending["fast"] = float(pending["fast"]) + float(fast[step, position])
                    pending["slow"] = float(pending["slow"]) + float(slow[step, position])
                    pending["direct"] = float(pending["direct"]) + float(direct[step, position])
                    pending["age"] = float(pending["age"]) + float(age[step, position])
                    seen = pending["seen"]
                    if not isinstance(seen, set):
                        raise RuntimeError("Internal shared-reservoir Reach tracker is invalid")
                    seen.add(reach)
                    if len(seen) < len(runtime.control_reach_ids):
                        # Do not route this arm separately; its water is now in
                        # the pending physical-reservoir inflow.
                        continue
                    if seen != set(runtime.control_reach_ids):
                        raise RuntimeError(
                            f"Shared reservoir {runtime.entity_id} did not receive its registered arms"
                        )
                    result = step_reservoir(
                        runtime.state,
                        rule,
                        ReservoirFluxes(
                            inflow_fast_m3=float(pending["fast"]),
                            inflow_slow_m3=float(pending["slow"]),
                            inflow_direct_m3=float(pending["direct"]),
                            inflow_age_moment_m3_day=float(pending["age"]),
                        ),
                        step_days=step_days,
                    )
                    runtime.state = result.state_next
                    target_position = reach_index[runtime.outflow_reach_id]
                    fast[step, target_position] += result.release_origin_fast_m3
                    slow[step, target_position] += result.release_origin_slow_m3
                    direct[step, target_position] += result.release_origin_direct_m3
                    age[step, target_position] += result.total_release_age_moment_m3_day
                    if keep_reservoir_diagnostics:
                        diagnostics.append(
                            {
                                "step": step,
                                "date": None if timestamp is None else str(timestamp.date()),
                                "reservoir_entity_id": runtime.entity_id,
                                "control_reach_ids": list(runtime.control_reach_ids),
                                "outflow_reach_id": runtime.outflow_reach_id,
                                "enabled": True,
                                "operator_calls_this_step": 1,
                                "storage_m3": float(result.state_next.storage_m3),
                                "storage_fast_m3": float(result.state_next.fast_water_m3),
                                "storage_slow_m3": float(result.state_next.slow_water_m3),
                                "storage_direct_m3": float(result.state_next.direct_water_m3),
                                "storage_age_moment_m3_day": float(
                                    result.state_next.age_moment_m3_day
                                ),
                                "controlled_release": float(result.controlled_release_m3),
                                "spill": float(result.spill_m3),
                                "total_release": float(result.total_release_m3),
                                "release_origin_fast": float(result.release_origin_fast_m3),
                                "release_origin_slow": float(result.release_origin_slow_m3),
                                "release_origin_direct": float(result.release_origin_direct_m3),
                                "release_age_moment": float(result.total_release_age_moment_m3_day),
                                "mean_release_age_day": result.mean_release_age_day,
                                "mass_balance_error": float(result.mass_balance_error_m3),
                                "fast_balance_error": float(result.fast_balance_error_m3),
                                "slow_balance_error": float(result.slow_balance_error_m3),
                                "direct_balance_error": float(result.direct_balance_error_m3),
                                "age_moment_balance_error": float(
                                    result.age_moment_balance_error_m3_day
                                ),
                                "state_tracer_error": float(result.state_tracer_error_m3),
                                "release_tracer_error": float(result.release_tracer_error_m3),
                            }
                        )
                    # The combined release has already been inserted once at
                    # the real common downstream Reach.  Suppress ordinary arm
                    # routing to prevent double counting.
                    continue

                capture_fraction = runtime.local_capture_fraction
                if rule.enabled and capture_fraction < 1.0:
                    local_fast_here = float(local_fast_source[step, position])
                    local_slow_here = float(local_slow_source[step, position])
                    local_direct_here = float(local_direct_source[step, position])
                    local_age_here = float(local_age_source[step, position])
                    upstream_fast = float(fast[step, position]) - local_fast_here
                    upstream_slow = float(slow[step, position]) - local_slow_here
                    upstream_direct = float(direct[step, position]) - local_direct_here
                    upstream_age = float(age[step, position]) - local_age_here
                    numerical_floor = -1e-9
                    if min(upstream_fast, upstream_slow, upstream_direct, upstream_age) < numerical_floor:
                        raise RuntimeError(
                            f"Reservoir {runtime.entity_id} has a negative accumulated-upstream "
                            "component; local-vs-upstream routing bookkeeping is inconsistent"
                        )
                    upstream_fast = max(0.0, upstream_fast)
                    upstream_slow = max(0.0, upstream_slow)
                    upstream_direct = max(0.0, upstream_direct)
                    upstream_age = max(0.0, upstream_age)
                    captured_fast = upstream_fast + capture_fraction * local_fast_here
                    captured_slow = upstream_slow + capture_fraction * local_slow_here
                    captured_direct = upstream_direct + capture_fraction * local_direct_here
                    captured_age = upstream_age + capture_fraction * local_age_here
                    bypass_fast = (1.0 - capture_fraction) * local_fast_here
                    bypass_slow = (1.0 - capture_fraction) * local_slow_here
                    bypass_direct = (1.0 - capture_fraction) * local_direct_here
                    bypass_age = (1.0 - capture_fraction) * local_age_here
                else:
                    # Preserve the exact frozen-parent arithmetic path when a
                    # reservoir is disabled.  Even an algebraically equivalent
                    # upstream/local subtraction and re-addition changes the
                    # last floating-point bits on the 230-Reach network.
                    captured_fast = float(fast[step, position])
                    captured_slow = float(slow[step, position])
                    captured_direct = float(direct[step, position])
                    captured_age = float(age[step, position])
                    bypass_fast = 0.0
                    bypass_slow = 0.0
                    bypass_direct = 0.0
                    bypass_age = 0.0

                result: ReservoirStepResult = step_reservoir(
                    runtime.state,
                    rule,
                    ReservoirFluxes(
                        inflow_fast_m3=captured_fast,
                        inflow_slow_m3=captured_slow,
                        inflow_direct_m3=captured_direct,
                        inflow_age_moment_m3_day=captured_age,
                    ),
                    step_days=step_days,
                )
                runtime.state = result.state_next
                fast[step, position] = result.release_origin_fast_m3 + bypass_fast
                slow[step, position] = result.release_origin_slow_m3 + bypass_slow
                direct[step, position] = result.release_origin_direct_m3 + bypass_direct
                age[step, position] = result.total_release_age_moment_m3_day + bypass_age
                if keep_reservoir_diagnostics:
                    diagnostics.append(
                        {
                            "step": step,
                            "date": None if timestamp is None else str(timestamp.date()),
                            "reservoir_entity_id": runtime.entity_id,
                            "control_reach_ids": list(runtime.control_reach_ids),
                            "control_reach_id": reach,
                            "outflow_reach_id": runtime.outflow_reach_id,
                            "enabled": bool(rule.enabled),
                            "local_capture_fraction": float(capture_fraction),
                            "captured_inflow": float(
                                captured_fast + captured_slow + captured_direct
                            ),
                            "bypass_inflow": float(
                                bypass_fast + bypass_slow + bypass_direct
                            ),
                            "operator_calls_this_step": (
                                len(runtime.control_reach_ids) if not rule.enabled else 1
                            ),
                            "storage_m3": float(result.state_next.storage_m3),
                            "storage_fast_m3": float(result.state_next.fast_water_m3),
                            "storage_slow_m3": float(result.state_next.slow_water_m3),
                            "storage_direct_m3": float(result.state_next.direct_water_m3),
                            "storage_age_moment_m3_day": float(
                                result.state_next.age_moment_m3_day
                            ),
                            "controlled_release": float(result.controlled_release_m3),
                            "spill": float(result.spill_m3),
                            "total_release": float(result.total_release_m3),
                            "release_origin_fast": float(result.release_origin_fast_m3),
                            "release_origin_slow": float(result.release_origin_slow_m3),
                            "release_origin_direct": float(result.release_origin_direct_m3),
                            "release_age_moment": float(
                                result.total_release_age_moment_m3_day
                            ),
                            "mean_release_age_day": result.mean_release_age_day,
                            "mass_balance_error": float(result.mass_balance_error_m3),
                            "fast_balance_error": float(result.fast_balance_error_m3),
                            "slow_balance_error": float(result.slow_balance_error_m3),
                            "direct_balance_error": float(result.direct_balance_error_m3),
                            "age_moment_balance_error": float(
                                result.age_moment_balance_error_m3_day
                            ),
                            "state_tracer_error": float(result.state_tracer_error_m3),
                            "release_tracer_error": float(result.release_tracer_error_m3),
                        }
                    )

            if reach in graph.downstream:
                target, fraction = graph.downstream[reach]
                target_position = reach_index[target]
                fast[step, target_position] += fraction * fast[step, position]
                slow[step, target_position] += fraction * slow[step, position]
                direct[step, target_position] += fraction * direct[step, position]
                age[step, target_position] += fraction * age[step, position]

    return NetworkRoutingResult(
        routed_fast=fast,
        routed_slow=slow,
        routed_direct=direct,
        routed_age_moment=age,
        reservoir_diagnostics=tuple(diagnostics),
    )
