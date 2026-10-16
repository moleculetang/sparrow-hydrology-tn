"""State-consistent daily reservoir water-balance operator.

The operator is deliberately independent of any optimizer.  Literature and
observations may later define priors for the parameters, but cannot alter the
mass-balance algebra implemented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import cos, isfinite, pi


ABS_TOL_M3 = 1.0e-6
REL_TOL = 1.0e-12


def _allowed_error(scale: float, absolute: float = ABS_TOL_M3, relative: float = REL_TOL) -> float:
    return absolute + relative * max(abs(scale), 1.0)


def _assert_finite(values: dict[str, float], label: str) -> None:
    bad = {name: value for name, value in values.items() if not isfinite(value)}
    if bad:
        raise ValueError(f"Non-finite {label}: {bad}")


@dataclass(frozen=True)
class ReservoirState:
    storage_m3: float
    fast_water_m3: float
    slow_water_m3: float
    direct_water_m3: float
    age_moment_m3_day: float

    @classmethod
    def empty(cls) -> "ReservoirState":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    def validate(self, tolerance: float = ABS_TOL_M3) -> None:
        values = asdict(self)
        _assert_finite(values, "reservoir state")
        if any(value < -tolerance for value in values.values()):
            raise ValueError(f"Negative reservoir state: {values}")
        tracer_sum = self.fast_water_m3 + self.slow_water_m3 + self.direct_water_m3
        if abs(tracer_sum - self.storage_m3) > _allowed_error(self.storage_m3, tolerance):
            raise ValueError(
                f"Tracer stock {tracer_sum} does not equal storage {self.storage_m3}"
            )
        if self.storage_m3 <= tolerance and self.age_moment_m3_day > tolerance:
            raise ValueError("An empty reservoir cannot retain a nonzero age moment")


@dataclass(frozen=True)
class ReservoirRule:
    capacity_m3: float
    dead_storage_m3: float
    target_storage_m3: float
    baseline_release_m3_step: float
    climatological_inflow_m3_step: float
    inflow_response: float
    storage_recovery_per_step: float
    minimum_release_m3_step: float
    maximum_controlled_release_m3_step: float
    withdrawal_floor_m3: float | None = None
    enabled: bool = True

    def validate(self) -> None:
        numeric = {
            name: value
            for name, value in asdict(self).items()
            if name != "enabled" and value is not None
        }
        _assert_finite(numeric, "reservoir rule")
        if self.capacity_m3 < 0:
            raise ValueError("capacity_m3 must be nonnegative")
        if not 0 <= self.dead_storage_m3 <= self.capacity_m3:
            raise ValueError("dead storage must lie within [0, capacity]")
        if not self.dead_storage_m3 <= self.target_storage_m3 <= self.capacity_m3:
            raise ValueError("target storage must lie within [dead storage, capacity]")
        if self.baseline_release_m3_step < 0:
            raise ValueError("baseline release must be nonnegative")
        if self.climatological_inflow_m3_step < 0:
            raise ValueError("climatological inflow must be nonnegative")
        if not 0 <= self.inflow_response <= 1:
            raise ValueError("inflow_response must lie within [0, 1]")
        if not 0 <= self.storage_recovery_per_step <= 1:
            raise ValueError("storage recovery must lie within [0, 1]")
        if self.minimum_release_m3_step < 0:
            raise ValueError("minimum release must be nonnegative")
        if self.maximum_controlled_release_m3_step < self.minimum_release_m3_step:
            raise ValueError("maximum release must be at least the minimum release")
        floor = self.dead_storage_m3 if self.withdrawal_floor_m3 is None else self.withdrawal_floor_m3
        if not 0 <= floor <= self.capacity_m3:
            raise ValueError("withdrawal floor must lie within [0, capacity]")


@dataclass(frozen=True)
class ReservoirFluxes:
    inflow_fast_m3: float
    inflow_slow_m3: float
    inflow_direct_m3: float = 0.0
    precipitation_m3: float = 0.0
    evaporation_demand_m3: float = 0.0
    withdrawal_demand_m3: float = 0.0
    inflow_age_moment_m3_day: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        _assert_finite(values, "reservoir flux")
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Flux inputs and demands must be nonnegative: {values}")
        inflow_total = (
            self.inflow_fast_m3 + self.inflow_slow_m3 + self.inflow_direct_m3
        )
        if inflow_total <= ABS_TOL_M3 and self.inflow_age_moment_m3_day > ABS_TOL_M3:
            raise ValueError("Zero reservoir inflow cannot carry a nonzero age moment")


@dataclass(frozen=True)
class ReservoirStepResult:
    state_next: ReservoirState
    controlled_release_m3: float
    spill_m3: float
    total_release_m3: float
    release_origin_fast_m3: float
    release_origin_slow_m3: float
    release_origin_direct_m3: float
    controlled_release_origin_fast_m3: float
    controlled_release_origin_slow_m3: float
    controlled_release_origin_direct_m3: float
    spill_origin_fast_m3: float
    spill_origin_slow_m3: float
    spill_origin_direct_m3: float
    evaporation_actual_m3: float
    withdrawal_actual_m3: float
    evaporation_fast_m3: float
    evaporation_slow_m3: float
    evaporation_direct_m3: float
    withdrawal_fast_m3: float
    withdrawal_slow_m3: float
    withdrawal_direct_m3: float
    controlled_release_age_moment_m3_day: float
    spill_age_moment_m3_day: float
    total_release_age_moment_m3_day: float
    evaporation_age_moment_m3_day: float
    withdrawal_age_moment_m3_day: float
    mean_release_age_day: float | None
    mass_balance_error_m3: float
    fast_balance_error_m3: float
    slow_balance_error_m3: float
    direct_balance_error_m3: float
    age_moment_balance_error_m3_day: float
    state_tracer_error_m3: float
    release_tracer_error_m3: float
    mass_balance_relative_error: float
    fast_balance_relative_error: float
    slow_balance_relative_error: float
    direct_balance_relative_error: float
    age_moment_balance_relative_error: float
    state_tracer_relative_error: float
    release_tracer_relative_error: float


def _partition(total: float, fast: float, slow: float, direct: float) -> tuple[float, float, float]:
    stock = fast + slow + direct
    if total <= 0 or stock <= 0:
        return 0.0, 0.0, 0.0
    scale = total / stock
    return fast * scale, slow * scale, direct * scale


def step_reservoir(
    state: ReservoirState,
    rule: ReservoirRule,
    fluxes: ReservoirFluxes,
    *,
    step_days: float = 1.0,
    tolerance: float = ABS_TOL_M3,
) -> ReservoirStepResult:
    """Advance one reservoir by one time step.

    All flux arguments are volumes per step.  The age moment is advanced in
    m3 day.  Under complete mixing every removal has the same source fractions
    and mean age as the pre-removal reservoir mixture.
    """

    if step_days <= 0:
        raise ValueError("step_days must be positive")
    state.validate(tolerance)
    rule.validate()
    fluxes.validate()

    inflow_total = (
        fluxes.inflow_fast_m3 + fluxes.inflow_slow_m3 + fluxes.inflow_direct_m3
    )

    if not rule.enabled:
        if state.storage_m3 > tolerance:
            raise ValueError("Disabled parent-degeneracy operator requires an empty state")
        if (
            fluxes.precipitation_m3 > tolerance
            or fluxes.evaporation_demand_m3 > tolerance
            or fluxes.withdrawal_demand_m3 > tolerance
        ):
            raise ValueError("Disabled exact-parent operator requires reservoir P/E/W to be zero")
        release_fast = fluxes.inflow_fast_m3
        release_slow = fluxes.inflow_slow_m3
        release_direct = fluxes.inflow_direct_m3
        total_release = release_fast + release_slow + release_direct
        release_age_moment = fluxes.inflow_age_moment_m3_day
        return ReservoirStepResult(
            state_next=ReservoirState.empty(),
            controlled_release_m3=total_release,
            spill_m3=0.0,
            total_release_m3=total_release,
            release_origin_fast_m3=release_fast,
            release_origin_slow_m3=release_slow,
            release_origin_direct_m3=release_direct,
            controlled_release_origin_fast_m3=release_fast,
            controlled_release_origin_slow_m3=release_slow,
            controlled_release_origin_direct_m3=release_direct,
            spill_origin_fast_m3=0.0,
            spill_origin_slow_m3=0.0,
            spill_origin_direct_m3=0.0,
            evaporation_actual_m3=0.0,
            withdrawal_actual_m3=0.0,
            evaporation_fast_m3=0.0,
            evaporation_slow_m3=0.0,
            evaporation_direct_m3=0.0,
            withdrawal_fast_m3=0.0,
            withdrawal_slow_m3=0.0,
            withdrawal_direct_m3=0.0,
            controlled_release_age_moment_m3_day=release_age_moment,
            spill_age_moment_m3_day=0.0,
            total_release_age_moment_m3_day=release_age_moment,
            evaporation_age_moment_m3_day=0.0,
            withdrawal_age_moment_m3_day=0.0,
            mean_release_age_day=(
                release_age_moment / total_release if total_release > 0 else None
            ),
            mass_balance_error_m3=0.0,
            fast_balance_error_m3=0.0,
            slow_balance_error_m3=0.0,
            direct_balance_error_m3=0.0,
            age_moment_balance_error_m3_day=0.0,
            state_tracer_error_m3=0.0,
            release_tracer_error_m3=0.0,
            mass_balance_relative_error=0.0,
            fast_balance_relative_error=0.0,
            slow_balance_relative_error=0.0,
            direct_balance_relative_error=0.0,
            age_moment_balance_relative_error=0.0,
            state_tracer_relative_error=0.0,
            release_tracer_relative_error=0.0,
        )

    fast_pre = state.fast_water_m3 + fluxes.inflow_fast_m3
    slow_pre = state.slow_water_m3 + fluxes.inflow_slow_m3
    direct_pre = (
        state.direct_water_m3
        + fluxes.inflow_direct_m3
        + fluxes.precipitation_m3
    )
    available_pre = state.storage_m3 + inflow_total + fluxes.precipitation_m3
    tracer_pre = fast_pre + slow_pre + direct_pre
    if abs(tracer_pre - available_pre) > _allowed_error(available_pre, tolerance):
        raise RuntimeError("Pre-removal source tracers do not close")

    aged_moment = (
        state.age_moment_m3_day
        + state.storage_m3 * step_days
        + fluxes.inflow_age_moment_m3_day
    )
    evaporation = min(fluxes.evaporation_demand_m3, available_pre)
    after_evaporation = available_pre - evaporation
    withdrawal_floor = (
        rule.dead_storage_m3
        if rule.withdrawal_floor_m3 is None
        else rule.withdrawal_floor_m3
    )
    withdrawal_available = max(after_evaporation - withdrawal_floor, 0.0)
    withdrawal = min(fluxes.withdrawal_demand_m3, withdrawal_available)
    nonrelease_removal = evaporation + withdrawal
    after_nonrelease = available_pre - nonrelease_removal

    evap_fast, evap_slow, evap_direct = _partition(
        evaporation, fast_pre, slow_pre, direct_pre
    )
    remaining_fast = fast_pre - evap_fast
    remaining_slow = slow_pre - evap_slow
    remaining_direct = direct_pre - evap_direct
    withdraw_fast, withdraw_slow, withdraw_direct = _partition(
        withdrawal, remaining_fast, remaining_slow, remaining_direct
    )
    remaining_fast -= withdraw_fast
    remaining_slow -= withdraw_slow
    remaining_direct -= withdraw_direct

    mean_mixture_age = aged_moment / available_pre if available_pre > 0 else 0.0
    evaporation_age_moment = mean_mixture_age * evaporation
    withdrawal_age_moment = mean_mixture_age * withdrawal
    age_after_nonrelease = (
        aged_moment - evaporation_age_moment - withdrawal_age_moment
    )

    proposed = (
        rule.baseline_release_m3_step
        + rule.inflow_response
        * (inflow_total - rule.climatological_inflow_m3_step)
        + rule.storage_recovery_per_step
        * (after_nonrelease - rule.target_storage_m3)
    )
    proposed = max(rule.minimum_release_m3_step, proposed)
    proposed = min(rule.maximum_controlled_release_m3_step, proposed)
    operational_available = max(after_nonrelease - rule.dead_storage_m3, 0.0)
    controlled = min(proposed, operational_available)
    spill = max(after_nonrelease - controlled - rule.capacity_m3, 0.0)
    total_release = controlled + spill
    storage_next = after_nonrelease - total_release

    release_fast, release_slow, release_direct = _partition(
        total_release, remaining_fast, remaining_slow, remaining_direct
    )
    if total_release > 0:
        controlled_fraction = controlled / total_release
    else:
        controlled_fraction = 0.0
    controlled_fast = release_fast * controlled_fraction
    controlled_slow = release_slow * controlled_fraction
    controlled_direct = release_direct * controlled_fraction
    spill_fast = release_fast - controlled_fast
    spill_slow = release_slow - controlled_slow
    spill_direct = release_direct - controlled_direct
    fast_next = remaining_fast - release_fast
    slow_next = remaining_slow - release_slow
    direct_next = remaining_direct - release_direct
    controlled_age_moment = mean_mixture_age * controlled
    spill_age_moment = mean_mixture_age * spill
    total_release_age_moment = controlled_age_moment + spill_age_moment
    age_next = age_after_nonrelease - total_release_age_moment
    mean_release_age = mean_mixture_age if total_release > 0 else None

    state_next = ReservoirState(
        storage_m3=max(storage_next, 0.0),
        fast_water_m3=max(fast_next, 0.0),
        slow_water_m3=max(slow_next, 0.0),
        direct_water_m3=max(direct_next, 0.0),
        age_moment_m3_day=max(age_next, 0.0),
    )
    state_next.validate(tolerance)
    mass_error = (
        state.storage_m3
        + inflow_total
        + fluxes.precipitation_m3
        - evaporation
        - withdrawal
        - total_release
        - state_next.storage_m3
    )
    fast_error = (
        state.fast_water_m3
        + fluxes.inflow_fast_m3
        - evap_fast
        - withdraw_fast
        - release_fast
        - state_next.fast_water_m3
    )
    slow_error = (
        state.slow_water_m3
        + fluxes.inflow_slow_m3
        - evap_slow
        - withdraw_slow
        - release_slow
        - state_next.slow_water_m3
    )
    direct_error = (
        state.direct_water_m3
        + fluxes.inflow_direct_m3
        + fluxes.precipitation_m3
        - evap_direct
        - withdraw_direct
        - release_direct
        - state_next.direct_water_m3
    )
    age_moment_error = (
        state.age_moment_m3_day
        + state.storage_m3 * step_days
        + fluxes.inflow_age_moment_m3_day
        - evaporation_age_moment
        - withdrawal_age_moment
        - total_release_age_moment
        - state_next.age_moment_m3_day
    )
    state_tracer_error = (
        state_next.fast_water_m3
        + state_next.slow_water_m3
        + state_next.direct_water_m3
        - state_next.storage_m3
    )
    release_tracer_error = (
        release_fast + release_slow + release_direct - total_release
    )
    throughput_scale = (
        state.storage_m3 + inflow_total + fluxes.precipitation_m3
    )
    fast_scale = state.fast_water_m3 + fluxes.inflow_fast_m3
    slow_scale = state.slow_water_m3 + fluxes.inflow_slow_m3
    direct_scale = (
        state.direct_water_m3
        + fluxes.inflow_direct_m3
        + fluxes.precipitation_m3
    )
    age_moment_scale = (
        state.age_moment_m3_day
        + state.storage_m3 * step_days
        + fluxes.inflow_age_moment_m3_day
    )
    mass_relative = abs(mass_error) / max(abs(throughput_scale), 1.0)
    fast_relative = abs(fast_error) / max(abs(fast_scale), 1.0)
    slow_relative = abs(slow_error) / max(abs(slow_scale), 1.0)
    direct_relative = abs(direct_error) / max(abs(direct_scale), 1.0)
    age_moment_relative = abs(age_moment_error) / max(abs(age_moment_scale), 1.0)
    state_relative = abs(state_tracer_error) / max(abs(state_next.storage_m3), 1.0)
    release_relative = abs(release_tracer_error) / max(abs(total_release), 1.0)
    if abs(mass_error) > _allowed_error(throughput_scale, tolerance):
        raise RuntimeError(f"Reservoir mass balance failed: {mass_error}")
    if (
        abs(fast_error) > _allowed_error(fast_scale, tolerance)
        or abs(slow_error) > _allowed_error(slow_scale, tolerance)
        or abs(direct_error) > _allowed_error(direct_scale, tolerance)
    ):
        raise RuntimeError(
            "Reservoir source-class balance failed: "
            f"fast={fast_error}, slow={slow_error}, direct={direct_error}"
        )
    if abs(age_moment_error) > _allowed_error(age_moment_scale, tolerance):
        raise RuntimeError(f"Reservoir age-moment balance failed: {age_moment_error}")
    if (
        abs(state_tracer_error) > _allowed_error(state_next.storage_m3, tolerance)
        or abs(release_tracer_error) > _allowed_error(total_release, tolerance)
    ):
        raise RuntimeError("Reservoir source-tracer closure failed")
    if state_next.storage_m3 > rule.capacity_m3 + _allowed_error(rule.capacity_m3, tolerance):
        raise RuntimeError("Reservoir storage exceeds capacity after spill")

    return ReservoirStepResult(
        state_next=state_next,
        controlled_release_m3=controlled,
        spill_m3=spill,
        total_release_m3=total_release,
        release_origin_fast_m3=release_fast,
        release_origin_slow_m3=release_slow,
        release_origin_direct_m3=release_direct,
        controlled_release_origin_fast_m3=controlled_fast,
        controlled_release_origin_slow_m3=controlled_slow,
        controlled_release_origin_direct_m3=controlled_direct,
        spill_origin_fast_m3=spill_fast,
        spill_origin_slow_m3=spill_slow,
        spill_origin_direct_m3=spill_direct,
        evaporation_actual_m3=evaporation,
        withdrawal_actual_m3=withdrawal,
        evaporation_fast_m3=evap_fast,
        evaporation_slow_m3=evap_slow,
        evaporation_direct_m3=evap_direct,
        withdrawal_fast_m3=withdraw_fast,
        withdrawal_slow_m3=withdraw_slow,
        withdrawal_direct_m3=withdraw_direct,
        controlled_release_age_moment_m3_day=controlled_age_moment,
        spill_age_moment_m3_day=spill_age_moment,
        total_release_age_moment_m3_day=total_release_age_moment,
        evaporation_age_moment_m3_day=evaporation_age_moment,
        withdrawal_age_moment_m3_day=withdrawal_age_moment,
        mean_release_age_day=mean_release_age,
        mass_balance_error_m3=mass_error,
        fast_balance_error_m3=fast_error,
        slow_balance_error_m3=slow_error,
        direct_balance_error_m3=direct_error,
        age_moment_balance_error_m3_day=age_moment_error,
        state_tracer_error_m3=state_tracer_error,
        release_tracer_error_m3=release_tracer_error,
        mass_balance_relative_error=mass_relative,
        fast_balance_relative_error=fast_relative,
        slow_balance_relative_error=slow_relative,
        direct_balance_relative_error=direct_relative,
        age_moment_balance_relative_error=age_moment_relative,
        state_tracer_relative_error=state_relative,
        release_tracer_relative_error=release_relative,
    )


def _topological_entity_order(graph: dict[str, set[str]]) -> list[str]:
    """Return deterministic upstream-to-downstream order or reject a cycle."""

    indegree = {entity: 0 for entity in graph}
    for downstream_entities in graph.values():
        for downstream_entity in downstream_entities:
            indegree[downstream_entity] += 1
    ready = sorted(entity for entity, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        entity = ready.pop(0)
        ordered.append(entity)
        for downstream_entity in sorted(graph[entity]):
            indegree[downstream_entity] -= 1
            if indegree[downstream_entity] == 0:
                ready.append(downstream_entity)
                ready.sort()
    if len(ordered) != len(graph):
        cyclic = sorted(entity for entity, degree in indegree.items() if degree > 0)
        raise ValueError(f"Reservoir entity graph contains a cycle: {cyclic}")
    return ordered


def validate_operator_registry(
    records: list[dict[str, object]],
    downstream_map: dict[int, int | None] | None = None,
) -> list[str]:
    """Validate entity uniqueness and return same-day routing order.

    Each record needs ``reservoir_entity_id``, ``operator_inflow_reaches`` and
    ``unique_outflow_reach``.  The real ``downstream_map`` is mandatory: every
    controlled Reach must drain *directly* to its registered outlet, because
    the pair identifies the unique edge replaced by the reservoir operator.
    Reservoir cascades separated by ordinary Reaches are discovered only after
    that outlet, by walking the locked network.
    """

    entity_ids: set[str] = set()
    controlled_reaches: dict[int, str] = {}
    normalized: list[tuple[str, set[int], int]] = []
    for record in records:
        entity = str(record["reservoir_entity_id"]).strip()
        if not entity:
            raise ValueError("reservoir_entity_id cannot be empty")
        inflows = [int(value) for value in record["operator_inflow_reaches"]]
        outflow = int(record["unique_outflow_reach"])
        if entity in entity_ids:
            raise ValueError(f"Duplicate reservoir entity: {entity}")
        entity_ids.add(entity)
        if not inflows or len(inflows) != len(set(inflows)):
            raise ValueError(f"Invalid or duplicate inflow reaches for {entity}")
        if outflow in inflows:
            raise ValueError(f"Reservoir {entity} routes to one of its own controlled reaches")
        for reach in inflows:
            if reach in controlled_reaches:
                raise ValueError(
                    f"Reach {reach} is controlled by both {controlled_reaches[reach]} and {entity}"
                )
            controlled_reaches[reach] = entity
        normalized.append((entity, set(inflows), outflow))

    if downstream_map is None:
        raise ValueError("A real downstream_map is required for registry validation")

    owner = {reach: entity for reach, entity in controlled_reaches.items()}
    graph: dict[str, set[str]] = {entity: set() for entity in entity_ids}
    network: dict[int, int | None] = {}
    for raw_reach, raw_downstream in downstream_map.items():
        reach = int(raw_reach)
        if reach in network:
            raise ValueError(f"Duplicate Reach after integer normalization: {reach}")
        downstream = None if raw_downstream is None else int(raw_downstream)
        if downstream == reach:
            raise ValueError(f"Reach network contains a self-cycle at {reach}")
        network[reach] = downstream
    required_reaches = controlled_reaches.keys() | {
        outflow for _, _, outflow in normalized
    }
    missing = sorted(reach for reach in required_reaches if reach not in network)
    if missing:
        raise ValueError(f"Reservoir registry Reaches absent from downstream map: {missing}")
    unknown_targets = sorted(
        downstream
        for downstream in network.values()
        if downstream is not None and downstream not in network
    )
    if unknown_targets:
        raise ValueError(f"Downstream map references unknown Reaches: {unknown_targets}")

    # Validate the complete supplied Reach graph, not only paths touched by a
    # reservoir.  A cycle elsewhere would still invalidate topological routing.
    for start in network:
        seen: set[int] = set()
        current: int | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"Reach network contains a cycle at {current}")
            seen.add(current)
            current = network[current]

    # A registered operator replaces exactly one locked control edge per
    # inflow Reach.  Allowing an arbitrary downstream path here would skip the
    # local inflow of intervening ordinary Reaches and break parent recovery.
    for entity, inflows, outflow in normalized:
        for inflow in inflows:
            actual_downstream = network[inflow]
            if actual_downstream != outflow:
                raise ValueError(
                    f"Reservoir {entity} control edge {inflow}->{outflow} disagrees "
                    f"with locked topology ({actual_downstream})"
                )

    # Find the first downstream reservoir even when ordinary Reaches lie
    # between two operators.  The resulting order is the mandatory within-day
    # execution order for propagating release volumes and age moments.
    for entity, _, outflow in normalized:
        current: int | None = outflow
        while current is not None:
            downstream_entity = owner.get(current)
            if downstream_entity is not None:
                if downstream_entity == entity:
                    raise ValueError(
                        f"Reservoir {entity} drains back into one of its own controlled Reaches"
                    )
                graph[entity].add(downstream_entity)
                break
            current = network[current]
    return _topological_entity_order(graph)


def _flood_drawdown_window(
    day: int,
    *,
    start_day: int,
    end_day: int,
    transition_days: float,
    days_in_year: int,
) -> float:
    """Periodic single-window indicator with raised-cosine shoulders."""

    relative_day = (day - start_day) % days_in_year
    flood_span = (end_day - start_day) % days_in_year
    if relative_day <= flood_span:
        return 1.0
    if transition_days <= 0:
        return 0.0
    if relative_day < flood_span + transition_days:
        phase = (relative_day - flood_span) / transition_days
        return 0.5 * (1.0 + cos(pi * phase))
    if relative_day > days_in_year - transition_days:
        phase = (days_in_year - relative_day) / transition_days
        return 0.5 * (1.0 + cos(pi * phase))
    return 0.0


def generate_seasonal_target_storage_curve(
    *,
    capacity_m3: float,
    dead_storage_m3: float,
    baseline_active_fraction: float,
    annual_harmonic_amplitude_fraction: float,
    annual_peak_day: int,
    flood_start_day: int,
    flood_end_day: int,
    flood_drawdown_fraction: float,
    transition_days: float = 15.0,
    days_in_year: int = 365,
) -> tuple[float, ...]:
    """Generate a low-dimensional periodic target-storage curve.

    The curve is one annual harmonic baseline minus one smooth flood-season
    drawdown window.  It has a fixed small parameter set and therefore cannot
    silently become 365 independent target-storage parameters.  Fractions are
    relative to active storage (capacity minus dead storage); combinations
    that would leave physical bounds are rejected instead of clipped.  The
    baseline fraction is the harmonic centre before flood drawdown, not the
    annual mean of the resulting curve.
    """

    numeric = {
        "capacity_m3": capacity_m3,
        "dead_storage_m3": dead_storage_m3,
        "baseline_active_fraction": baseline_active_fraction,
        "annual_harmonic_amplitude_fraction": annual_harmonic_amplitude_fraction,
        "flood_drawdown_fraction": flood_drawdown_fraction,
        "transition_days": transition_days,
    }
    _assert_finite(numeric, "seasonal target-curve parameter")
    if days_in_year not in (365, 366):
        raise ValueError("days_in_year must be 365 or 366")
    if not 0 <= dead_storage_m3 < capacity_m3:
        raise ValueError("Seasonal target curve requires 0 <= dead storage < capacity")
    if not 0 <= baseline_active_fraction <= 1:
        raise ValueError("baseline_active_fraction must lie within [0, 1]")
    if not 0 <= annual_harmonic_amplitude_fraction <= 1:
        raise ValueError("annual harmonic amplitude must lie within [0, 1]")
    if not 0 <= flood_drawdown_fraction <= 1:
        raise ValueError("flood drawdown fraction must lie within [0, 1]")
    for label, day in {
        "annual_peak_day": annual_peak_day,
        "flood_start_day": flood_start_day,
        "flood_end_day": flood_end_day,
    }.items():
        if not 1 <= day <= days_in_year:
            raise ValueError(f"{label} must lie within [1, days_in_year]")
    flood_span = (flood_end_day - flood_start_day) % days_in_year
    if flood_span == 0:
        raise ValueError("Flood window must not span zero or the entire year")
    if transition_days < 0 or flood_span + 2 * transition_days >= days_in_year:
        raise ValueError("Flood-window shoulders overlap or have invalid duration")

    active_storage = capacity_m3 - dead_storage_m3
    targets: list[float] = []
    for day in range(1, days_in_year + 1):
        harmonic = annual_harmonic_amplitude_fraction * cos(
            2.0 * pi * (day - annual_peak_day) / days_in_year
        )
        flood_window = _flood_drawdown_window(
            day,
            start_day=flood_start_day,
            end_day=flood_end_day,
            transition_days=transition_days,
            days_in_year=days_in_year,
        )
        active_fraction = (
            baseline_active_fraction
            + harmonic
            - flood_drawdown_fraction * flood_window
        )
        if not -REL_TOL <= active_fraction <= 1.0 + REL_TOL:
            raise ValueError(
                "Seasonal target parameters leave physical storage bounds at "
                f"day {day}: active_fraction={active_fraction}"
            )
        active_fraction = min(max(active_fraction, 0.0), 1.0)
        targets.append(dead_storage_m3 + active_storage * active_fraction)
    return tuple(targets)


def generate_calendar_seasonal_target_storage_curve(
    *,
    year: int,
    capacity_m3: float,
    dead_storage_m3: float,
    baseline_active_fraction: float,
    annual_harmonic_amplitude_fraction: float,
    annual_peak_month_day: tuple[int, int],
    flood_start_month_day: tuple[int, int],
    flood_end_month_day: tuple[int, int],
    flood_drawdown_fraction: float,
    transition_days: float = 15.0,
) -> tuple[float, ...]:
    """Calendar-safe wrapper preserving month/day phase in leap years.

    Literature priors such as "May--August" must use this wrapper.  It converts
    the registered month/day boundaries separately for each simulated year, so
    dates after February do not drift by one day in leap years.
    """

    if not 1 <= year <= 9998:
        raise ValueError("year must be supported by the Gregorian calendar")

    def day_of_year(month_day: tuple[int, int], label: str) -> int:
        if len(month_day) != 2:
            raise ValueError(f"{label} must be a (month, day) pair")
        month, day = month_day
        try:
            value = date(year, int(month), int(day))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label} for {year}: {month_day}") from exc
        return value.timetuple().tm_yday

    days_in_year = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    return generate_seasonal_target_storage_curve(
        capacity_m3=capacity_m3,
        dead_storage_m3=dead_storage_m3,
        baseline_active_fraction=baseline_active_fraction,
        annual_harmonic_amplitude_fraction=annual_harmonic_amplitude_fraction,
        annual_peak_day=day_of_year(annual_peak_month_day, "annual_peak_month_day"),
        flood_start_day=day_of_year(flood_start_month_day, "flood_start_month_day"),
        flood_end_day=day_of_year(flood_end_month_day, "flood_end_month_day"),
        flood_drawdown_fraction=flood_drawdown_fraction,
        transition_days=transition_days,
        days_in_year=days_in_year,
    )
