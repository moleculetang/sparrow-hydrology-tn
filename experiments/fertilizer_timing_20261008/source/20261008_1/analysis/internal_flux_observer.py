"""Read-only reconstruction of the accepted kernel's daily internal transfers.

High/low transition replay follows the exact forward representation. Releases
use the forward high stock, not a rounded high-minus-compensation surrogate.
Fluxes are diagnostics: none are written back into a physical recurrence.
"""
import numpy as np


def stage_semantics(stage):
    """Units and annual summation meaning; a daily snapshot is not a flux."""
    if stage=='available_before_uptake_kg':
        return dict(units='kg N daily pre-uptake stock',sum_value_units='kg N day',
            quantity_kind='stock_snapshot',annual_sum_is_external_mass=False,
            annual_sum_meaning='sum of daily stock snapshots; not nitrogen input or exported mass')
    if stage in ('plant_potential_demand_kg','plant_capacity_limited_demand_kg','plant_supply_shortfall_kg'):
        return dict(units='kg N daily requested opportunity',sum_value_units='kg N summed daily requests',
            quantity_kind='potential_or_unmet_daily_request',annual_sum_is_external_mass=False,
            annual_sum_meaning='sum of daily requests/opportunities; carried deficits can recur; not harvest debt or observed production')
    external={'external_plant_kg','external_active_kg','external_protected_kg','external_available_kg'}
    actual={'mineralize_active_kg','mineralize_protected_kg','plant_actual_uptake_kg','mobilized_kg',
        'fast_export_kg','lower_input_kg','slow_export_kg','nonexport_loss_kg','plant_external_output_kg',
        'plant_internal_active_return_kg','plant_internal_protected_return_kg',
        'reservoir_captures','reservoir_releases','network_channel_removed','network_terminal'}
    if stage not in external|actual:raise ValueError('UNREGISTERED_INTERNAL_TRACE_STAGE '+stage)
    return dict(units='kg N/day',sum_value_units='kg N',quantity_kind='actual_daily_transfer',
        annual_sum_is_external_mass=stage in external,
        annual_sum_meaning='sum of actual daily transfers; internal transfers must not be added as external sources')


def observe(result):
    from d29_platform.precision_transfer import transfer_expansion
    i = result._inputs
    nt, nr, nl = i['shape']
    before = result.states[:-1].copy()
    for day in i['transition_days']:
        low = np.zeros_like(before[day])
        low[..., 1:] = -i['compensation_history'][day].reshape(nr, nl, 4)
        before[day], _, _ = transfer_expansion(before[day], low, i['matrices'][i['event'][day]])
    b = before.reshape(nt, nr*nl, 5)
    source = i['sources']
    pa, pp, pm, pl, ff, lr = i['probabilities']
    out = i['outflows']
    realized = i['realized_outflows']
    ka, kp = pa*b[..., 1], pp*b[..., 2]
    available = b[..., 3] + source[..., 3] + ka + kp
    planned_total = out[..., 0] + out[..., 1] + out[..., 2]
    need = np.maximum(0., i['target'] + planned_total - b[..., 0] - source[..., 0])
    opportunity = np.minimum(need, i['plant_growth_capacity'])
    uptake = np.minimum(available, opportunity)
    remainder = available - uptake
    mobilized = pm*remainder
    lower_input = (1-ff)*mobilized
    fast = ff*mobilized
    slow = lr*(b[..., 4] + lower_input)
    loss = pl*((1-pm)*remainder)
    inferred = (result.states[1:, ..., 0].reshape(nt, nr*nl)
                + realized.sum(-1) - b[..., 0] - source[..., 0])
    errors = dict(
        uptake_balance_kg=float(np.max(abs(inferred-uptake), initial=0.)),
        fast_replay_kg=float(np.max(abs(fast-result.fluxes[..., 0].reshape(nt, nr*nl)), initial=0.)),
        slow_replay_kg=float(np.max(abs(slow-result.fluxes[..., 1].reshape(nt, nr*nl)), initial=0.)),
        loss_replay_kg=float(np.max(abs(loss-result.fluxes[..., 2].reshape(nt, nr*nl)), initial=0.)),
        plant_export_replay_kg=float(np.max(abs(realized[..., 0]-result.fluxes[..., 3].reshape(nt, nr*nl)), initial=0.)),
        negative_uptake_kg=float(np.max(np.maximum(0., -uptake), initial=0.)),
        uptake_above_supply_or_capacity_kg=float(np.max(np.maximum(0., uptake-np.minimum(available, opportunity)), initial=0.)))
    fields = dict(mineralize_active_kg=ka, mineralize_protected_kg=kp,
                  available_before_uptake_kg=available, plant_potential_demand_kg=need,
                  plant_capacity_limited_demand_kg=opportunity, plant_actual_uptake_kg=uptake,
                  plant_supply_shortfall_kg=np.maximum(0., opportunity-uptake),
                  mobilized_kg=mobilized, fast_export_kg=fast, lower_input_kg=lower_input,
                  slow_export_kg=slow, nonexport_loss_kg=loss,
                  plant_external_output_kg=realized[..., 0],
                  plant_internal_active_return_kg=realized[..., 1],
                  plant_internal_protected_return_kg=realized[..., 2],
                  external_plant_kg=source[..., 0], external_active_kg=source[..., 1],
                  external_protected_kg=source[..., 2], external_available_kg=source[..., 3])
    arrays = {}
    for name, values in fields.items():
        values = values.reshape(nt, nr, nl)
        if not np.isfinite(values).all():
            raise ValueError('NONFINITE_INTERNAL_TRACE '+name)
        arrays[name+'_daily_by_reach'] = values.sum(axis=2)
        arrays[name+'_daily_by_land'] = values.sum(axis=1)
        arrays[name+'_annual_by_reach_land'] = values.sum(axis=0)
    receipt = dict(passed=max(errors.values())<=1e-7, errors=errors,
                   daily_joint_reach_land_tensor_saved=False,
                   flux3_is_external_plant_output_not_uptake=True,
                   internal_transfers_are_not_external_sources=True,
                   demand_is_potential_not_added_nitrogen=True,
                   mineralization_uses_previous_high_stock=True,
                   transition_uses_forward_high_low=True,
                   physical_recurrence_modified=False)
    return arrays, receipt


def routing_checks(local, routing):
    """Topology-expanded reach series are not added as a basin load.

Only incremental local input, channel removal, terminal export and reservoir
stock change enter the basin budget. Reservoir captures/releases remain one
column per actual shared reservoir, irrespective of its control count.
"""
    local = np.asarray(local, dtype=np.float64)
    if local.ndim != 2 or not np.isfinite(local).all():
        raise ValueError('FINITE_DAILY_INCREMENTAL_LOCAL_REQUIRED')
    nt = len(local)
    for name in ('inlet', 'preout', 'official', 'channel_removed'):
        if routing[name].shape != local.shape or not np.isfinite(routing[name]).all():
            raise ValueError('ROUTING_REACH_DATE_SUPPORT '+name)
    captures, releases, stocks = [np.asarray(routing[k]) for k in ('captures', 'releases', 'stocks')]
    if captures.shape != releases.shape or captures.shape != stocks.shape or captures.shape[0] != nt:
        raise ValueError('SINGLE_RESERVOIR_DATE_SUPPORT_REQUIRED')
    if not np.isfinite(np.concatenate((captures, releases, stocks), axis=1)).all():
        raise ValueError('FINITE_RESERVOIR_TRACE_REQUIRED')
    old = np.vstack((np.zeros((1, stocks.shape[1])), stocks[:-1]))
    # Use wider accumulation only for independent diagnostics, never to change
    # recurrence values or disguise negative states/outputs.
    ld = np.longdouble
    reservoir_residual = old.astype(ld)+captures.astype(ld)-releases.astype(ld)-stocks.astype(ld)
    daily = (local.astype(ld).sum(1)-routing['channel_removed'].astype(ld).sum(1)
             -np.asarray(routing['terminal'], dtype=ld)
             -(stocks.astype(ld)-old.astype(ld)).sum(1))
    scale = np.abs(local.astype(ld)).sum(1)
    if np.any((scale==0)&(daily!=0)):
        # Initial reservoir release on a zero-local-input day is legitimate;
        # a throughput scale includes it while basin residual remains explicit.
        scale = np.maximum(scale, np.abs(releases.astype(ld)).sum(1)+np.abs(old.astype(ld)).sum(1))
    relative = np.divide(abs(daily), scale, out=np.zeros(nt, dtype=ld), where=scale!=0)
    undefined_nonzero = bool(np.any((scale==0)&(daily!=0)))
    whole = float(local.astype(ld).sum()-routing['channel_removed'].astype(ld).sum()
                  -np.asarray(routing['terminal'], dtype=ld).sum()-stocks[-1].astype(ld).sum())
    whole_scale = float(np.abs(local.astype(ld)).sum())
    whole_relative = abs(whole)/whole_scale if whole_scale else 0. if whole==0 else None
    reservoir_scale = np.abs(old.astype(ld))+np.abs(captures.astype(ld))
    reservoir_relative = np.divide(abs(reservoir_residual), reservoir_scale,
                                  out=np.zeros_like(reservoir_residual), where=reservoir_scale!=0)
    rv_undefined = bool(np.any((reservoir_scale==0)&(reservoir_residual!=0)))
    minimum = min(float(np.min(v, initial=0.)) for v in (captures, releases, stocks))
    receipt = dict(passed=not undefined_nonzero and not rv_undefined and whole_relative is not None
                   and max(float(relative.max(initial=0.)), float(reservoir_relative.max(initial=0.)), whole_relative)<=1e-10
                   and minimum>=-1e-7,
                   daily_basin_relative_max=float(relative.max(initial=0.)),
                   reservoir_relative_max=float(reservoir_relative.max(initial=0.)),
                   basin_total_residual_kg=whole, basin_total_relative=whole_relative,
                   zero_scale_nonzero_residual=undefined_nonzero or rv_undefined,
                   minimum_reservoir_mass_kg=minimum, reservoir_count=stocks.shape[1],
                   expanded_reach_or_nested_station_fluxes_summed_as_basin_input=False,
                   physical_recurrence_modified=False)
    return dict(basin_residual_kg_day=np.asarray(daily, dtype=float),
                reservoir_residual_kg_day=np.asarray(reservoir_residual, dtype=float)), receipt
