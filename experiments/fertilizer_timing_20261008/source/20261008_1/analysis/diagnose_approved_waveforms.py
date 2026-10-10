"""Read-only post-freeze waveforms under each path's approved scientific gate.

Reuses the immutable shape calculations, but not their original-only admission
entry. No physical rerun or observation read. All-case scheduling requires the
actual 44-path delivery; a separately labelled single-baseline preflight is not
that delivery. Plant-debt decisions use the dedicated accurate annual reader.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis'), str(ROOT / 'scripts')]
from d29_platform.runtime import sha, write_json


def approved_gate(job):
    science = ROOT.with_name('20261008_1_maize_scope_v2') if job['calendar'] == 'K' else ROOT
    gate = science / 'evidence/formal_gate.json'
    proof = json.loads(gate.read_text(encoding='utf8'))
    if proof.get('passed') is not True or proof.get('sealed') is not True:
        raise RuntimeError('ACTUAL_APPROVED_SCIENTIFIC_GATE_REQUIRED')
    return sha(gate)


def accepted_point(folder, job, support):
    from assemble_approved_delivery import validate_frozen
    files = validate_frozen(folder, job, approved_gate(job), support)
    physical = json.loads((folder / 'physical_ledger.json').read_text(encoding='utf8'))
    freeze = json.loads((folder / 'prediction_freeze.json').read_text(encoding='utf8'))
    return physical, freeze, files


def extract(folder, job, destination, support):
    """One isolated pure-data extraction; immutable sources are never rewritten."""
    import numpy as np
    import pandas as pd
    from physical_waveform_diagnostics import annual_shapes, shape
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    if not destination.is_relative_to((ROOT / 'outputs').resolve()) or destination.is_relative_to(folder):
        raise RuntimeError('SEPARATE_ANALYSIS_OUTPUT_NAMESPACE_REQUIRED')
    physical, freeze, files = accepted_point(folder, job, support)
    if (destination / 'complete.json').exists():
        raise RuntimeError('EXISTING_DIAGNOSTIC_REVIEW_NO_RERUN')
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('PARTIAL_DIAGNOSTIC_REQUIRES_REVIEW')
    dates = pd.date_range('1961-01-01', '2024-12-31')
    labels = physical['labels']; flows = []; station_rows = []
    with np.load(folder / 'full_history_physical_source_ledger.npz', allow_pickle=False) as z:
        for key in ('local_kg_day', 'local_fast_kg_day', 'local_slow_kg_day'):
            values = z[key]
            if values.shape != (len(dates), 230) or not np.isfinite(values).all():
                raise RuntimeError('FINITE_COMPLETE230REACH_HISTORY_REQUIRED')
            flows.extend(annual_shapes(values.sum(axis=1), dates, key, 'kg N/day'))
        external = z['external_by_source_kg_day']; tags = z['local_source_kg_day']
        if (external.shape != (len(dates), len(labels)) or
                tags.shape != (len(dates), 230, len(labels)) or
                not np.isfinite(external).all() or not np.isfinite(tags).all()):
            raise RuntimeError('FINITE_FULL_SOURCE_HISTORY_REQUIRED')
        for index, label in enumerate(labels):
            flows.extend(annual_shapes(external[:, index], dates, 'external_input', 'kg N/day', source=label))
            flows.extend(annual_shapes(tags[:, :, index].sum(axis=1), dates,
                                       'incremental_local_export', 'kg N/day', source=label))
    with np.load(folder / 'station_source_ledger.npz', allow_pickle=False) as z:
        if not np.array_equal(z['labels'], np.asarray(labels, dtype=str)):
            raise RuntimeError('EXACT_STATION_SOURCE_LABEL_ORDER_REQUIRED')
        ti, keys = z['day_index'], z['station_key']
        mass, water, tagmass = z['total_mass_kg'], z['water_m3'], z['mass_by_source_kg']
        if (ti.ndim != 1 or not np.issubdtype(ti.dtype, np.integer) or (ti < 0).any() or
                (ti >= len(dates)).any() or len(keys) != len(ti) or mass.shape != (len(ti),) or
                water.shape != mass.shape or tagmass.shape != (len(ti), len(labels))):
            raise RuntimeError('EXACT_SAME_DAY_MASS_WATER_SUPPORT_REQUIRED')
        if not all(np.isfinite(a).all() for a in (mass, water, tagmass)) or (water < 0).any():
            raise RuntimeError('FINITE_NONNEGATIVE_WATER_BOUNDARY_REQUIRED')
        if set(keys) != set(support['station_keys']):
            raise RuntimeError('COMPLETE_CERTIFIED_STATION_SUPPORT_REQUIRED')
        table = pd.DataFrame(dict(station_key=keys, date=dates[ti]))
        if table.duplicated(['station_key', 'date']).any():
            raise RuntimeError('DUPLICATE_STATION_DAY_BOUNDARY')
        for station, positions in table.groupby('station_key', sort=True).indices.items():
            positions = np.asarray(positions); positions = positions[np.argsort(ti[positions])]
            d = dates[ti[positions]]
            station_rows.extend(annual_shapes(mass[positions], d, 'OU_boundary_mass', 'kg N/day', station))
            station_rows.extend(annual_shapes(water[positions], d, 'OU_boundary_water', 'm3/day', station))
            for index, label in enumerate(labels):
                station_rows.extend(annual_shapes(tagmass[positions, index], d,
                                                 'OU_boundary_source_mass', 'kg N/day', station, label))
            for year in range(2016, 2025):
                take = positions[dates[ti[positions]].year == year]
                valid = take[water[take] > 0]
                if len(valid):
                    descriptor = shape(1000 * mass[valid] / water[valid], dates[ti[valid]])
                    descriptor.update(nse_status='not_computed_prediction_shape_only_no_outcomes_read',
                                      mass_timing_status='concentration_calendar_shape_not_mass_transit')
                    station_rows.append(dict(stage='OU_concentration_prediction_shape', units='mg N/L',
                        year=year, station_key=station, source=None,
                        sum_value_units='mg N day/L_on_positive_water_days_not_basin_mass',
                        zero_water_days=int(np.sum(water[take] == 0)), **descriptor))
                else:
                    station_rows.append(dict(stage='OU_concentration_prediction_shape', units='mg N/L',
                        year=year, station_key=station, source=None, days=0,
                        zero_water_days=len(take), nse_status='not_applicable_no_water_or_no_dates',
                        mass_timing_status='undefined_concentration_zero_water_no_epsilon'))
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, rows in [('domain_flux_shapes.csv', flows), ('station_boundary_shapes.csv', station_rows)]:
        path = destination / name; pd.DataFrame(rows).to_csv(path, index=False); outputs[name] = sha(path)
    result = dict(passed=True, registered_job=job, formal_gate_sha256=approved_gate(job),
        parameter_sha256=freeze['parameter_sha256'], source_folder=str(folder),
        source_freeze_sha256=files['prediction_freeze.json'], physical_receipt_sha256=files['physical_ledger.json'],
        scope='approved_frozen_physical_shapes_not_TN_accuracy', station_count=len(set(keys)),
        history=[1961, 2024], reported_years=[2016, 2024], source_multiplier=1.,
        output_sha256=outputs, source_sha256=sha(Path(__file__)),
        immutable_shape_helper_sha256=sha(ROOT / 'analysis/physical_waveform_diagnostics.py'),
        physical_oracle_calls=0, evaluation_labels_read=False, negative_values_clipped=False,
        basin_total_is_incremental_reach_export_not_nested_station_sum=True,
        concentration_at_zero_water_is_undefined=True,
        timing_is_calendar_shape_not_identified_transport_time=True,
        plant_upgrade_guard_source='analysis/diagnose_approved_plant_debt.py')
    write_json(destination / 'complete.json', result)
    return result


def main(jid, mode):
    from host_resources import registry, registry_root
    lease = registry.status(root=registry_root())['leases'].get(os.environ.get('D29_LEASE_TOKEN'))
    if (not lease or lease['project'] != '20261008_1' or lease['pid'] != os.getpid() or
            lease['created'] != registry.identity(os.getpid()) or lease['threads'] != 1 or
            lease['job'] != 'analysis/approved_waveforms_' + mode + '/' + jid):
        raise RuntimeError('EXACT_SINGLE_THREAD_DIAGNOSTIC_LEASE_REQUIRED')
    jobs = json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf8'))
    job = next(j for j in jobs if j['id'] == jid)
    if mode == 'preflight':
        if jid != 'TIM_F24_C0_V0_s0':
            raise RuntimeError('PREFLIGHT_ONLY_EXISTING_FULLY_SUFFICIENT_BASELINE')
        folder = ROOT / 'outputs/jobs' / jid
    elif mode == 'final':
        from approved_epoch_delivery import admitted_folders
        folder = admitted_folders(ROOT)[jid]
    else:
        raise RuntimeError('DECLARED_DIAGNOSTIC_MODE_REQUIRED')
    support = json.loads((ROOT / 'evidence/expected_station_support.json').read_text(encoding='utf8'))
    result = extract(folder, job, ROOT / 'outputs' / ('approved_waveforms_' + mode) / jid, support)
    print(json.dumps(dict(passed=True, job=jid, mode=mode, station_count=result['station_count'])))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
