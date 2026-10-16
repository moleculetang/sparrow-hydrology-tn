"""Full-history read-only replay using each frozen job's own scientific root.

The approved maize K root must supply its adapter, chain and routing modules.
Imported results may live in the delivery archive; diagnostic writes use a new
nonregistered job namespace in the matching science root. No evaluation read.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def science_root(job):
    if job['calendar'] not in ('C0', 'C1', 'C2', 'K'):
        raise RuntimeError('REGISTERED_CALENDAR_REQUIRED')
    return ROOT.with_name('20261008_1_maize_scope_v2') if job['calendar'] == 'K' else ROOT


def install_paths(science):
    paths = [science, science/'scripts', science/'analysis', science/'work',
             science/'vendor/legacy22/scripts',
             science/'vendor/legacy22/vendor/expert/tn_challenge',
             ROOT/'analysis', ROOT/'deployment']
    sys.path[:0] = [str(p) for p in paths]


def require_scientific_module(module, science):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(science.resolve()):
        raise RuntimeError('WRONG_SCIENTIFIC_RUNTIME '+module.__name__+' '+str(path))
    return str(path.relative_to(science))


def export_comparison(actual, expected):
    import numpy as np
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or
            not np.isfinite(expected).all()):
        raise RuntimeError('FINITE_SAME_SUPPORT_EXPORT_REQUIRED')
    error = abs(actual-expected)
    return dict(bitwise=np.array_equal(actual, expected),
                max_abs_difference_kg=float(np.max(error, initial=0.)),
                passed=bool(np.all(error <= 1e-6)))


def prediction_comparison(actual, expected, value, reference_value):
    import numpy as np
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or
            not np.isfinite(expected).all() or not np.isfinite([value, reference_value]).all()):
        raise RuntimeError('FINITE_SAME_SUPPORT_PREDICTION_REQUIRED')
    error = abs(actual-expected)
    objective_error = abs(float(value)-float(reference_value))
    return dict(bitwise_training_prediction=np.array_equal(actual, expected),
        training_prediction_max_error_mg_l=float(np.max(error, initial=0.)),
        prediction_passed=bool(np.all(error <= 1e-8*(1+abs(expected)))),
        objective_replay_error=objective_error,
        objective_passed=objective_error <= 1e-8*(1+abs(reference_value)))


def main(jid, mode):
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    job = next(j for j in jobs if j['id'] == jid)
    science = science_root(job)
    # Fresh process required: never switch a loaded physical runtime in place.
    if any(n in sys.modules for n in ('timing.adapter', 'timing.chain', 'train_job', 'freeze_job', 'routing')):
        raise RuntimeError('FRESH_PROCESS_FOR_EACH_SCIENTIFIC_ROOT_REQUIRED')
    install_paths(science)
    from d29_platform.runtime import configure, sha, write_json
    configure()
    from host_resources import registry, registry_root
    lease = registry.status(root=registry_root())['leases'].get(os.environ.get('D29_LEASE_TOKEN'))
    namespace = 'internal_trace_approved_'+mode+'_v1'
    if (not lease or lease['project'] != '20261008_1' or lease['pid'] != os.getpid() or
            lease['created'] != registry.identity(os.getpid()) or lease['threads'] != 1 or
            lease['job'] != 'analysis/'+namespace+'/'+jid):
        raise RuntimeError('EXACT_SINGLE_THREAD_INTERNAL_TRACE_LEASE_REQUIRED')
    if mode == 'preflight':
        if jid != 'TIM_F24_C0_V0_s0':
            raise RuntimeError('PREFLIGHT_ONLY_FULLY_ACCEPTED_BASELINE')
        folder = ROOT/'outputs/jobs'/jid
    elif mode == 'final':
        from approved_epoch_delivery import admitted_folders
        folder = admitted_folders(ROOT)[jid]
    else:
        raise RuntimeError('DECLARED_TRACE_MODE_REQUIRED')
    from diagnose_approved_waveforms import accepted_point, approved_gate
    support = json.loads((ROOT/'evidence/expected_station_support.json').read_text(encoding='utf8'))
    physical, freeze, bound = accepted_point(folder, job, support)
    # The original-only admission functions may alter search paths. Restore
    # matching-root precedence before importing any physical/scientific module.
    install_paths(science)
    import train_job
    import freeze_job
    import timing.chain as chain_module
    import routing as routing_module
    import timing.adapter as adapter_module
    import d29_platform.runtime as runtime_module
    modules = [train_job, freeze_job, chain_module, routing_module, adapter_module, runtime_module]
    runtime_files = {m.__name__: require_scientific_module(m, science) for m in modules}
    _, gate = train_job.validate_admission()
    if gate != approved_gate(job) or gate != freeze['formal_gate_sha256']:
        raise RuntimeError('MATCHING_APPROVED_RUNTIME_GATE_REQUIRED')
    if next(j for j in json.loads((science/'config/jobs.json').read_text()) if j['id']==jid) != job:
        raise RuntimeError('MATCHING_SCIENTIFIC_JOB_REQUIRED')
    import numpy as np
    import pandas as pd
    from internal_flux_observer import observe, routing_checks, stage_semantics
    from physical_waveform_diagnostics import annual_shapes
    target = science/'outputs/jobs'/(namespace+'_'+jid)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('EXISTING_TRACE_REVIEW_NO_RERUN')
    point_path = folder/'frozen_parameters.npy'
    point = np.load(point_path, allow_pickle=False)
    # Original adapter changes no frozen result; its diagnostic ID is distinct.
    derived = dict(job, id=namespace+'_'+jid)
    a = freeze_job.training_only_adapter(derived)
    if a.folder.resolve() != target.resolve():
        raise RuntimeError('CORRECT_SCIENCE_DIAGNOSTIC_NAMESPACE_REQUIRED')
    contract=json.loads((folder/'training_contract.json').read_text())
    from native_bound_adapter_v2 import canonical_bounds
    if (a.names != contract['names'] or a.scale.tolist()!=contract['scale'] or
            [list(pair) for pair in canonical_bounds(a.bounds)]!=contract['bounds']):
        raise RuntimeError('MATCHING_FROZEN_COORDINATE_NAMES_REQUIRED')
    dates = pd.date_range('1961-01-01','2024-12-31')
    if not a.data.dates.equals(dates):
        raise RuntimeError('FULL64YEAR_DATE_SUPPORT_REQUIRED')
    years = []; shapes = []; native = chain_module.TimingChain._run
    start = time.monotonic()
    with np.load(folder/'full_history_physical_source_ledger.npz',allow_pickle=False) as z:
        local=z['local_kg_day']; fast=z['local_fast_kg_day']; slow=z['local_slow_kg_day']
    for array in (local,fast,slow):
        if array.shape != (len(dates),230) or not np.isfinite(array).all():
            raise RuntimeError('FINITE230REACH64YEAR_REFERENCE_REQUIRED')
    def wrapped(chain, year, initial, eta, correction):
        result,mask = native(chain,year,initial,eta,correction)
        ix = np.flatnonzero(a.data.dates.year==year)
        actual = dict(local=result.fluxes[...,:2].sum(axis=(2,3)),
                      fast=result.fluxes[...,0].sum(axis=2),slow=result.fluxes[...,1].sum(axis=2))
        expected = dict(local=local[ix],fast=fast[ix],slow=slow[ix])
        comparisons = {k:export_comparison(actual[k],expected[k]) for k in actual}
        arrays,receipt = observe(result)
        if not receipt['passed'] or not all(c['passed'] for c in comparisons.values()):
            write_json(target/'failed_year.json',dict(year=year,receipt=receipt,comparisons=comparisons))
            raise RuntimeError('INTERNAL_TRACE_REFERENCE_MISMATCH '+str(year))
        output = {k:v for k,v in arrays.items() if year>=2016 or not k.endswith('_daily_by_reach')}
        output['date'] = a.data.dates[ix].to_numpy(dtype='datetime64[D]')
        p=target/'yearly'/f'{year}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,**output)
        for key,values in arrays.items():
            if key.endswith('_daily_by_land'):
                shapes.extend(annual_shapes(values.sum(axis=1),a.data.dates[ix],
                    key.removesuffix('_daily_by_land'),'kg N/day'))
        years.append(dict(year=year,passed=True,internal_receipt=receipt,
            source_ledger_export_comparison=comparisons,file=str(p.relative_to(target)),sha256=sha(p)))
        write_json(target/'progress.json',dict(completed_years=len(years),last_year=year,
            registered_job=job,no_evaluation_outcomes_read=True,formal_gate_sha256=gate))
        return result,mask
    chain_module.TimingChain._run=wrapped
    try: value,_=a.value_gradient(point,forward_only=True)
    finally: chain_module.TimingChain._run=native
    if [r['year'] for r in years] != list(range(1961,2025)):
        raise RuntimeError('EXACT64YEARS_OBSERVED_REQUIRED')
    base,_=a.unpack(point)
    if base[a.eta_index] != 0.:
        raise RuntimeError('TRACE_SOURCE_MULTIPLIER_NOT_ONE')
    routing=routing_module.route(a.model.daily_data,local,vf=float(base[a.base_names.index('v_f')]))
    residual,network=routing_checks(local,routing)
    if not network['passed']:
        write_json(target/'failed_network.json',network)
        raise RuntimeError('INTERNAL_TRACE_NETWORK_CLOSURE_FAILED')
    keep=a.data.dates.year>=2016;rp=target/'routing_trace.npz'
    np.savez_compressed(rp,**{k:routing[k][keep] for k in ('inlet','preout','official','channel_removed')},
        **{k:routing[k] for k in ('captures','releases','stocks','terminal')},**residual,
        date=a.data.dates.to_numpy(dtype='datetime64[D]'),reach_date=a.data.dates[keep].to_numpy(dtype='datetime64[D]'),
        global_reach_ids=a.model.daily_data.global_reach_ids)
    for name in ('captures','releases'):
        for k in range(routing[name].shape[1]):
            shapes.extend(annual_shapes(routing[name][:,k],a.data.dates,'reservoir_'+name,'kg N/day',source=str(k)))
    for name in ('channel_removed','terminal'):
        values=routing[name].sum(1) if routing[name].ndim==2 else routing[name]
        shapes.extend(annual_shapes(values,a.data.dates,'network_'+name,'kg N/day'))
    for row in shapes: row.update(stage_semantics(row['stage']))
    sp=target/'internal_and_network_shapes.csv';pd.DataFrame(shapes).to_csv(sp,index=False)
    reference=np.load(folder/'audited_training_prediction.npy',allow_pickle=False)
    comparison=prediction_comparison(a.prediction,reference,value,physical['objective'])
    if not comparison['prediction_passed'] or not comparison['objective_passed']:
        write_json(target/'failed_prediction.json',comparison)
        raise RuntimeError('ACCEPTED_TRAINING_PREDICTION_OR_OBJECTIVE_CHANGED')
    # Revalidate every bound artifact, not only the parameter file, after replay.
    _,_,after=accepted_point(folder,job,support)
    if after != bound:
        raise RuntimeError('IMMUTABLE_FROZEN_ARTIFACT_CHANGED_DURING_TRACE')
    receipt=dict(passed=True,registered_job=job,mode=mode,formal_gate_sha256=gate,
        science_root=str(science.resolve()),source_folder=str(folder.resolve()),target=str(target.resolve()),
        source_point_sha256=sha(point_path),source_freeze_sha256=bound['prediction_freeze.json'],
        bound_files=bound,physical_ledger_sha256=bound['physical_ledger.json'],
        reference_training_prediction_sha256=sha(folder/'audited_training_prediction.npy'),
        runtime_files={name:dict(file=relative,sha256=sha(science/relative)) for name,relative in runtime_files.items()},
        full_history=[1961,2024],daily_reach_trace_years=[2016,2024],years=years,
        training_objective=float(value),network=network,**comparison,
        routing_metadata=a.model.daily_data.metadata,routing_trace_sha256=sha(rp),shapes_sha256=sha(sp),
        source_multiplier=1.,new_scientific_training_calls=0,diagnostic_forward_calls=1,
        no_evaluation_outcomes_read=True,no_physical_equations_or_source_labels_changed=True,
        full44_completed_claimed=False,seconds=time.monotonic()-start,
        source_sha256=sha(Path(__file__)),observer_sha256=sha(ROOT/'analysis/internal_flux_observer.py'),
        shape_helper_sha256=sha(ROOT/'analysis/physical_waveform_diagnostics.py'))
    write_json(target/'complete.json',receipt)
    print(json.dumps(dict(passed=True,job=jid,mode=mode,years=len(years),seconds=receipt['seconds'],target=str(target))))


if __name__=='__main__': main(sys.argv[1],sys.argv[2])
