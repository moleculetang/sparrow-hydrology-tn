"""Replay an immutable accepted point with read-only yearly flux observers.

Writes only a new analysis namespace. Every annual local fast/slow/export array
must agree with the independently accepted source ledger within the existing
1e-6kg cross-host mass tolerance; bitwise status is retained separately.
Evaluation outcomes are not read or scored.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
import numpy as np
import pandas as pd
from d29_platform.runtime import write_json,sha
from physical_waveform_diagnostics import verified,annual_shapes
from internal_flux_observer import observe,routing_checks,stage_semantics


def main(source_folder,namespace):
    from host_resources import registry,registry_root
    from train_job import validate_admission
    from freeze_job import training_only_adapter
    from timing.chain import TimingChain
    validate_admission()
    token=os.environ.get('D29_LEASE_TOKEN')
    lease=registry.status(root=registry_root())['leases'].get(token)
    if not lease or lease['pid']!=os.getpid() or lease['created']!=registry.identity(os.getpid()) or lease['project']!='20261008_1':
        raise RuntimeError('EXACT_BOUND_ANALYSIS_LEASE_REQUIRED')
    source_folder=Path(source_folder).resolve()
    if not source_folder.is_relative_to(ROOT.resolve()):raise ValueError('IN_EXPERIMENT_ACCEPTED_POINT_REQUIRED')
    physical,identity,kind=verified(source_folder)
    jid=identity['registered_job'] if kind=='fixed_parameter_replay' else identity['job']['id']
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    if not namespace.startswith('internal_trace_') or '/' in namespace or '\\' in namespace:
        raise ValueError('NEW_INTERNAL_TRACE_NAMESPACE_REQUIRED')
    expected_lease='analysis/'+namespace+'/'+jid
    if lease['job']!=expected_lease:raise RuntimeError('INTERNAL_TRACE_JOB_LEASE_MISMATCH')
    # The sealed adapter permits only outputs/jobs. A distinct nonregistered
    # diagnostic ID keeps its writes away from all44owned scientific paths.
    target=ROOT/'outputs/jobs'/(namespace+'_'+jid)
    if (target/'complete.json').exists():raise RuntimeError('ACCEPTED_TRACE_ALREADY_EXISTS_NO_RERUN')
    if target.exists() and any(target.iterdir()):raise RuntimeError('PARTIAL_TRACE_REQUIRES_REVIEW_BEFORE_RECOVERY')
    target.mkdir(parents=True,exist_ok=True)
    point_path=source_folder/('point.npy' if kind=='fixed_parameter_replay' else 'frozen_parameters.npy')
    point=np.load(point_path,allow_pickle=False)
    original=dict(job);derived=dict(job,id=namespace+'_'+jid)
    # Predicate pushdown excludes heldout rows before the adapter sees them;
    # the previously completed diagnostic batches used logical train slicing.
    a=training_only_adapter(derived)
    if a.folder.resolve()!=target.resolve():raise RuntimeError('DIAGNOSTIC_NAMESPACE_REDIRECTION_FAILED')
    years=[];shapes=[];native=TimingChain._run;start=time.monotonic()
    with np.load(source_folder/'full_history_physical_source_ledger.npz',allow_pickle=False) as saved:
        local=saved['local_kg_day'];fast=saved['local_fast_kg_day'];slow=saved['local_slow_kg_day']
    def wrapped(chain,year,initial,eta,correction):
        result,mask=native(chain,year,initial,eta,correction)
        ix=np.flatnonzero(a.data.dates.year==year)
        actual=dict(local=result.fluxes[...,:2].sum(axis=(2,3)),
                    fast=result.fluxes[...,0].sum(axis=2),slow=result.fluxes[...,1].sum(axis=2))
        expected=dict(local=local[ix],fast=fast[ix],slow=slow[ix])
        comparisons={k:dict(bitwise=np.array_equal(actual[k],expected[k]),
            max_abs_difference_kg=float(np.max(abs(actual[k]-expected[k]),initial=0.)),
            passed=bool(np.all(abs(actual[k]-expected[k])<=1e-6))) for k in actual}
        arrays,receipt=observe(result)
        if not receipt['passed'] or not all(c['passed'] for c in comparisons.values()):
            write_json(target/'failed_year.json',dict(year=year,receipt=receipt,source_ledger_export_comparison=comparisons))
            raise RuntimeError('READ_ONLY_INTERNAL_REPLAY_FAILED '+str(year))
        # Daily reach sums are retained in the intervention period; full-history
        # land/day and annual reach/land diagnostics preserve warmup provenance.
        output={k:v for k,v in arrays.items() if year>=2016 or not k.endswith('_daily_by_reach')}
        output['date']=a.data.dates[ix].to_numpy(dtype='datetime64[D]')
        p=target/'yearly'/f'{year}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,**output)
        for key,values in arrays.items():
            if key.endswith('_daily_by_land'):
                shapes.extend(annual_shapes(values.sum(axis=1),a.data.dates[ix],key.removesuffix('_daily_by_land'),'kg N/day'))
        years.append(dict(year=year,passed=True,internal_receipt=receipt,
                          source_ledger_export_comparison=comparisons,file=str(p.relative_to(target)),sha256=sha(p)))
        write_json(target/'progress.json',dict(completed_years=len(years),last_year=year,
            registered_job=original,no_evaluation_outcomes_read=True))
        return result,mask
    TimingChain._run=wrapped
    try:value,_=a.value_gradient(point,forward_only=True)
    finally:TimingChain._run=native
    from routing import route
    base,_=a.unpack(point);vf=float(base[a.base_names.index('v_f')])
    routing=route(a.model.daily_data,local,vf=vf)
    residual,network=routing_checks(local,routing)
    if not network['passed']:
        write_json(target/'failed_network.json',network)
        raise RuntimeError('INTERNAL_TRACE_NETWORK_CLOSURE_FAILED')
    keep=(a.data.dates.year>=2016)
    rp=target/'routing_trace.npz'
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
    for row in shapes:row.update(stage_semantics(row['stage']))
    sp=target/'internal_and_network_shapes.csv';pd.DataFrame(shapes).to_csv(sp,index=False)
    reference_file=source_folder/('training_support_prediction.npy' if kind=='fixed_parameter_replay' else 'audited_training_prediction.npy')
    if kind=='fixed_parameter_replay' and sha(reference_file)!=identity['prediction_sha256']:
        raise RuntimeError('ACCEPTED_REPLAY_TRAINING_PREDICTION_CHANGED')
    reference_prediction=np.load(reference_file,allow_pickle=False)
    prediction_equal=np.array_equal(a.prediction,reference_prediction)
    prediction_error=float(np.max(abs(a.prediction-reference_prediction),initial=0.))
    prediction_passed=bool(np.all(abs(a.prediction-reference_prediction)<=1e-8*(1+abs(reference_prediction))))
    error=abs(float(value)-float(physical['objective']))
    if not prediction_passed or error>1e-8*(1+abs(value)):
        write_json(target/'failed_prediction.json',dict(bitwise_training_prediction=prediction_equal,
            training_prediction_max_error_mg_l=prediction_error,prediction_passed=prediction_passed,objective_error=error))
        raise RuntimeError('OBSERVED_FORWARD_CHANGED_ACCEPTED_PREDICTION')
    if sha(point_path)!=(identity['point_sha256'] if kind=='fixed_parameter_replay' else identity['parameter_sha256']):
        raise RuntimeError('IMMUTABLE_POINT_CHANGED_DURING_TRACE')
    receipt=dict(passed=True,registered_job=original,scope=kind+'_read_only_internal_and_network_replay',
        full_history=[1961,2024],daily_reach_trace_years=[2016,2024],years=years,
        source_folder=str(source_folder.relative_to(ROOT)),source_point_sha256=sha(point_path),
        physical_ledger_sha256=sha(source_folder/'physical_ledger.json'),
        reference_training_prediction_sha256=sha(reference_file),
        formal_gate_sha256=sha(ROOT/'evidence/formal_gate.json'),bitwise_training_prediction=prediction_equal,
        training_prediction_max_error_mg_l=prediction_error,prediction_passed=prediction_passed,
        cross_host_mass_tolerance_kg=1e-6,cross_host_prediction_tolerance='1e-8*(1+abs(reference))',
        training_objective=value,objective_replay_error=error,network=network,
        routing_metadata=a.model.daily_data.metadata,routing_trace_sha256=sha(rp),shapes_sha256=sha(sp),
        new_scientific_training_calls=0,diagnostic_forward_calls=1,no_evaluation_outcomes_read=True,
        no_physical_equations_or_source_labels_changed=True,seconds=time.monotonic()-start,
        identities={n:sha(ROOT/n) for n in ['analysis/internal_flux_observer.py','analysis/trace_accepted_internal_fluxes.py']})
    write_json(target/'complete.json',receipt)
    print(json.dumps(dict(passed=True,registered_job=jid,years=len(years),seconds=receipt['seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_folder');p.add_argument('--namespace',default='internal_trace_pilot_v1')
    a=p.parse_args();main(a.source_folder,a.namespace)
