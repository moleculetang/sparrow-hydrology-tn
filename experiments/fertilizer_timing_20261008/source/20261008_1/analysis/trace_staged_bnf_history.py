"""Passive BNF calendar provenance replay at an independently frozen point.

BNF remains the unchanged uniform research input. These tags describe its
modelled inventory paths, not a counterfactual or measured biological timing.
Scientific equations, annual targets and physical recurrence are unmodified.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'analysis'), str(ROOT/'deployment')]
from trace_approved_internal_fluxes import science_root, install_paths, require_scientific_module
from trace_approved_internal_fluxes import export_comparison, prediction_comparison
from bnf_calendar_partition import partition, LABELS


def main(jid, mode):
    jobs = json.loads((ROOT/'config/jobs.json').read_text())
    job = next(j for j in jobs if j['id'] == jid)
    science = science_root(job)
    if any(n in sys.modules for n in ('timing.chain','timing.adapter','train_job','freeze_job')):
        raise RuntimeError('FRESH_PROCESS_FOR_MATCHING_BNF_RUNTIME_REQUIRED')
    install_paths(science)
    from d29_platform.runtime import configure, sha, write_json
    configure()
    from host_resources import registry, registry_root
    namespace = 'bnf_trace_approved_'+mode+'_v1'
    lease = registry.status(root=registry_root())['leases'].get(os.environ.get('D29_LEASE_TOKEN'))
    if (not lease or lease['project'] != '20261008_1' or lease['pid'] != os.getpid()
            or lease['created'] != registry.identity(os.getpid()) or lease['threads'] != 1
            or lease['job'] != 'analysis/'+namespace+'/'+jid):
        raise RuntimeError('EXACT_SINGLE_THREAD_BNF_TRACE_LEASE_REQUIRED')
    if mode != 'staged':
        raise RuntimeError('STAGED_BNF_ONLY_ENTRY_REQUIRED')
    from staged_frozen_sources import resolve
    folder = resolve(jid)
    from diagnose_approved_waveforms import accepted_point, approved_gate
    support = json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    physical, freeze, bound = accepted_point(folder,job,support)
    install_paths(science)
    import train_job, freeze_job, source_ledger
    import timing.chain as chain_module
    import timing.adapter as adapter_module
    import timing.profiles as profile_module
    import routing as routing_module
    import d29_platform.timing_labels as labels_module
    modules = [train_job,freeze_job,source_ledger,chain_module,adapter_module,
               profile_module,routing_module,labels_module]
    runtime_files = {m.__name__: require_scientific_module(m,science) for m in modules}
    _, gate = train_job.validate_admission()
    if gate != approved_gate(job) or gate != freeze['formal_gate_sha256']:
        raise RuntimeError('MATCHING_BNF_FROZEN_GATE_REQUIRED')
    if next(j for j in json.loads((science/'config/jobs.json').read_text()) if j['id']==jid) != job:
        raise RuntimeError('MATCHING_BNF_SCIENTIFIC_JOB_REQUIRED')
    import numpy as np
    import pandas as pd
    target = science/'outputs/jobs'/(namespace+'_'+jid)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('EXISTING_BNF_TRACE_REVIEW_NO_RERUN')
    point = np.load(folder/'frozen_parameters.npy',allow_pickle=False)
    a = freeze_job.training_only_adapter(dict(job,id=namespace+'_'+jid))
    if a.folder.resolve() != target.resolve():
        raise RuntimeError('MATCHING_BNF_DIAGNOSTIC_NAMESPACE_REQUIRED')
    contract = json.loads((folder/'training_contract.json').read_text())
    from native_bound_adapter_v2 import canonical_bounds
    if (a.names != contract['names'] or a.scale.tolist() != contract['scale']
            or [list(p) for p in canonical_bounds(a.bounds)] != contract['bounds']):
        raise RuntimeError('EXACT_BNF_FROZEN_PARAMETER_CONTRACT_REQUIRED')
    dates = pd.date_range('1961-01-01','2024-12-31')
    if not a.data.dates.equals(dates):
        raise RuntimeError('FULL64YEAR_BNF_DATE_SUPPORT_REQUIRED')
    dependencies = ['inputs/reference/agriculture_reference/annual_crop_activity_reference.parquet',
                    'inputs/calendar_support/reach_profiles.json','config/conditional_reference.json']
    input_bindings = {p:sha(science/p) for p in dependencies}
    activity = pd.read_parquet(science/dependencies[0])
    profiles = json.loads((science/dependencies[1]).read_text())
    residue = json.loads((science/dependencies[2]).read_text())['crop_residue']
    with np.load(folder/'full_history_physical_source_ledger.npz',allow_pickle=False) as z:
        references = {k:z[v] for k,v in [('local','local_kg_day'),('fast','local_fast_kg_day'),('slow','local_slow_kg_day')]}
    if any(x.shape!=(len(dates),230) or not np.isfinite(x).all() for x in references.values()):
        raise RuntimeError('EXACT230REACH_BNF_REFERENCE_REQUIRED')
    state = a.inputs.initial()
    tagged_state = np.zeros((*state.shape[:2],len(LABELS),5),dtype=np.float64)
    tagged_state[...,0,:] = state  # Initial stocks are assigned exactly once.
    correction = None
    local_tags = np.zeros((len(dates),230,len(LABELS)),dtype=np.float64)
    years = []; crop_rows = []; native = chain_module.TimingChain._run
    start = time.monotonic()
    def wrapped(chain,year,initial,eta,corr):
        nonlocal tagged_state,correction
        if eta != 0.:
            raise RuntimeError('BNF_TRACE_SOURCE_MULTIPLIER_NOT_ONE')
        result,mask = native(chain,year,initial,eta,corr)
        ix = np.flatnonzero(a.data.dates.year==year)
        post = initial[...,0].copy()
        if int(ix[0]) in a.inputs.transitions:
            post[:,:12] = np.einsum('ri,rij->rj',post[:,:12],a.inputs.transitions[int(ix[0])])
        generated = a.inputs.annual(year,post)
        tags,rows,partition_receipt = partition(year,generated['tags'],a.inputs.labels,
            activity[activity.year.eq(year)],profiles,lambda y,s:profile_module.profile(y,s)[2],
            residue['returned'],residue['exported_unresolved_fates'])
        del generated
        z = source_ledger.propagate_source_labels(result,tagged_initial=tagged_state,
            tagged_sources=tags,labels=LABELS,record_history=True,compensated=True,
            initial_compensation=correction)
        represented_final = z.final.copy()
        represented_final[...,1:] -= z.compensation
        maxima = dict(local=result.max_local_balance_kg,source_balance=z.max_local_balance_kg,
                      state_sum=z.max_state_sum_error_kg,flux_sum=z.max_flux_sum_error_kg)
        actual = dict(local=result.fluxes[...,:2].sum(axis=(2,3)),
                      fast=result.fluxes[...,0].sum(axis=2),slow=result.fluxes[...,1].sum(axis=2))
        comparisons = {k:export_comparison(actual[k],references[k][ix]) for k in actual}
        minimum = min(float(z.state_history.min()),float(represented_final.min()),float(z.flux_history.min()))
        if max(maxima.values())>1e-6 or minimum < -1e-7 or not all(c['passed'] for c in comparisons.values()):
            write_json(target/'failed_year.json',dict(year=year,maxima=maxima,minimum=minimum,comparisons=comparisons))
            raise RuntimeError('BNF_SOURCE_REPLAY_FAILED '+str(year))
        local_tags[ix] = z.flux_history[...,:2].sum(axis=(2,4))
        file = target/'yearly'/f'{year}.npz';file.parent.mkdir(exist_ok=True)
        arrays = dict(date=dates[ix].to_numpy(dtype='datetime64[D]'),labels=np.asarray(LABELS),
            source_by_land_label_entry_kg_day=tags.sum(axis=1),
            stock_high_by_land_label_state_kg_day=z.state_history[1:].sum(axis=1),
            flux_by_land_label_destination_kg_day=z.flux_history.sum(axis=1),
            end_year_stock_by_reach_land_label_state_kg=represented_final,
            end_year_high=z.final,end_year_compensation=z.compensation)
        if year>=2016:
            arrays['local_by_reach_label_kg_day'] = local_tags[ix]
        np.savez_compressed(file,**arrays)
        tagged_state = z.final.copy();correction = z.compensation.copy()
        crop_rows.extend(rows)
        years.append(dict(year=year,passed=True,maxima=maxima,minimum_label_kg=minimum,
            partition=partition_receipt,reference_exports=comparisons,file=str(file.relative_to(target)),sha256=sha(file)))
        write_json(target/'progress.json',dict(completed_years=len(years),last_year=year,
            job=jid,mode=mode,evaluation_labels_read=False))
        return result,mask
    chain_module.TimingChain._run = wrapped
    try:
        value,_ = a.value_gradient(point,forward_only=True)
    finally:
        chain_module.TimingChain._run = native
    if [r['year'] for r in years] != list(range(1961,2025)):
        raise RuntimeError('EXACT64YEARS_BNF_REQUIRED')
    base,_ = a.unpack(point)
    if base[a.eta_index] != 0.:
        raise RuntimeError('BNF_TRACE_MULTIPLIER_IDENTITY')
    from internal_flux_observer import routing_checks
    routed = routing_module.route(a.model.daily_data,references['local'],vf=float(base[a.base_names.index('v_f')]))
    _, network = routing_checks(references['local'],routed)
    local_error = float(np.max(abs(local_tags.sum(axis=2)-references['local']),initial=0.))
    comparison = prediction_comparison(a.prediction,np.load(folder/'audited_training_prediction.npy',allow_pickle=False),value,physical['objective'])
    if (not network['passed'] or local_error>1e-6 or not comparison['prediction_passed'] or not comparison['objective_passed']):
        raise RuntimeError('BNF_TRACE_PREDICTION_OR_ROUTING_CHANGED')
    table = target/'crop_BNF_growth_support_2016_2024.csv'
    pd.DataFrame(crop_rows).to_csv(table,index=False)
    _,_,after = accepted_point(folder,job,support)
    if after != bound or any(sha(science/p)!=digest for p,digest in input_bindings.items()):
        raise RuntimeError('BNF_FROZEN_ARTIFACT_OR_DIAGNOSTIC_INPUT_CHANGED')
    receipt = dict(passed=True,registered_job=job,mode=mode,formal_gate_sha256=gate,
        source_folder=str(folder.resolve()),target=str(target.resolve()),science_root=str(science.resolve()),
        bound_files=bound,source_point_sha256=sha(folder/'frozen_parameters.npy'),
        source_freeze_sha256=bound['prediction_freeze.json'],input_bindings=input_bindings,
        runtime_files={n:dict(file=p,sha256=sha(science/p)) for n,p in runtime_files.items()},
        years=years,full_history=[1961,2024],labels=list(LABELS),states=['P','Sa','Sp','N','L'],
        destinations=['fast','slow','loss','plant_export'],source_entries=['plant','active_SON','protected_SON','available_N'],
        source_multiplier=1.,crop_rows=len(crop_rows),crop_table_sha256=sha(table),
        local_source_sum_max_error_kg=local_error,network=network,training_objective=float(value),**comparison,
        inventory_interpretation='passive_model_provenance_not_counterfactual_or_actual_BNF_dates',
        daily_stocks='high_components_only; corrected_endpoints_and_compensation_saved_separately',
        V0_growth_windows='V1_configured_calendar_reference_not_V0_capacity',
        unknown_calendar='unknown_not_false_fallow',BNF_timing_changed=False,
        no_physical_equations_changed=True,evaluation_labels_read=False,NSE='not_applicable',
        new_training_calls=0,diagnostic_forward_calls=1,full44_completed_claimed=False,
        seconds=time.monotonic()-start,worker_sha256=sha(Path(__file__)),
        partition_sha256=sha(ROOT/'analysis/bnf_calendar_partition.py'))
    write_json(target/'complete.json',receipt)
    print(json.dumps(dict(passed=True,job=jid,mode=mode,years=len(years),target=str(target))))


if __name__ == '__main__':main(sys.argv[1],sys.argv[2])
