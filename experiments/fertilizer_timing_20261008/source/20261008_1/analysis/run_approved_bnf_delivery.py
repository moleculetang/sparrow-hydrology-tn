"""Resource-gated BNF-only diagnostic delivery; never a science scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from controller_lock import acquire
from trace_approved_internal_fluxes import science_root
from bnf_calendar_partition import LABELS


def require_controls():
    path = ROOT/'outputs/bnf_partition_software_controls/known_values_and_refusals.json'
    value = json.loads(path.read_text())
    expected = {'known_growth_and_unknown_not_false_fallow','two_independent_BNF_entrances_and_zero_growth_plan',
        'no_calendar_intervention_during_warmup','known_zero_without_epsilon_or_nan',
        'annual_BNF_mass_change_rejected','duplicate_crop_budget_rejected','unknown_BNF_mass_rejected',
        'missing_activity_not_known_zero','missing_reach_not_known_zero','double_residue_fate_rejected',
        'unregistered_BNF_entry_rejected','invalid_growth_profile_rejected','negative_source_not_clipped'}
    if (value.get('passed') is not True or set(value.get('cases',{}))!=expected
            or any(v is not True for v in value['cases'].values()) or value.get('evaluation_labels_read') is not False
            or value.get('implementation_sha256')!={p:sha(ROOT/p) for p in
                ('analysis/bnf_calendar_partition.py','analysis/test_bnf_calendar_partition.py')}):
        raise RuntimeError('CURRENT_BNF_PARTITION_CONTROLS_REQUIRED')
    return sha(path)


def verified_case(destination,source,job,mode):
    from diagnose_approved_waveforms import accepted_point,approved_gate
    support = json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    _,freeze,bound = accepted_point(source,job,support)
    v = json.loads((destination/'complete.json').read_text())
    science = science_root(job)
    worker=ROOT/'analysis/trace_approved_bnf_history.py'
    if mode=='staged':
        from staged_frozen_sources import resolve
        from build_staged_bnf_worker import derive
        if source.resolve()!=resolve(job['id']).resolve():
            raise RuntimeError('EXACT_STAGED_BNF_SOURCE_REQUIRED')
        candidate=ROOT/'analysis/trace_staged_bnf_history.py'
        if candidate.read_text(encoding='utf8')!=derive(worker.read_text(encoding='utf8')):
            raise RuntimeError('EXACT_AUDITED_BNF_BODY_REQUIRED')
        worker=candidate
    if (v.get('passed') is not True or v.get('registered_job')!=job or v.get('mode')!=mode
            or v.get('formal_gate_sha256')!=approved_gate(job) or v.get('bound_files')!=bound
            or v.get('source_point_sha256')!=freeze['parameter_sha256']
            or v.get('source_freeze_sha256')!=bound['prediction_freeze.json']
            or v.get('target')!=str(destination.resolve()) or v.get('source_folder')!=str(source.resolve())
            or v.get('science_root')!=str(science.resolve()) or v.get('labels')!=list(LABELS)
            or v.get('source_multiplier')!=1. or v.get('full_history')!=[1961,2024]
            or v.get('new_training_calls')!=0 or v.get('diagnostic_forward_calls')!=1
            or v.get('BNF_timing_changed') is not False or v.get('no_physical_equations_changed') is not True
            or v.get('evaluation_labels_read') is not False or v.get('prediction_passed') is not True
            or v.get('objective_passed') is not True or v.get('network',{}).get('passed') is not True
            or not 0<=v.get('local_source_sum_max_error_kg',float('inf'))<=1e-6
            or v.get('worker_sha256')!=sha(worker)
            or v.get('partition_sha256')!=sha(ROOT/'analysis/bnf_calendar_partition.py')):
        raise RuntimeError('EXACT_BNF_DIAGNOSTIC_BINDINGS_REQUIRED')
    expected = {'train_job','freeze_job','source_ledger','timing.chain','timing.adapter',
                'timing.profiles','routing','d29_platform.timing_labels'}
    if set(v.get('runtime_files',{}))!=expected:
        raise RuntimeError('COMPLETE_BNF_SCIENTIFIC_RUNTIME_REQUIRED')
    for relative,digest in v.get('input_bindings',{}).items():
        p=(science/relative).resolve()
        if not p.is_relative_to(science.resolve()) or sha(p)!=digest:
            raise RuntimeError('BNF_DIAGNOSTIC_INPUT_CHANGED')
    if set(v.get('input_bindings',{}))!={'inputs/reference/agriculture_reference/annual_crop_activity_reference.parquet',
                                      'inputs/calendar_support/reach_profiles.json','config/conditional_reference.json'}:
        raise RuntimeError('BNF_DIAGNOSTIC_INPUT_COVERAGE_REQUIRED')
    for record in v['runtime_files'].values():
        p=(science/record['file']).resolve()
        if not p.is_relative_to(science.resolve()) or sha(p)!=record['sha256']:
            raise RuntimeError('BNF_SCIENTIFIC_RUNTIME_BYTES_CHANGED')
    years=v.get('years',[])
    if [r['year'] for r in years]!=list(range(1961,2025)):
        raise RuntimeError('EXACT64YEAR_BNF_COVERAGE_REQUIRED')
    import numpy as np
    import pandas as pd
    for row in years:
        p=(destination/row['file']).resolve()
        if (not p.is_relative_to((destination/'yearly').resolve()) or sha(p)!=row['sha256']
                or row.get('passed') is not True or not row.get('maxima')
                or any(not np.isfinite(x) or not 0<=x<=1e-6 for x in row['maxima'].values())
                or not np.isfinite(row['minimum_label_kg']) or row['minimum_label_kg'] < -1e-7
                or not row.get('reference_exports') or any(c.get('passed') is not True for c in row['reference_exports'].values())):
            raise RuntimeError('BNF_ANNUAL_RECEIPT_CHANGED')
        with np.load(p,allow_pickle=False) as z:
            n=366 if __import__('calendar').isleap(row['year']) else 365
            dates=pd.date_range(f"{row['year']}-01-01",f"{row['year']}-12-31").to_numpy(dtype='datetime64[D]')
            if not np.array_equal(z['date'],dates) or z['labels'].tolist()!=list(LABELS):
                raise RuntimeError('BNF_ANNUAL_LABEL_DATE_IDENTITY')
            expected_shapes={'source_by_land_label_entry_kg_day':(n,13,5,4),
                'stock_high_by_land_label_state_kg_day':(n,13,5,5),
                'flux_by_land_label_destination_kg_day':(n,13,5,4),
                'end_year_stock_by_reach_land_label_state_kg':(230,13,5,5),
                'end_year_high':(230,13,5,5),'end_year_compensation':(230,13,5,4)}
            if row['year']>=2016:expected_shapes['local_by_reach_label_kg_day']=(n,230,5)
            for key,shape in expected_shapes.items():
                a=z[key]
                if a.shape!=shape or not np.isfinite(a).all() or (key!='end_year_compensation' and a.min() < -1e-7):
                    raise RuntimeError('BNF_ANNUAL_ARRAY_SUPPORT_OR_NONNEGATIVITY')
            corrected=z['end_year_high'].copy();corrected[...,1:]-=z['end_year_compensation']
            if not np.array_equal(corrected,z['end_year_stock_by_reach_land_label_state_kg']):
                raise RuntimeError('BNF_ENDPOINT_COMPENSATION_IDENTITY')
    table=destination/'crop_BNF_growth_support_2016_2024.csv'
    if sha(table)!=v['crop_table_sha256']:
        raise RuntimeError('BNF_CROP_TABLE_BYTES_CHANGED')
    rows=pd.read_csv(table)
    if (len(rows)!=v['crop_rows'] or set(rows.year)!=set(range(2016,2025))
            or rows.duplicated(['year','reach_id','crop']).any()
            or any(set(g.reach_id)!=set(range(1,231)) for _,g in rows.groupby('year'))):
        raise RuntimeError('COMPLETE_NONREPLICATED_BNF_CROP_SUPPORT_REQUIRED')
    return dict(job=job['id'],target=str(destination),complete_sha256=sha(destination/'complete.json'),
        formal_gate_sha256=v['formal_gate_sha256'],seconds=v['seconds'],crop_rows=v['crop_rows'],
        local_source_sum_max_error_kg=v['local_source_sum_max_error_kg'],objective_replay_error=v['objective_replay_error'])


def main(mode):
    if os.name!='nt' or mode not in ('preflight','final'):
        raise RuntimeError('LOCAL_DECLARED_BNF_DELIVERY_REQUIRED')
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    if mode=='preflight':
        selected=[j for j in jobs if j['id']=='TIM_F24_C0_V0_s0']
        sources={j['id']:ROOT/'outputs/jobs'/j['id'] for j in selected}
    else:
        from approved_epoch_delivery import admitted_folders
        sources=admitted_folders(ROOT)  # Reject incomplete44 before acquiring resources.
        selected=jobs
    controls_sha=require_controls()
    namespace='bnf_trace_approved_'+mode+'_v1'
    folder=ROOT/'outputs'/('approved_bnf_'+mode);folder.mkdir(exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('BNF_DELIVERY_ALREADY_COMPLETE_NO_RERUN')
    marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('BNF_CONTROLLER_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(marker,owner);cases=[];events=[];start=time.monotonic()
    for job in selected:
        jid=job['id'];destination=science_root(job)/'outputs/jobs'/(namespace+'_'+jid)
        if (destination/'complete.json').exists():
            cases.append(verified_case(destination,sources[jid],job,mode));continue
        if destination.exists() and any(destination.iterdir()):raise RuntimeError('PARTIAL_BNF_TRACE_REQUIRES_REVIEW')
        preflight=ROOT/'outputs/jobs'/('bnf_trace_approved_preflight_v1_'+jid)
        if mode=='final' and jid=='TIM_F24_C0_V0_s0' and (preflight/'complete.json').exists():
            from preflight_replay_reuse import require_controls,verified_for_final
            require_controls('bnf',job)
            cases.append(verified_for_final('bnf',preflight,sources[jid],job));continue
        staged=science_root(job)/'outputs/jobs'/('bnf_trace_approved_staged_v1_'+jid)
        if mode=='final' and (staged/'complete.json').exists():
            proof=json.loads((ROOT/'evidence/staged_bnf_reuse_controls.json').read_text())
            if (not proof.get('passed') or (proof.get('actual_full_history_cases')!=len(proof.get('cases',[])) or jid not in {c['job'] for c in proof.get('cases',[])} or len({c['job'] for c in proof.get('cases',[])})!=len(proof.get('cases',[]))) or
                    proof.get('exact44_admission_preserved') is not True or
                    proof.get('resolver_sha256')!=sha(ROOT/'analysis/staged_frozen_sources.py') or
                    proof.get('batch_reader_sha256')!=sha(ROOT/'analysis/staged_batch_receipts.py') or
                    proof.get('source_sha256')!=sha(ROOT/'analysis/reuse_staged_bnf_trace.py') or
                    proof.get('launcher_sha256')!=sha(Path(__file__)) or
                    proof.get('final_verifier_sha256')!=sha(ROOT/'analysis/audit_final_approved_delivery.py')):
                raise RuntimeError('CURRENT_ACTUAL_STAGED_BNF_REUSE_CONTROLS_REQUIRED')
            from reuse_staged_bnf_trace import verified_for_final
            cases.append(verified_for_final(staged,sources[jid],job));continue
        lease=registry.acquire('20261008_1','analysis/'+namespace+'/'+jid,
            threads=1,peak_bytes=6500000000,root=registry_root())
        write_json(folder/'resource_decision.json',lease)
        if not lease.get('granted'):
            write_json(folder/'status.json',dict(phase='resource_wait_no_science_changes',owner=owner,mode=mode,cases=cases))
            return 4
        child=None;peak=dict(rss=0,private_commit=0,peak_commit=0)
        try:
            with (folder/(jid+'.log')).open('ab') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                    lease['token'],'analysis/trace_approved_bnf_history.py',jid,mode],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                registry.bind(lease['token'],child.pid,root=registry_root())
                write_json(folder/'status.json',dict(phase='BNF_history_replay_running',owner=owner,job=jid,
                    child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,mode=mode,cases=cases))
                while child.poll() is None:
                    memory=registry.process_memory(child.pid)
                    for key in peak:
                        if memory.get(key) is not None:peak[key]=max(peak[key],memory[key])
                    time.sleep(.5)
            event=dict(job=jid,exit_code=child.returncode,peak=peak);events.append(event)
            write_json(folder/(jid+'.returned.json'),event)
            if child.returncode:
                write_json(folder/'status.json',dict(phase='BNF_trace_requires_review',owner=owner,event=event,cases=cases,mode=mode))
                return 2
            cases.append(verified_case(destination,sources[jid],job,mode))
        finally:
            if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
    expected=1 if mode=='preflight' else 44
    if len(cases)!=expected or len({c['job'] for c in cases})!=expected:raise RuntimeError('EXACT_BNF_MODE_PATH_COUNT_REQUIRED')
    result=dict(passed=True,mode=mode,cases=cases,events=events,accounted_paths=len(cases),
        full44_BNF_completed=mode=='final',actual_K_cases=sum(j['calendar']=='K' for j in selected),
        controls_sha256=controls_sha,evaluation_labels_read=False,new_training_calls=0,
        staged_cases_reused=sum(c.get('execution_mode')=='staged_BNF_reused_after_final44' for c in cases),
        preflight_cases_reused=sum(c.get('execution_mode')=='preflight_BNF_reused_after_final44' for c in cases),
        diagnostic_forward_calls=len(cases),new_forward_calls_this_delivery=len(events),
        seconds=time.monotonic()-start,launcher_sha256=sha(Path(__file__)),
        worker_sha256=sha(ROOT/'analysis/trace_approved_bnf_history.py'),partition_sha256=sha(ROOT/'analysis/bnf_calendar_partition.py'))
    write_json(folder/'complete.json',result)
    write_json(folder/'status.json',dict(owner=owner,phase='preflight_complete_not_full44' if mode=='preflight' else 'approved44_BNF_complete',**result))
    print(json.dumps(dict(passed=True,mode=mode,cases=len(cases),seconds=result['seconds'])))
    return 0


if __name__=='__main__':sys.exit(main(sys.argv[1]))
