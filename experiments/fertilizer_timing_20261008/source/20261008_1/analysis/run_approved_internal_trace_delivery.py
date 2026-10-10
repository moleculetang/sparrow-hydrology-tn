"""Local lease-bound full-history replays, with immutable resume receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from controller_lock import acquire


def verified_case(destination,source,job,mode):
    from trace_approved_internal_fluxes import science_root
    from diagnose_approved_waveforms import accepted_point,approved_gate
    support=json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    _,freeze,bound=accepted_point(source,job,support)
    value=json.loads((destination/'complete.json').read_text())
    science=science_root(job)
    worker=ROOT/'analysis/trace_approved_internal_fluxes.py'
    if mode=='staged':
        from staged_frozen_sources import resolve
        from build_staged_trace_worker import derive
        if source.resolve()!=resolve(job['id']).resolve():
            raise RuntimeError('EXACT_STAGED_FROZEN_SOURCE_REQUIRED')
        candidate=ROOT/'analysis/trace_staged_internal_fluxes.py'
        if candidate.read_text(encoding='utf8')!=derive(worker.read_text(encoding='utf8')):
            raise RuntimeError('EXACT_AUDITED_REPLAY_BODY_REQUIRED')
        worker=candidate
    if (value.get('passed') is not True or value.get('registered_job')!=job or value.get('mode')!=mode or
            value.get('formal_gate_sha256')!=approved_gate(job) or value.get('bound_files')!=bound or
            value.get('source_folder')!=str(source.resolve()) or value.get('target')!=str(destination.resolve()) or
            value.get('science_root')!=str(science.resolve()) or value.get('source_multiplier')!=1. or
            value.get('source_freeze_sha256')!=bound['prediction_freeze.json'] or
            value.get('source_point_sha256')!=freeze['parameter_sha256'] or
            value.get('source_sha256')!=sha(worker) or
            value.get('observer_sha256')!=sha(ROOT/'analysis/internal_flux_observer.py') or
            value.get('shape_helper_sha256')!=sha(ROOT/'analysis/physical_waveform_diagnostics.py') or
            value.get('diagnostic_forward_calls')!=1 or value.get('new_scientific_training_calls')!=0 or
            value.get('no_evaluation_outcomes_read') is not True or
            value.get('prediction_passed') is not True or value.get('objective_passed') is not True or
            value.get('network',{}).get('passed') is not True or value.get('full_history')!=[1961,2024]):
        raise RuntimeError('EXACT_FULL_HISTORY_TRACE_BINDING_REQUIRED')
    years=value.get('years',[])
    if [r['year'] for r in years]!=list(range(1961,2025)):
        raise RuntimeError('EXACT64YEAR_TRACE_RECEIPT_REQUIRED')
    for row in years:
        path=(destination/row['file']).resolve()
        if (not path.is_relative_to((destination/'yearly').resolve()) or
                row.get('passed') is not True or row.get('internal_receipt',{}).get('passed') is not True or
                not row.get('source_ledger_export_comparison') or
                any(c.get('passed') is not True for c in row['source_ledger_export_comparison'].values()) or
                sha(path)!=row['sha256']):
            raise RuntimeError('ANNUAL_INTERNAL_TRACE_CHANGED')
    for name,key in [('routing_trace.npz','routing_trace_sha256'),
                     ('internal_and_network_shapes.csv','shapes_sha256')]:
        if sha(destination/name)!=value[key]:raise RuntimeError('TRACE_OUTPUT_BYTES_CHANGED')
    expected={'train_job','freeze_job','timing.chain','routing','timing.adapter','d29_platform.runtime'}
    if set(value.get('runtime_files',{}))!=expected:
        raise RuntimeError('COMPLETE_SCIENTIFIC_RUNTIME_BINDING_REQUIRED')
    for record in value['runtime_files'].values():
        path=(science/record['file']).resolve()
        if not path.is_relative_to(science.resolve()) or sha(path)!=record['sha256']:
            raise RuntimeError('SCIENTIFIC_RUNTIME_BYTES_CHANGED')
    return dict(job=job['id'],formal_gate_sha256=value['formal_gate_sha256'],
        complete_sha256=sha(destination/'complete.json'),target=str(destination),
        bitwise_training_prediction=value['bitwise_training_prediction'],
        objective_replay_error=value['objective_replay_error'],seconds=value['seconds'])


def main(mode):
    if os.name!='nt' or mode not in ('preflight','final'):
        raise RuntimeError('LOCAL_DECLARED_TRACE_DELIVERY_REQUIRED')
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    if mode=='preflight':
        selected=[j for j in jobs if j['id']=='TIM_F24_C0_V0_s0']
        sources={j['id']:ROOT/'outputs/jobs'/j['id'] for j in selected}
    else:
        from approved_epoch_delivery import admitted_folders
        sources=admitted_folders(ROOT)  # Exact44 admission before resource requests.
        selected=jobs
    controls=json.loads((ROOT/'evidence/approved_internal_trace_controls.json').read_text())
    if (controls.get('passed') is not True or
            controls.get('worker_sha256')!=sha(ROOT/'analysis/trace_approved_internal_fluxes.py') or
            controls.get('launcher_sha256')!=sha(Path(__file__))):
        raise RuntimeError('CURRENT_TRACE_CONTROL_ACCEPTANCE_REQUIRED')
    from trace_approved_internal_fluxes import science_root
    namespace='internal_trace_approved_'+mode+'_v1'
    folder=ROOT/'outputs'/('approved_internal_trace_'+mode)
    folder.mkdir(exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('TRACE_DELIVERY_COMPLETED_NO_RERUN')
    marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if registry.identity(old['pid'])==old['created']:
            raise RuntimeError('TRACE_CONTROLLER_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(marker,owner);cases=[];events=[];start=time.monotonic()
    for job in selected:
        jid=job['id'];destination=science_root(job)/'outputs/jobs'/(namespace+'_'+jid)
        if (destination/'complete.json').exists():
            cases.append(verified_case(destination,sources[jid],job,mode));continue
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError('PARTIAL_TRACE_REQUIRES_REVIEW')
        staged=science_root(job)/'outputs/jobs'/('internal_trace_approved_staged_v1_'+jid)
        if mode=='final' and (staged/'complete.json').exists():
            proof=json.loads((ROOT/'evidence/staged_trace_reuse_controls.json').read_text())
            if (not proof.get('passed') or (proof.get('actual_full_history_cases')!=len(proof.get('cases',[])) or jid not in {c['job'] for c in proof.get('cases',[])} or len({c['job'] for c in proof.get('cases',[])})!=len(proof.get('cases',[]))) or
                    proof.get('exact44_admission_preserved') is not True or
                    proof.get('resolver_sha256')!=sha(ROOT/'analysis/staged_frozen_sources.py') or
                    proof.get('batch_reader_sha256')!=sha(ROOT/'analysis/staged_batch_receipts.py') or
                    proof.get('source_sha256')!=sha(ROOT/'analysis/reuse_staged_internal_trace.py') or
                    proof.get('launcher_sha256')!=sha(Path(__file__)) or
                    proof.get('final_verifier_sha256')!=sha(ROOT/'analysis/audit_final_approved_delivery.py')):
                raise RuntimeError('CURRENT_ACTUAL_STAGED_REUSE_CONTROLS_REQUIRED')
            from reuse_staged_internal_trace import verified_for_final
            cases.append(verified_for_final(staged,sources[jid],job));continue
        lease=registry.acquire('20261008_1','analysis/'+namespace+'/'+jid,
            threads=1,peak_bytes=6500000000,root=registry_root())
        write_json(folder/'resource_decision.json',lease)
        if not lease.get('granted'):
            write_json(folder/'status.json',dict(phase='resource_wait_same_controller_reentry_required',
                owner=owner,mode=mode,cases=cases,scientific_signals=0));return 4
        child=None;peak=dict(rss=0,private_commit=0,peak_commit=0)
        try:
            with (folder/(jid+'.log')).open('ab') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                    lease['token'],'analysis/trace_approved_internal_fluxes.py',jid,mode],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                registry.bind(lease['token'],child.pid,root=registry_root())
                write_json(folder/'status.json',dict(phase='full_history_internal_trace_running',
                    owner=owner,job=jid,child=dict(pid=child.pid,created=registry.identity(child.pid)),
                    lease=lease,mode=mode,cases=cases,scientific_signals=0))
                while child.poll() is None:
                    memory=registry.process_memory(child.pid)
                    for key in peak:
                        if memory.get(key) is not None:peak[key]=max(peak[key],memory[key])
                    time.sleep(.5)
            event=dict(job=jid,exit_code=child.returncode,peak=peak);events.append(event)
            write_json(folder/(jid+'.returned.json'),event)
            if child.returncode:
                write_json(folder/'status.json',dict(phase='internal_trace_requires_review',owner=owner,
                    event=event,cases=cases,mode=mode,scientific_signals=0));return 2
            cases.append(verified_case(destination,sources[jid],job,mode))
        finally:
            if child is None or child.poll() is not None:
                registry.release(lease['token'],root=registry_root())
    expected=1 if mode=='preflight' else 44
    if len(cases)!=expected or len({c['job'] for c in cases})!=expected:
        raise RuntimeError('EXACT_MODE_TRACE_ACCOUNTING_REQUIRED')
    result=dict(passed=True,mode=mode,cases=cases,events=events,accounted_paths=len(cases),
        full44_internal_replay_completed=mode=='final',actual_K_cases=sum(j['calendar']=='K' for j in selected),
        scientific_signals=0,evaluation_labels_read=False,new_training_calls=0,
        diagnostic_forward_calls=len(cases),new_forward_calls_this_delivery=len(events),
        staged_cases_reused=sum(c.get('execution_mode')=='staged_reused_after_final44' for c in cases),
        seconds=time.monotonic()-start,
        launcher_sha256=sha(Path(__file__)),worker_sha256=sha(ROOT/'analysis/trace_approved_internal_fluxes.py'))
    write_json(folder/'complete.json',result);write_json(folder/'status.json',dict(owner=owner,
        phase='preflight_complete_not_full44' if mode=='preflight' else 'approved44_internal_replay_complete',**result))
    print(json.dumps(dict(passed=True,mode=mode,cases=len(cases),seconds=result['seconds'])))
    return 0


if __name__=='__main__': sys.exit(main(sys.argv[1]))
