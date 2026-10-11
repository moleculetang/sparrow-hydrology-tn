"""One-shot SSH event return, repaired qualification, then finite prefit stages.

No timer job and no blind training dispatch. Exact scientific source activation
remains the responsibility of the existing remote repair controller.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import write_json,sha
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def run(args,**kwargs):
    r=subprocess.run(args,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    if r.returncode:raise RuntimeError('QUALIFICATION_SUBPROCESS_FAILED '+r.stdout[-5000:]+' '+r.stderr[-5000:])
    return r.stdout


def main():
    if os.name!='nt':raise RuntimeError('LOCAL_ONE_SHOT_QUALIFICATION_MANAGER_ONLY')
    folder=ROOT/'outputs/transition_qualification_wait';folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    state=dict(owner=owner,phase='single_remote_requalification_completion_wait',scientific_fits_dispatched=0,scheduled_task_created=False)
    write_json(folder/'owner.json',owner);write_json(folder/'status.json',state)
    try:
        code=f'''
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
owner=json.loads((root/'outputs/transition_requalification_workstation/owner.json').read_text())
while registry.identity(owner['pid'])==owner['created']:time.sleep(20)
status=root/'outputs/transition_requalification_workstation/status.json'
r=json.loads(status.read_text())
if r['phase']!='full_history_new_adjoint_returned_FD_merge_pending':raise RuntimeError('REMOTE_REQUALIFICATION_FAILED '+r['phase'])
archive=root/'transfer/completed_transition_requalification_v1.tar.gz'
if not archive.exists():
    with tarfile.open(archive,'w:gz') as tar:
        for jid in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):
            p=root/'outputs/jobs'/('transition_requalification_'+jid);tar.add(p,arcname=p.relative_to(root).as_posix())
        tar.add(status,arcname=status.relative_to(root).as_posix())
h=hashlib.sha256()
with archive.open('rb') as stream:
    for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
print(json.dumps(dict(archive=str(archive),sha256=h.hexdigest(),phase=r['phase'])))
'''
        returned=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code).strip())
        state.update(phase='verified_transition_return',returned=returned);write_json(folder/'status.json',state)
        archive=ROOT/'deployment'/Path(returned['archive']).name
        if not archive.exists():run(['scp',*OPTIONS,'pc@10.5.8.38:'+returned['archive'],str(archive)])
        if sha(archive)!=returned['sha256']:raise RuntimeError('TRANSITION_RETURN_HASH_CONFLICT')
        run([sys.executable,'-B',str(ROOT/'deployment/import_diagnostics.py'),str(archive),returned['sha256'],'--kind','transition'])
        # The original coordinate waiter imports the old slices before stopping
        # on now-stale initial evidence. Never restart its scientific workers.
        marker=ROOT/'outputs/qualification_wait/owner.json'
        if marker.exists():
            old=json.loads(marker.read_text());
            while registry.identity(old['pid'])==old['created']:time.sleep(20)
        local=json.loads((ROOT/'outputs/local_compatibility/status.json').read_text())
        for entry in (local.get('owner'),local.get('child')):
            if entry:
                while registry.identity(entry['pid'])==entry['created']:time.sleep(20)
        local=json.loads((ROOT/'outputs/local_compatibility/status.json').read_text())
        if local.get('exit_code')!=0:raise RuntimeError('CURRENT_LOCAL_COMPATIBILITY_FAILED')
        for script in ('requalify_transition_derivatives.py','compare_transition_hosts.py','review_production_code.py','seal_gate.py'):
            state['phase']='review_'+script;write_json(folder/'status.json',state)
            output=run([sys.executable,'-B',str(ROOT/'deployment'/script)])
            (folder/(script+'.log')).write_text(output,encoding='utf8')
        # Adopt an already live finite-stage owner; otherwise resume only the
        # safe failed/absent helper after the genuine current gate is present.
        pipeline=ROOT/'outputs/gate_pipeline/status.json'
        active=None
        if pipeline.exists():
            prior=json.loads(pipeline.read_text());entry=prior.get('owner')
            if entry and registry.identity(entry['pid'])==entry['created']:active=entry
        state['phase']='finite_replays_and_measurements_after_repaired_gate';write_json(folder/'status.json',state)
        if active:
            while registry.identity(active['pid'])==active['created']:time.sleep(20)
            result=json.loads(pipeline.read_text())
            if result['phase']!='replays_complete_measured_campaign_ownership_review_pending':raise RuntimeError('EXISTING_FINITE_PIPELINE_FAILED')
        else:run([sys.executable,'-B',str(ROOT/'deployment/continue_after_gate.py')])
        # Also handles the older live pipeline version which did not call this.
        run([sys.executable,'-B',str(ROOT/'deployment/compare_stage_measurements.py')])
        state['phase']='prefit_stages_complete_campaign_ownership_review_pending';write_json(folder/'status.json',state)
        return 0
    except Exception as exc:
        state.update(phase='failure_requires_review',error=str(exc));write_json(folder/'status.json',state);raise


if __name__=='__main__':sys.exit(main())
