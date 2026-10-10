"""One persistent SSH terminal wait, private verified return, no fit dispatch.

Transport remains separate from scientific promotion and numerical sufficiency.
An insufficient terminal phase returns evidence for review, not an NSE result.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from host_resources import registry
from controller_lock import acquire
from d29_platform.runtime import write_json,sha
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from campaign_return import import_archive


def run(args,**kwargs):
    p=subprocess.run(args,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    if p.returncode:raise RuntimeError('TERMINAL_RETURN_FAILURE_NO_REDISPATCH '+p.stderr[-2500:]+p.stdout[-2500:])
    return p.stdout


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_RETURN_COORDINATOR_REQUIRED')
    folder=ROOT/'outputs/campaign_terminal_return';folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('EXISTING_TERMINAL_RETURN_REQUIRES_REVIEW_NOT_OVERWRITE')
    admission=json.loads((ROOT/'outputs/campaign_admission/status.json').read_text())
    ownership=json.loads((ROOT/'deployment/path_ownership.json').read_text())
    gate=sha(ROOT/'evidence/formal_gate.json');ownership_hash=sha(ROOT/'deployment/path_ownership.json')
    if gate!=admission['formal_gate_sha256'] or ownership_hash!=admission['ownership_sha256']:
        raise RuntimeError('CURRENT_ADMITTED_IDENTITIES_REQUIRED')
    jobs=sorted(k for k,v in ownership['owners'].items() if v=='workstation')
    if len(jobs)!=42:raise RuntimeError('EXACT_REMOTE42OWNER_PLAN_REQUIRED')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    handoff=ROOT/'outputs/campaign_event_wait/workstation_event.json'
    write_json(folder/'status.json',dict(owner=owner,
        phase='existing_single_ssh_event_waiter_handoff_pending',
        remote_controller=admission['remote_owner'],scheduled_task_created=False,
        no_training_dispatch=True,no_evaluation_outcomes_read=True))
    # The first-stage observer owns the current long SSH connection. Wait for
    # its actual remote result, so this follower does not duplicate that wait.
    while not handoff.exists():time.sleep(20)
    first=json.loads(handoff.read_text())
    if first['state']['owner']!=admission['remote_owner']:
        raise RuntimeError('FIRST_EVENT_REMOTE_OWNER_CHANGED')
    module=ROOT/'deployment/campaign_return.py';module_hash=sha(module)
    stage=REMOTE_ROOT+'/.incoming_terminal_transport_'+module_hash[:16]
    # This helper is outside the scientific seal and cannot dispatch a worker.
    prep=f'''
import pathlib
p=pathlib.Path({stage!r});p.mkdir(exist_ok=True)
'''
    run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=prep)
    run(['scp',*OPTIONS,str(module),'pc@10.5.8.38:'+stage+'/campaign_return.py'])
    state=dict(owner=owner,phase='single_connection_remote_campaign_terminal_wait',
        remote_controller=admission['remote_owner'],formal_gate_sha256=gate,ownership_sha256=ownership_hash,
        helper_sha256=module_hash,no_training_dispatch=True,no_evaluation_outcomes_read=True,
        live_checkpoint_transfer=False,scheduled_task_created=False)
    write_json(folder/'status.json',state)
    code=f'''
import hashlib,json,pathlib,sys,time
root=pathlib.Path({REMOTE_ROOT!r});stage=pathlib.Path({stage!r})
if hashlib.sha256((stage/'campaign_return.py').read_bytes()).hexdigest()!={module_hash!r}:
    raise RuntimeError('OPERATIONAL_TRANSPORT_HELPER_HASH_MISMATCH')
sys.path[:0]=[str(stage),str(root.parent/'.compute_coordination')]
import resource_registry as registry
from campaign_return import terminal_manifest,make_archive
state_root=root.parent/'.compute_coordination/state';owner={admission['remote_owner']!r}
current=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
if current!=owner:raise RuntimeError('PINNED_CONTROLLER_IDENTITY_ALREADY_CHANGED')
while registry.identity(owner['pid'])==owner['created']:time.sleep(20)
while True:
    lease=registry.acquire('20261008_1','transport/terminal_result_pack',threads=1,peak_bytes=500000000,root=state_root)
    if lease.get('granted'):break
    time.sleep(20)
try:
    registry.affinity(lease['cpus'])
    manifest=terminal_manifest(root,registry,state_root,owner)
    if manifest['formal_gate_sha256']!={gate!r} or manifest['ownership_sha256']!={ownership_hash!r}:
        raise RuntimeError('CAMPAIGN_TERMINAL_TRANSFER_IDENTITY_CHANGED')
    result=make_archive(root,manifest,root/'transfer')
    # Recheck stopped ownership and the exact file set after packaging.
    if terminal_manifest(root,registry,state_root,owner)!=manifest:
        raise RuntimeError('TERMINAL_MANIFEST_CHANGED_DURING_PACK')
    result.update(terminal_phase=manifest['terminal_phase'],owned_paths=manifest['owned_paths'],
        raw_source_or_observation_tables_packaged=False,snapshot_is_convergence_certificate=False)
    print(json.dumps(result))
finally:registry.release(lease['token'],root=state_root)
'''
    try:
        returned=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code))
        state.update(phase='remote_terminal_bytes_returning',returned=returned);write_json(folder/'status.json',state)
        archive=ROOT/'deployment'/Path(returned['archive']).name
        if archive.exists() and sha(archive)!=returned['sha256']:raise RuntimeError('LOCAL_ARCHIVE_BYTE_CONFLICT')
        if not archive.exists():
            temporary=archive.with_name(archive.name+'.transfer_'+str(os.getpid())+'.tmp')
            run(['scp',*OPTIONS,'pc@10.5.8.38:'+returned['archive'],str(temporary)])
            if sha(temporary)!=returned['sha256']:raise RuntimeError('RETURNED_ARCHIVE_SHA256_MISMATCH')
            os.replace(temporary,archive)
        target=ROOT/'outputs'/('imported_workstation_campaign_'+returned['manifest_sha256'][:16])
        imported=import_archive(archive,returned['sha256'],target,jobs,gate,ownership_hash)
        state.update(phase='verified_private_terminal_snapshot_requires_scientific_review',transport_passed=True,
                     receipt=imported,all44numerically_sufficient_certified=False)
        write_json(folder/'complete.json',state);write_json(folder/'status.json',state)
        return 0
    except Exception as exc:
        state.update(phase='connection_or_transport_requires_review',transport_passed=False,error=str(exc))
        write_json(folder/'status.json',state)
        raise


if __name__=='__main__':sys.exit(main())
