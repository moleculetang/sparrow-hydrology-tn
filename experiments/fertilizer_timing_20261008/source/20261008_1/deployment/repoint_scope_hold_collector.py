"""Rebind terminal transport to a proven new dispatcher after old wait exits."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from host_resources import registry
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main():
    result_folder=ROOT/'outputs/maize_scope_revision_candidate/collector_repoint_v3'
    result_folder.mkdir(exist_ok=True)
    if (result_folder/'complete.json').exists():raise RuntimeError('EXISTING_COLLECTOR_REPOINT_NO_RETRY')
    status_path=ROOT/'outputs/campaign_terminal_return/status.json'
    old=json.loads(status_path.read_text(encoding='utf8'))
    if old['owner']!={'pid':81360,'created':'134360066631580682'}:
        raise RuntimeError('PINNED_OLD_COLLECTOR_OWNER_CHANGED')
    if registry.identity(old['owner']['pid'])==old['owner']['created']:
        raise RuntimeError('OLD_COLLECTOR_STILL_LIVE_NO_DUPLICATE_SSH_WAIT')
    if old['phase']!='connection_or_transport_requires_review' or old.get('transport_passed'):
        raise RuntimeError('OLD_COLLECTOR_TERMINAL_REVIEW_REQUIRED')
    code=f'''
import hashlib,json,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
handoff=json.loads((root/'outputs/contract_hold_handoff_v3/complete.json').read_text())
current=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
owner=handoff['new_owner']
if not handoff['passed'] or current['owner']!=owner or reg.identity(owner['pid'])!=owner['created']:
    raise RuntimeError('ACTUAL_NEW_DISPATCHER_IDENTITY_REQUIRED')
if not set(handoff['held_jobs'])<={{r['job'] for r in current['reviews']}}:
    raise RuntimeError('CURRENT_K_HOLD_NOT_PRESENT')
files={{}}
for rel in ('outputs/contract_hold_handoff_v3/complete.json','outputs/contract_hold_handoff_v2/status.json','outputs/contract_hold_native_controls_v3/complete.json'):
    files[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
leases=reg.status(root=root.parent/'.compute_coordination/state')['leases']
print(json.dumps(dict(handoff=handoff,status=current,files=files,memory=reg.memory(),
    leases=leases,all_live=all(reg.identity(x['pid'])==x['created'] for x in current['running']))))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    actual=json.loads(p.stdout)
    if not actual['all_live']:raise RuntimeError('TRANSITIONING_CHILD_REVIEW_BEFORE_REPOINT')
    write_json(result_folder/'actual_remote.json',actual)
    for rel,digest in actual['files'].items():
        target=result_folder/rel.replace('/','_')
        if not target.exists():
            p=subprocess.run(['scp',*OPTIONS,'pc@10.5.8.38:'+REMOTE_ROOT+'/'+rel,str(target)],capture_output=True,timeout=60)
            if p.returncode:raise RuntimeError('HANDOFF_RECEIPT_TRANSFER_FAILED')
        if sha(target)!=digest:raise RuntimeError('HANDOFF_RECEIPT_HASH_MISMATCH')
    admission_path=ROOT/'outputs/campaign_admission/status.json'
    admission=json.loads(admission_path.read_text(encoding='utf8'))
    handoff=actual['handoff']
    if admission['remote_owner']!=handoff['old_owner'] or sha(ROOT/'evidence/formal_gate.json')!=handoff['gate_sha256'] or sha(ROOT/'deployment/path_ownership.json')!=handoff['ownership_sha256']:
        raise RuntimeError('ORIGINAL_ADMISSION_HANDOFF_MISMATCH')
    preserve=result_folder/'pre_repoint';preserve.mkdir(exist_ok=True)
    paths=[admission_path,status_path,ROOT/'outputs/campaign_terminal_return/owner.json',
           ROOT/'outputs/campaign_event_wait/workstation_event.json']
    for source in paths:
        if not source.exists():continue
        target=preserve/(source.parent.name+'_'+source.name)
        if target.exists():raise RuntimeError('PRESERVATION_CONFLICT_NO_OVERWRITE')
        shutil.copyfile(source,target)
        if sha(target)!=sha(source):raise RuntimeError('PRESERVATION_HASH_MISMATCH')
    admission['remote_owner']=handoff['new_owner']
    admission['original_epoch_K_hold']=dict(held_jobs=handoff['held_jobs'],
        actual_handoff_receipt=str(result_folder/'outputs_contract_hold_handoff_v3_complete.json'),
        scientific_gate_changed=False,ownership_changed=False,
        approved_expanded_K_science_gate_passed=False)
    write_json(admission_path,admission)
    # This is a genuine dispatcher handoff event, not a fabricated fit exit.
    write_json(ROOT/'outputs/campaign_event_wait/workstation_event.json',
        dict(state=actual['status'],event='verified_original_epoch_dispatcher_handoff',
             handoff_receipt_sha256=actual['files']['outputs/contract_hold_handoff_v3/complete.json'],
             fit_completion_asserted=False,scheduled_task_created=False))
    env=os.environ.copy();env['PYTHONUTF8']='1'
    with (result_folder/'collector.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'deployment/await_campaign_terminal_return.py')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    result=dict(passed=True,old_collector_native_exited=True,old_status_preserved=True,
                new_collector=dict(pid=child.pid,created=registry.identity(child.pid)),
                remote_controller=handoff['new_owner'],scientific_sources_changed=False,
                new_training_dispatched=False,scheduled_task_created=False)
    write_json(result_folder/'complete.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
