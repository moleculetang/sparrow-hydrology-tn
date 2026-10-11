"""One SSH event wait and private verified two-contract terminal import.

The old single-gate collector is retained as historical evidence. This new
collector binds the actual replacement parent and preserves every import.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'deployment')]
from host_resources import registry
from controller_lock import acquire
from d29_platform.runtime import sha, write_json
from approved_scope_return import import_archive
from approved_parent_lineage import current
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def call(args, **kwargs):
    run = subprocess.run(args, text=True, encoding='utf8', capture_output=True, **kwargs)
    if run.returncode:
        raise RuntimeError(run.stderr+run.stdout)
    return run.stdout


def main():
    if os.name != 'nt':
        raise RuntimeError('WINDOWS_PRIVATE_RETURN_COORDINATOR_REQUIRED')
    handoff = current(ROOT)
    if not handoff['passed'] or handoff['scientific_signals'] != 0:
        raise RuntimeError('ACTUAL_VERIFIED_PARENT_HANDOFF_REQUIRED')
    legacy_path = ROOT/'outputs/campaign_terminal_return/status.json'
    legacy = json.loads(legacy_path.read_text(encoding='utf8'))
    if legacy.get('remote_controller') != handoff['old_owner']:
        raise RuntimeError('EXACT_LEGACY_OBSERVER_PARENT_REQUIRED')
    if registry.identity(legacy['owner']['pid']) == legacy['owner']['created']:
        raise RuntimeError('OLD_SINGLE_GATE_OBSERVER_STILL_LIVE_DEFER_NEW_WAIT')
    remote_owner = handoff['new_owner']
    scope = handoff['scope']
    ownership_path = ROOT/'deployment/path_ownership.json'
    from repartition_policy_v1 import read_ownership
    ownership = read_ownership(ROOT)
    jobs = sorted(j for j, role in ownership['owners'].items() if role == 'workstation')
    if len(jobs) not in (39,42):
        raise RuntimeError('EXACT_APPROVED_REMOTE_PARTITION_REQUIRED')
    original_gate = sha(ROOT/'evidence/formal_gate.json')
    corrected_gate = sha(ROOT.with_name('20261008_1_maize_scope_v2')/'evidence/formal_gate.json')
    for jid, epoch in scope['epochs'].items():
        if scope['gates'][jid] != (corrected_gate if epoch == 'approved_maize_K' else original_gate):
            raise RuntimeError('CURRENT_TWO_GATE_SCOPE_REQUIRED')
    owner_key = hashlib.sha256(json.dumps(remote_owner, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT/'outputs/approved_scope_terminal_return'/owner_key
    folder.mkdir(parents=True, exist_ok=True)
    lifetime = acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():
        raise RuntimeError('THIS_PARENT_TERMINAL_RETURN_ALREADY_IMPORTED_REVIEW')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    ownership_hash = sha(ownership_path)
    helper = ROOT/'deployment/approved_scope_return.py'
    source = helper.read_text(encoding='utf8')
    controls = json.loads((ROOT/'evidence/approved_scope_return_controls.json').read_text(encoding='utf8'))
    if not controls['passed'] or controls['module_sha256'] != sha(helper):
        raise RuntimeError('CURRENT_TWO_CONTRACT_TRANSPORT_CONTROLS_REQUIRED')
    state = dict(owner=owner, remote_owner=remote_owner, scope=scope,
                 legacy_observer_owner=legacy['owner'], legacy_observer_status_sha256=sha(legacy_path),
                 ownership_sha256=ownership_hash, helper_sha256=sha(helper),
                 phase='single_connection_two_contract_terminal_wait',
                 scheduled_task_created=False, scientific_dispatch=False,
                 evaluation_labels_read=False)
    write_json(folder/'owner.json', owner)
    write_json(folder/'status.json', state)
    code = f'''
import hashlib,json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root/'deployment'),str(root.parent/'.compute_coordination')]
import resource_registry as reg
scope={scope!r};owner={remote_owner!r};ownership_hash={ownership_hash!r}
ns={{'__name__':'approved_scope_terminal_transport'}}
exec(compile({source!r},'approved_scope_terminal_transport','exec'),ns)
current=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
if current!=owner:raise RuntimeError('ACTUAL_REPLACEMENT_PARENT_IDENTITY_CHANGED')
while reg.identity(owner['pid'])==owner['created']:time.sleep(20)
state_root=root.parent/'.compute_coordination/state'
while True:
 lease=reg.acquire('20261008_1','transport/approved_scope_terminal_pack',threads=1,peak_bytes=500000000,root=state_root)
 if lease.get('granted'):break
 time.sleep(20)
try:
 reg.affinity(lease['cpus'])
 manifest=ns['terminal_manifest'](root,reg,state_root,owner,scope)
 if manifest['ownership_sha256']!=ownership_hash:raise RuntimeError('CURRENT_REMOTE_OWNERSHIP_CHANGED')
 packfile=root/'transfer'/('approved_scope_pack_{owner_key}.json')
 if packfile.exists():
  packed=json.loads(packfile.read_text())
  if packed['manifest_sha256']!=hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode('utf8')).hexdigest():
   raise RuntimeError('EXISTING_PACK_MANIFEST_CHANGED_NO_REPACK')
  if ns['digest'](Path(packed['archive']))!=packed['sha256']:raise RuntimeError('EXISTING_PACK_BYTES_CHANGED')
 else:
  packed=ns['make_archive'](root,manifest,scope,root/'transfer')
  with packfile.open('x') as stream:json.dump(packed,stream,indent=2)
 if ns['terminal_manifest'](root,reg,state_root,owner,scope)!=manifest:
  raise RuntimeError('TERMINAL_RESULT_CHANGED_DURING_PACK')
 print(json.dumps(packed))
finally:reg.release(lease['token'],root=state_root)
'''
    try:
        returned = json.loads(call(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'], input=code))
        relative = Path(returned['archive']).name
        if returned['archive'] != REMOTE_ROOT+'/transfer/'+relative:
            raise RuntimeError('EXACT_REMOTE_PRIVATE_TRANSFER_ARCHIVE_REQUIRED')
        archive = folder/relative
        if archive.exists() and sha(archive) != returned['sha256']:
            raise RuntimeError('LOCAL_PRIVATE_ARCHIVE_CONFLICT')
        if not archive.exists():
            temporary = archive.with_name(archive.name+'.'+str(os.getpid())+'.transfer')
            call(['scp', *OPTIONS, 'pc@10.5.8.38:'+returned['archive'], str(temporary)])
            if sha(temporary) != returned['sha256']:
                raise RuntimeError('ACTUAL_RETURNED_ARCHIVE_BYTES_CHANGED')
            os.replace(temporary, archive)
        target = ROOT/'outputs'/('imported_workstation_approved_campaign_'+returned['manifest_sha256'][:20])
        imported = import_archive(archive, returned['sha256'], target, jobs, scope, ownership_hash)
        ack = dict(version=2, transport_passed=True, remote_owner=remote_owner,
                   job_contracts=imported['job_contracts'], ownership_sha256=ownership_hash,
                   archive_relative='transfer/'+relative, archive_sha256=returned['sha256'],
                   manifest_sha256=returned['manifest_sha256'], current_host_snapshot_imported=True,
                   all_paths_numerically_sufficient_certified=False)
        ack_code = f'''
import hashlib,json
from pathlib import Path
root=Path({REMOTE_ROOT!r});ack={ack!r};path=root/'evidence/approved_scope_terminal_import_ack_{owner_key}.json'
if json.loads((root/'outputs/campaign_workstation/owner.json').read_text())!=ack['remote_owner']:
 raise RuntimeError('ACK_REMOTE_PARENT_CHANGED')
archive=root/ack['archive_relative'];h=hashlib.sha256()
with archive.open('rb') as stream:
 for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
if h.hexdigest()!=ack['archive_sha256']:raise RuntimeError('ACK_EXACT_IMPORTED_ARCHIVE_REQUIRED')
if path.exists() and json.loads(path.read_text())!=ack:raise RuntimeError('EXISTING_TWO_CONTRACT_ACK_CONFLICT')
if not path.exists():
 with path.open('x') as stream:json.dump(ack,stream,indent=2)
print(json.dumps(ack))
'''
        actual_ack = json.loads(call(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'], input=ack_code))
        if actual_ack != ack:
            raise RuntimeError('ACTUAL_TWO_CONTRACT_ACK_READBACK_CHANGED')
        state.update(phase='verified_two_contract_terminal_import_requires_scientific_review',
                     transport_passed=True, imported=imported, returned=returned, actual_ack=actual_ack,
                     all44qualified_frozen_certified=False)
        write_json(folder/'complete.json', state)
        write_json(folder/'status.json', state)
        print(json.dumps(dict(transport_passed=True, target=str(target),
                              terminal_phase=imported['snapshot_terminal_phase'],
                              numerical_sufficiency_certified=False)))
    except BaseException as exc:
        state.update(phase='observation_or_transport_review_required', error=repr(exc),
                     transport_passed=False, remote_scientific_processes_restarted=False)
        write_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
