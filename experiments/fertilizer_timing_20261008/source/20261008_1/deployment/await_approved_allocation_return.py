"""Single SSH event wait for independently reviewed allocation artifacts.

The scientific result import stays immutable. This later import receives the
full finished job directories, including the 128 annual allocation files per
path. No science or artifact worker is launched here, and no ACK is invented.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from controller_lock import acquire
from d29_platform.runtime import sha,write_json
from approved_scope_policy import load_qualified
from approved_scope_return import import_archive
from approved_parent_lineage import current
from host_resources import registry
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def remote_code(scope, science_owner, artifact_owner, ownership_hash, source, helper_hash, artifact_folder='outputs/approved_frozen_allocations_workstation'):
    return f'''
import hashlib,json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root/'deployment'),str(root.parent/'.compute_coordination')]
import resource_registry as reg
scope={scope!r};science_owner={science_owner!r};artifact_owner={artifact_owner!r}
ARTIFACT_FOLDER={artifact_folder!r}
f=root/ARTIFACT_FOLDER
status=f/'status.json'
s=json.loads(status.read_text())
if s['owner']!=artifact_owner or s.get('approved_scope')!=scope:
 raise RuntimeError('EXACT_ARTIFACT_CONTROLLER_AND_SCOPE_REQUIRED')
if s.get('source_sha256')!={helper_hash!r}:
 raise RuntimeError('EXACT_ARTIFACT_CONTROLLER_SOURCE_REQUIRED')
while reg.identity(artifact_owner['pid'])==artifact_owner['created']:time.sleep(20)
raw=status.read_bytes();s=json.loads(raw)
if s['owner']!=artifact_owner or s['phase']!='all_owned_allocations_independently_reviewed' or s.get('running') is not None:
 raise RuntimeError('ACTUAL_ALLOCATION_TERMINAL_REVIEW_INCOMPLETE')
if s.get('approved_scope')!=scope or s.get('source_sha256')!={helper_hash!r}:
 raise RuntimeError('ACTUAL_ALLOCATION_TERMINAL_IDENTITY_CHANGED')
ns={{'__name__':'approved_allocation_terminal_transport'}}
exec(compile({source!r},'approved_allocation_terminal_transport','exec'),ns)
state_root=root.parent/'.compute_coordination/state'
while True:
 lease=reg.acquire('20261008_1','transport/approved_allocation_terminal_pack',threads=1,peak_bytes=500000000,root=state_root)
 if lease.get('granted'):break
 time.sleep(20)
try:
 reg.affinity(lease['cpus'])
 manifest=ns['terminal_manifest'](root,reg,state_root,science_owner,scope)
 if manifest['ownership_sha256']!={ownership_hash!r} or manifest['terminal_phase']!='all_owned_paths_frozen':
  raise RuntimeError('EXACT_SCIENTIFIC_TERMINAL_REQUIRED')
 jobs=manifest['owned_paths']
 if len(jobs) not in (39,42) or sorted(s['owned_paths'])!=jobs or sorted(s['completed'])!=jobs or s.get('blocked'):
  raise RuntimeError('EXACT_APPROVED_REMOTE_REVIEWED_ALLOCATIONS_REQUIRED')
 from approved_scope_policy import route
 reviews={{}}
 for jid in jobs:
  science=route(root,jid,scope);folder=science/'outputs/jobs'/jid;directory=folder/'input_allocation'
  allocation=json.loads((directory/'complete.json').read_text());review=json.loads((directory/'independent_artifact_review.json').read_text())
  freeze=json.loads((folder/'prediction_freeze.json').read_text())
  expected={{f'{{kind}}_{{year}}.parquet' for kind in ['budgets','profiles'] for year in range(1961,2025)}}
  if (allocation.get('passed') is not True or allocation.get('job')!=jid or allocation.get('source_multiplier')!=1.
      or allocation.get('formal_gate_sha256')!=scope['gates'][jid]
      or allocation.get('parameter_sha256')!=freeze['parameter_sha256']
      or set(allocation.get('artifact_sha256',{{}}))!=expected):
   raise RuntimeError('EXACT_FROZEN_FULL_HISTORY_ALLOCATION_REQUIRED '+jid)
  if (review.get('passed') is not True or review.get('job')!=jid or review.get('years')!=64
      or review.get('reaches')!=230 or review.get('crops')!=21
      or review.get('formal_gate_sha256')!=scope['gates'][jid]
      or review.get('parameter_sha256')!=freeze['parameter_sha256']
      or review.get('allocation_receipt_sha256')!=ns['digest'](directory/'complete.json')
      or review.get('verifier_sha256')!=ns['digest'](science/'analysis/verify_timing_allocation_artifacts.py')):
   raise RuntimeError('ACTUAL_INDEPENDENT_ARTIFACT_READBACK_REQUIRED '+jid)
  for name,digest in allocation['artifact_sha256'].items():
   if ns['digest'](directory/name)!=digest:raise RuntimeError('REVIEWED_ALLOCATION_BYTES_CHANGED '+jid+'/'+name)
  reviews[jid]=dict(allocation_sha256=ns['digest'](directory/'complete.json'),review_sha256=ns['digest'](directory/'independent_artifact_review.json'))
 if status.read_bytes()!=raw:raise RuntimeError('ALLOCATION_CONTROLLER_CHANGED_DURING_PACK_PREPARATION')
 owner_key=hashlib.sha256(json.dumps(artifact_owner,sort_keys=True).encode()).hexdigest()[:16]
 packfile=root/'transfer'/('approved_allocation_pack_'+owner_key+'.json')
 manifest_hash=hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode()).hexdigest()
 if packfile.exists():
  packed=json.loads(packfile.read_text())
  if packed['manifest_sha256']!=manifest_hash or ns['digest'](Path(packed['archive']))!=packed['sha256']:
   raise RuntimeError('EXISTING_ALLOCATION_ARCHIVE_CHANGED_NO_REPACK')
 else:
  packed=ns['make_archive'](root,manifest,scope,root/'transfer')
  with packfile.open('x') as stream:json.dump(packed,stream,indent=2)
 if ns['terminal_manifest'](root,reg,state_root,science_owner,scope)!=manifest or status.read_bytes()!=raw:
  raise RuntimeError('ACTUAL_TERMINAL_ARTIFACTS_CHANGED_DURING_PACK')
 print(json.dumps(dict(packed=packed,artifact_owner=artifact_owner,science_owner=science_owner,
  artifact_controller_status=s,artifact_controller_status_sha256=hashlib.sha256(raw).hexdigest(),reviews=reviews)))
finally:reg.release(lease['token'],root=state_root)
'''


def call(args,**kwargs):
    p=subprocess.run(args,text=True,encoding='utf8',capture_output=True,**kwargs)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    return p.stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-launch-receipt',type=Path,required=True)
    args=p.parse_args()
    scope=load_qualified(ROOT)
    handoff=current(ROOT)
    launch=json.loads(args.artifact_launch_receipt.read_text())
    if not handoff.get('passed') or not launch.get('passed') or launch.get('scope')!=scope:
        raise RuntimeError('ACTUAL_VERIFIED_SCIENCE_AND_ARTIFACT_LAUNCH_REQUIRED')
    owner=launch['artifact_owner'];science_owner=handoff['new_owner']
    if launch.get('science_owner')!=science_owner:
        raise RuntimeError('EXACT_CURRENT_SCIENTIFIC_PARENT_REQUIRED')
    declaration=ROOT/'outputs/local_repartition_operations_v1/complete.json'
    repartition=declaration.exists()
    helper=ROOT/('deployment/repartition_allocation_controller_v1.py' if repartition else 'deployment/approved_frozen_allocation_controller.py')
    artifact_folder='outputs/approved_frozen_allocations_repartition_workstation' if repartition else 'outputs/approved_frozen_allocations_workstation'
    if launch.get('controller_sha256')!=sha(helper):
        raise RuntimeError('EXACT_LAUNCHED_ARTIFACT_CONTROLLER_SOURCE_REQUIRED')
    key=hashlib.sha256(json.dumps(owner,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'outputs/approved_allocation_terminal_return'/key;folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('ALLOCATION_RETURN_ALREADY_IMPORTED_REVIEW_NO_RESTART')
    ownership_path=ROOT/'deployment/path_ownership.json'
    from repartition_policy_v1 import read_ownership
    ownership=read_ownership(ROOT)
    jobs=sorted(j for j,h in ownership['owners'].items() if h=='workstation')
    source=(ROOT/'deployment/approved_scope_return.py').read_text(encoding='utf8')
    state=dict(owner=dict(pid=os.getpid(),created=registry.identity(os.getpid())),
      phase='single_connection_allocation_terminal_wait',scope=scope,artifact_owner=owner,
      science_owner=science_owner,artifact_launch_receipt_sha256=sha(args.artifact_launch_receipt),
      ownership_sha256=sha(ownership_path),scheduled_task_created=False,evaluation_labels_read=False,
      scientific_dispatch=False,scientific_signals=0)
    write_json(folder/'status.json',state)
    try:
        actual=json.loads(call(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
          input=remote_code(scope,science_owner,owner,sha(ownership_path),source,sha(helper),artifact_folder)))
        packed=actual['packed'];name=Path(packed['archive']).name
        if packed['archive']!=REMOTE_ROOT+'/transfer/'+name:
            raise RuntimeError('EXACT_PRIVATE_REMOTE_ARCHIVE_REQUIRED')
        archive=folder/name
        if archive.exists() and sha(archive)!=packed['sha256']:raise RuntimeError('LOCAL_ALLOCATION_ARCHIVE_CONFLICT')
        if not archive.exists():
            temporary=archive.with_name(name+'.'+str(os.getpid())+'.transfer')
            call(['scp',*OPTIONS,'pc@10.5.8.38:'+packed['archive'],str(temporary)])
            if sha(temporary)!=packed['sha256']:raise RuntimeError('ACTUAL_ALLOCATION_ARCHIVE_CHANGED')
            os.replace(temporary,archive)
        target=ROOT/'outputs'/('imported_workstation_approved_allocations_'+packed['manifest_sha256'][:20])
        imported=import_archive(archive,packed['sha256'],target,jobs,scope,sha(ownership_path))
        for jid,proof in actual['reviews'].items():
            f=target/'outputs/jobs'/jid/'input_allocation'
            if sha(f/'complete.json')!=proof['allocation_sha256'] or sha(f/'independent_artifact_review.json')!=proof['review_sha256']:
                raise RuntimeError('ACTUAL_RETURNED_REVIEW_IDENTITY_CHANGED')
        state.update(phase='verified_allocation_terminal_import_requires_final44assembly',
          transport_passed=True,imported=imported,actual=actual,archive=str(archive),
          all44delivery_certified=False)
        write_json(folder/'complete.json',state);write_json(folder/'status.json',state)
        print(json.dumps(dict(transport_passed=True,target=str(target),reviewed_remote_paths=len(actual['reviews']),
          scientific_dispatch=False,all44delivery_certified=False)))
    except BaseException as exc:
        state.update(phase='allocation_observation_or_transport_review_required',error=repr(exc),
          transport_passed=False,remote_scientific_processes_restarted=False)
        write_json(folder/'status.json',state);raise


if __name__=='__main__':main()
