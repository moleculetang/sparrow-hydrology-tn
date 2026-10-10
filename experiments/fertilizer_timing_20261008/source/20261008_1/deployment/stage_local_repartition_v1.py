"""Prepare exact 39/5 operation, transfer code, run tiny native controls only."""
import json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from approved_parent_lineage import current
from approved_scope_policy import load_qualified
from repartition_policy_v1 import propose,IDS
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON

FILES=['deployment/repartition_policy_v1.py','deployment/repartition_scope_controller_v1.py',
 'deployment/repartition_allocation_controller_v1.py','deployment/handoff_local_repartition_v1.py',
 'deployment/test_repartition_native_v1.py','deployment/test_repartition_policy_v1.py',
 'deployment/test_repartition_allocation_controller_v1.py','deployment/approved_parent_lineage.py',
 'deployment/await_approved_campaign_return.py','deployment/await_approved_allocation_return.py',
 'analysis/assemble_approved_delivery.py','analysis/assemble_approved_allocation_delivery.py',
 'evidence/repartition_policy_controls_v1.json','evidence/repartition_allocation_controller_controls_v1.json',
 'outputs/windows_capacity_probe_v1/independent_readback.json',
 'deployment/windows_capacity_probe_v1.py','deployment/measure_windows_parallel_targets_v1.py',
 'deployment/verify_windows_capacity_probe_v1.py']


def call(args,**kwargs):
    p=subprocess.run(args,text=True,encoding='utf8',capture_output=True,**kwargs)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    return p.stdout


def remote(code):
    return call(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,timeout=120)


def main():
    folder=ROOT/'outputs/local_repartition_preparation_v1';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():raise RuntimeError('PREPARATION_COMPLETE_REVIEW_NO_REPEAT')
    declaration=ROOT/'deployment/local_repartition_v1.json'
    if declaration.exists():raise RuntimeError('EXISTING_DECLARATION_REVIEW_NO_RECREATE')
    capacity_path=ROOT/'outputs/windows_capacity_probe_v1/independent_readback.json'
    capacity=json.loads(capacity_path.read_text());ownership_path=ROOT/'deployment/path_ownership.json'
    before=json.loads(ownership_path.read_text());jobs=json.loads((ROOT/'config/jobs.json').read_text())
    after=propose(before,jobs,capacity);expected=current(ROOT)['new_owner'];scope=load_qualified(ROOT)
    code=f'''
import json,os,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root/'scripts')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry,registry_root
s=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
a=json.loads((root/'outputs/approved_frozen_allocations_workstation/status.json').read_text())
if s['owner']!={expected!r} or registry.identity(s['owner']['pid'])!=s['owner']['created']:raise RuntimeError('EXACT_CURRENT_PARENT_REQUIRED')
if a['running'] is not None or registry.identity(a['owner']['pid'])!=a['owner']['created']:raise RuntimeError('IDLE_LIVE_ARTIFACT_PARENT_REQUIRED')
for jid in {list(IDS)!r}:
 if (root/'outputs/jobs'/jid).exists():raise RuntimeError('CANDIDATE_ALREADY_STARTED '+jid)
print(json.dumps(dict(old_artifact_owner=a['owner'],science_owner=s['owner'],scope=s['approved_scope'])))
'''
    state=json.loads(remote(code))
    if state['scope']!=scope:raise RuntimeError('BOTH_HOST_QUALIFIED_SCOPE_REQUIRED')
    bp=ROOT/'deployment/local_repartition_v1_before.json';ap=ROOT/'deployment/local_repartition_v1_after.json'
    if bp.exists() or ap.exists():raise RuntimeError('EXISTING_PARTITION_SNAPSHOT_REVIEW')
    bp.write_bytes(ownership_path.read_bytes());write_json(ap,after)
    source_files={p:sha(ROOT/p) for p in FILES}
    plan=dict(version=1,candidates=list(IDS),old_owner=expected,old_artifact_owner=state['old_artifact_owner'],
        scope=scope,before_ownership_sha256=sha(bp),after_ownership_sha256=sha(ap),
        capacity_sha256=sha(capacity_path),source_files=source_files,science_paths=44,
        nonK=30,corrected_K=14,workstation_paths=39,local_paths=5,local_max_active=3,
        scientific_signals_allowed=0,started_checkpoint_migration_allowed=False)
    write_json(declaration,plan)
    archive=folder/'operational_sources.tar'
    with tarfile.open(archive,'w') as stream:
        for rel in FILES+['deployment/local_repartition_v1.json','deployment/local_repartition_v1_before.json',
                          'deployment/local_repartition_v1_after.json']:
            p=ROOT/rel
            if p.is_symlink() or p.stat().st_size>2000000:raise RuntimeError('ONLY_BOUNDED_REGULAR_OPERATION_FILES_ALLOWED')
            stream.add(p,arcname=rel,recursive=False)
    destination=REMOTE_ROOT+'/transfer/local_repartition_operational_v1.tar'
    call(['scp',*OPTIONS,str(archive),'pc@10.5.8.38:'+destination],timeout=120)
    code=f'''
import hashlib,json,os,sys,tarfile
from pathlib import Path
root=Path({REMOTE_ROOT!r});archive=Path({destination!r})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(archive)!={sha(archive)!r}:raise RuntimeError('ACTUAL_OPERATION_ARCHIVE_BYTES_CHANGED')
allowed={set(FILES+['deployment/local_repartition_v1.json','deployment/local_repartition_v1_before.json','deployment/local_repartition_v1_after.json'])!r}
with tarfile.open(archive) as stream:
 members=stream.getmembers()
 if len(members)!=len(allowed) or {{m.name for m in members}}!=allowed:raise RuntimeError('EXACT_OPERATION_PACKAGE_REQUIRED')
 for m in members:
  p=root/m.name
  if not m.isfile() or not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise RuntimeError('NO_LINK_OR_EXTERNAL_OPERATION_FILES')
  data=stream.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  temporary=p.with_name(p.name+'.stage.tmp');temporary.write_bytes(data);os.replace(temporary,p)
for rel,h in {source_files!r}.items():
 if sha(root/rel)!=h:raise RuntimeError('STAGED_BYTES_MISMATCH '+rel)
print(json.dumps(dict(passed=True,files=len(allowed),ownership_changed=False)))
'''
    stage=json.loads(remote(code));write_json(folder/'staged.json',stage)
    fixture=call(['ssh',*OPTIONS,'pc@10.5.8.38',
        'D29_RESOURCE_STATE=/home/pc/TWY/SPARROW/Test/.compute_coordination/state '+REMOTE_PYTHON+
        ' -X utf8 -B '+REMOTE_ROOT+'/deployment/test_repartition_native_v1.py '+REMOTE_ROOT],timeout=120)
    value=json.loads(fixture)
    if not value.get('passed'):raise RuntimeError('ACTUAL_NATIVE_FIXTURE_FAILED')
    write_json(ROOT/'evidence/repartition_native_controls_v1.json',value)
    preflight=call(['ssh',*OPTIONS,'pc@10.5.8.38',
        'D29_RESOURCE_STATE=/home/pc/TWY/SPARROW/Test/.compute_coordination/state '+REMOTE_PYTHON+
        ' -X utf8 -B '+REMOTE_ROOT+'/deployment/handoff_local_repartition_v1.py '+REMOTE_ROOT+' --preflight-only'],timeout=120)
    check=json.loads(preflight)
    if not check.get('passed'):raise RuntimeError('ACTUAL_NATIVE_PREFLIGHT_FAILED')
    write_json(folder/'complete.json',dict(passed=True,preflight_only=True,staged=stage,native=value,
        preflight=check,declaration_sha256=sha(declaration),source_files=source_files,
        ownership_changed=False,scientific_signals=0,science_paths_dispatched=0))
    print(json.dumps(dict(passed=True,preflight_only=True,current_children=check['current_children'],
                         candidates=check['candidates'],ownership_changed=False)))


if __name__=='__main__':main()
