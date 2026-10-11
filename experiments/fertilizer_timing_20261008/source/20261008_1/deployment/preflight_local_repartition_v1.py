"""Read-only eligibility evidence for exactly three never-started non-K paths.

Never changes ownership, controller, waiters, scientific state or host leases.
This is preparation, not authority to run under mismatched ownership files.
"""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from train_job import validate_admission
from approved_parent_lineage import current
from approved_scope_policy import load_qualified
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON

IDS=['TIM_S191_C0_V0_s0','TIM_S191_C1_V0_s0','TIM_S191_C2_V0_s0']


def main():
    _,gate=validate_admission()
    ownership_path=ROOT/'deployment/path_ownership.json'
    ownership=json.loads(ownership_path.read_text())
    if ({j for j,h in ownership['owners'].items() if h=='local'}!=
        {'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'} or
            len([h for h in ownership['owners'].values() if h=='workstation'])!=42):
        raise RuntimeError('ORIGINAL42_2_OWNERSHIP_REQUIRED')
    capacity_path=ROOT/'outputs/windows_capacity_probe_v1/independent_readback.json'
    capacity=json.loads(capacity_path.read_text())
    if (capacity.get('passed') is not True or capacity['gate_sha256']!=gate or
            capacity.get('lease_closeout_verified') is not True or
            capacity['maximum_private_commit']>ownership['peak_bytes']['local']):
        raise RuntimeError('ACTUAL_PASSED_CAPACITY_REQUIRED')
    for relative,digest in capacity['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('CAPACITY_SOURCE_CHANGED')
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    selected=[next(j for j in jobs if j['id']==jid) for jid in IDS]
    scope=load_qualified(ROOT)
    if any(j['calendar']=='K' or ownership['owners'][j['id']]!='workstation' or
           scope['epochs'][j['id']]!='original_nonK' for j in selected):
        raise RuntimeError('EXACT_THREE_NONK_WORKSTATION_CANDIDATES_REQUIRED')
    leases=registry.status(root=registry_root())['leases']
    if any((ROOT/'outputs/jobs'/jid).exists() for jid in IDS) or any(
            v['project']=='20261008_1' and v['job'].split('/')[-1] in IDS for v in leases.values()):
        raise RuntimeError('LOCAL_CANDIDATE_NOT_NEVER_STARTED')
    expected=current(ROOT)['new_owner']
    code=f'''
import json,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
from host_resources import registry,registry_root
from d29_platform.runtime import sha
from approved_scope_policy import load_qualified
ownership=json.loads((root/'deployment/path_ownership.json').read_text())
state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if sha(root/'deployment/path_ownership.json')!={sha(ownership_path)!r}:raise RuntimeError('BOTH_HOST_OWNERSHIP_BYTES_REQUIRED')
if state['owner']!={expected!r} or registry.identity({expected['pid']})!={expected['created']!r}:raise RuntimeError('EXACT_CURRENT_DISPATCHER_REQUIRED')
scope=load_qualified(root)
if scope!={scope!r}:raise RuntimeError('BOTH_HOST_SCIENTIFIC_SCOPE_REQUIRED')
leases=registry.status(root=registry_root())['leases']
ids={IDS!r}
for jid in ids:
 if (root/'outputs/jobs'/jid).exists() or jid in {{r['job'] for r in state['running']}}:raise RuntimeError('REMOTE_CANDIDATE_ALREADY_STARTED '+jid)
 if any(v['project']=='20261008_1' and v['job'].split('/')[-1]==jid for v in leases.values()):raise RuntimeError('REMOTE_CANDIDATE_ALREADY_LEASED '+jid)
print(json.dumps(dict(passed=True,parent=state['owner'],candidates=ids,all_candidate_folders_absent=True,
 running=[dict(job=r['job'],pid=r['pid'],created=r['created'],live=registry.identity(r['pid'])==r['created']) for r in state['running']],
 ownership_sha256=sha(root/'deployment/path_ownership.json'))))
'''
    run=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',
        'D29_RESOURCE_STATE=/home/pc/TWY/SPARROW/Test/.compute_coordination/state '+REMOTE_PYTHON+' -B -'],
        input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
    if run.returncode:raise RuntimeError(run.stderr+run.stdout)
    remote=json.loads(run.stdout)
    if not all(r['live'] for r in remote['running']):raise RuntimeError('CURRENT_CHILD_IDENTITY_REVIEW_REQUIRED')
    result=dict(passed=True,preflight_only=True,candidates=IDS,capacity_sha256=sha(capacity_path),
        original_ownership_sha256=sha(ownership_path),remote=remote,gate_sha256=gate,
        source_sha256=sha(Path(__file__)),ownership_changed=False,new_scientific_jobs=0,
        scientific_signals=0,science_checkpoints_read=False,evaluation_labels_read=False,
        next_step='tested two-host ownership transition and re-bound collectors before dispatch')
    out=ROOT/'outputs/local_repartition_preflight_v1';out.mkdir(exist_ok=False)
    write_json(out/'complete.json',result)
    print(json.dumps(dict(passed=True,preflight_only=True,candidates=IDS,current_remote_live=len(remote['running']))))


if __name__=='__main__':main()
