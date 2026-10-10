"""Bind a PID/creation launch receipt to the actual full remote controller identity."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON

def bind(launch,state,live):
    expected=launch['artifact_owner'];actual=state['owner']
    if not live or any(expected[k]!=actual.get(k) for k in ('pid','created')):
        raise RuntimeError('ACTUAL_SAME_NATIVE_LAUNCH_IDENTITY_REQUIRED')
    if actual.get('host')!='pc' or set(actual)!={'pid','created','host'}:
        raise RuntimeError('EXACT_REMOTE_ARTIFACT_CONTROLLER_IDENTITY_SCHEMA_REQUIRED')
    if len(state.get('owned_paths',[]))!=39 or len(set(state['owned_paths']))!=39:
        raise RuntimeError('EXACT39_REMOTE_ARTIFACT_PATHS_REQUIRED')
    if launch['scope']!=state.get('approved_scope') or launch['controller_sha256']!=state.get('source_sha256'):
        raise RuntimeError('ACTUAL_SAME_SCOPE_AND_CONTROLLER_SOURCE_REQUIRED')
    result=dict(launch);result['artifact_owner']=dict(actual)
    return result

def main():
    f=ROOT/'outputs/local_repartition_operations_v1'
    original=f/'allocation_launch.json';target=f/'allocation_verified_launch.json'
    if target.exists():raise RuntimeError('EXISTING_VERIFIED_LAUNCH_REVIEW_NO_REPEAT')
    launch=json.loads(original.read_text())
    code=f'''
import json,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as registry
p=root/'outputs/approved_frozen_allocations_repartition_workstation/status.json'
s=json.loads(p.read_text());o=s['owner']
print(json.dumps(dict(state=s,live=registry.identity(o['pid'])==o['created'])))
'''
    r=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True,check=True)
    actual=json.loads(r.stdout);value=bind(launch,actual['state'],actual['live'])
    value['original_native_launch_sha256']=sha(original)
    write_json(target,value)
    write_json(f/'allocation_launch_identity_readback.json',dict(passed=True,remote_readback=actual,original_sha256=sha(original),canonical_sha256=sha(target),scientific_dispatch=False))
    print(json.dumps(dict(passed=True,actual_owner=value['artifact_owner'],scientific_dispatch=False)))

if __name__=='__main__':main()
