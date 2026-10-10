"""New versioned operational files; preserve the failed v2 attempt."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import sha,write_json

FILES=['deployment/controller_recovery_policy.py','deployment/contract_hold_controller_v3.py',
       'deployment/handoff_contract_hold_v3.py','evidence/contract_hold_controller_build_v3.json',
       'evidence/controller_recovery_policy_known_errors.json']

def run(args,**kwargs):
    p=subprocess.run(args,text=True,encoding='utf8',capture_output=True,timeout=60,**kwargs)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    return p.stdout

def main():
    manifest={rel:sha(ROOT/rel) for rel in FILES}
    stage=REMOTE_ROOT+'/.incoming_contract_hold_v3_'+manifest['deployment/handoff_contract_hold_v3.py'][:16]
    run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=f"from pathlib import Path\nPath({stage!r}).mkdir(exist_ok=True)\n")
    for index,rel in enumerate(FILES):
        run(['scp',*OPTIONS,str(ROOT/rel),'pc@10.5.8.38:'+stage+'/'+str(index)])
    code=f'''
import hashlib,json,shutil,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});stage=Path({stage!r});manifest={manifest!r}
sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
old=json.loads((root/'outputs/contract_hold_handoff_v2/status.json').read_text())
if old.get('passed') or old.get('old_dispatcher_terminated') or old.get('scientific_signals')!=0 or "token" not in old.get('error',''):
    raise RuntimeError('FAILED_V2_REVIEW_DOES_NOT_PERMIT_VERSIONED_REPAIR')
owner=old['old_owner']
if reg.identity(owner['pid'])!=owner['created']:raise RuntimeError('ORIGINAL_CONTROLLER_NO_LONGER_LIVE')
gate=json.loads((root/'evidence/formal_gate.json').read_text())
for rel in manifest:
    if rel in gate['identities']:raise RuntimeError('SEALED_SOURCE_REPLACEMENT_FORBIDDEN')
for index,(rel,digest) in enumerate(manifest.items()):
    source=stage/str(index);target=root/rel
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('STAGED_HASH_MISMATCH')
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('EXISTING_OPERATIONAL_FILE_CONFLICT')
for index,(rel,digest) in enumerate(manifest.items()):
    target=root/rel
    if not target.exists():shutil.copyfile(stage/str(index),target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('ACTIVATED_HASH_MISMATCH')
print(json.dumps(dict(passed=True,files=manifest,original_dispatcher_live=True,
    failed_v2_retained=True,scientific_sources_changed=False)))
'''
    result=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code))
    write_json(ROOT/'outputs/maize_scope_revision_candidate/remote_operational_helpers_v3.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
