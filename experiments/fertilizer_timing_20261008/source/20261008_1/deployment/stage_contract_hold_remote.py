"""Hash-verified operational helpers; no sealed scientific file is replaced."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import sha,write_json

FILES=['deployment/contract_hold_policy.py','deployment/contract_hold_controller.py',
       'deployment/handoff_contract_hold.py','deployment/maize_range_revision_v2.json',
       'evidence/contract_hold_controller_build.json','evidence/contract_hold_policy_known_errors.json']


def run(args,**kwargs):
    p=subprocess.run(args,text=True,encoding='utf8',capture_output=True,timeout=60,**kwargs)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    return p.stdout


def main():
    control=json.loads((ROOT/'evidence/contract_hold_policy_known_errors.json').read_text(encoding='utf8'))
    if not control['passed'] or control['source_sha256']!=sha(ROOT/'deployment/contract_hold_policy.py'):
        raise RuntimeError('CURRENT_LOCAL_HOLD_CONTROLS_REQUIRED')
    manifest={rel:sha(ROOT/rel) for rel in FILES}
    stage=REMOTE_ROOT+'/.incoming_contract_hold_'+manifest['deployment/handoff_contract_hold.py'][:16]
    preparation=f"from pathlib import Path\np=Path({stage!r})\np.mkdir(exist_ok=True)\n"
    run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=preparation)
    for index,rel in enumerate(FILES):
        run(['scp',*OPTIONS,str(ROOT/rel),'pc@10.5.8.38:'+stage+'/'+str(index)])
    activation=f'''
import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE_ROOT!r});stage=Path({stage!r});manifest={manifest!r}
gate=json.loads((root/'evidence/formal_gate.json').read_text())
for rel in manifest:
    if rel in gate['identities']:raise RuntimeError('SEALED_SOURCE_WRITE_FORBIDDEN '+rel)
for index,(rel,digest) in enumerate(manifest.items()):
    source=stage/str(index)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('STAGED_HELPER_HASH_MISMATCH '+rel)
    target=root/rel
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('EXISTING_HELPER_CONFLICT '+rel)
for index,(rel,digest) in enumerate(manifest.items()):
    source=stage/str(index);target=root/rel;target.parent.mkdir(exist_ok=True)
    if not target.exists():shutil.copyfile(source,target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('ACTIVATED_HELPER_HASH_MISMATCH '+rel)
print(json.dumps(dict(passed=True,files=manifest,scientific_files_replaced=False)))
'''
    result=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=activation))
    write_json(ROOT/'outputs/maize_scope_revision_candidate/remote_operational_helpers.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
