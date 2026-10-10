"""Hash-verified auxiliary deployment and read-only native recovery preflight."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main():
    root=Path(__file__).resolve().parents[1]
    files=['deployment/recover_approved_startup_failures.py','deployment/startup_failure_recovery_policy.py',
        'deployment/approved_parent_lineage.py','evidence/startup_failure_recovery_controls.json',
        'evidence/approved_parent_lineage_controls.json',
        'outputs/approved_scope_operations_transport/actual_remote_handoff_return.json']
    payload={rel:base64.b64encode((root/rel).read_bytes()).decode() for rel in files}
    code=f'''
import ast,base64,hashlib,json,os
from pathlib import Path
root=Path({REMOTE_ROOT!r});payload={payload!r};result={{}}
original='outputs/approved_scope_operations_transport/actual_remote_handoff_return.json'
for rel,data in payload.items():
 path=root/rel;raw=base64.b64decode(data)
 if rel.endswith('.py'):ast.parse(raw.decode())
 if rel==original:
  native=root/'outputs/approved_scope_handoff_v1/complete.json'
  if json.loads(native.read_bytes())!=json.loads(raw):raise RuntimeError('ACTUAL_ORIGINAL_NATIVE_HANDOFF_MUST_MATCH')
  if path.exists() and path.read_bytes()!=raw:raise RuntimeError('EXISTING_HANDOFF_RECORD_IS_IMMUTABLE')
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists() or path.read_bytes()!=raw:
  temporary=path.with_name(path.name+'.startup_recovery_stage')
  temporary.write_bytes(raw);os.replace(temporary,path)
 result[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(result))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,
        text=True,encoding='utf8',capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr)
    actual=json.loads(p.stdout)
    if actual!={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in files}:
        raise RuntimeError('ACTUAL_AUXILIARY_STAGING_BYTES_MUST_MATCH')
    f=root/'outputs/approved_scope_operations_transport/startup_recovery_handoff_v1';f.mkdir(exist_ok=True)
    (f/'staged.json').write_text(json.dumps(dict(passed=True,files=actual,science_changes=0),indent=2))
    pre=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B '+REMOTE_ROOT+
        '/deployment/recover_approved_startup_failures.py '+REMOTE_ROOT+' --preflight-only'],
        text=True,encoding='utf8',capture_output=True,timeout=60)
    (f/'preflight_stdout.json').write_text(pre.stdout);(f/'preflight_stderr.txt').write_text(pre.stderr)
    if pre.returncode:raise RuntimeError(pre.stderr)
    r=json.loads(pre.stdout)
    print(json.dumps(dict(staged=True,preflight_passed=r['passed'],live_scientific_children=len(r['verified_children']),
        old_owner=r['old_owner'],artifact_owner=r['artifact_owner'])))


if __name__=='__main__':main()
