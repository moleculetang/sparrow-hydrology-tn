"""Hash-pinned auxiliary deployment only; no science gate or controller launch."""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment'))
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON

FILES = [
    'deployment/approved_scope_controller.py',
    'deployment/approved_scope_policy.py',
    'deployment/approved_scope_handoff_policy.py',
    'deployment/handoff_approved_scope.py',
    'deployment/test_approved_scope_native_handoff.py',
    'evidence/approved_scope_controller_build.json',
    'evidence/approved_scope_controller_controls.json',
    'evidence/approved_scope_policy_controls.json',
    'evidence/approved_scope_handoff_controls_v2.json',
]


def main():
    folder = ROOT/'outputs/approved_scope_operations_transport'
    folder.mkdir(exist_ok=True)
    path = folder/'actual_remote_auxiliary_staging.json'
    if path.exists():
        raise RuntimeError('ACTUAL_STAGING_ALREADY_RECORDED_REVIEW_NOT_RESTART')
    files = {}
    for relative in FILES:
        raw = (ROOT/relative).read_bytes()
        files[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                               bytes=len(raw), base64=base64.b64encode(raw).decode('ascii'))
    code = f'''
import base64,hashlib,json,os
from pathlib import Path,PurePosixPath
root=Path({REMOTE_ROOT!r})
files={files!r}
allowed={FILES!r}
if set(files)!=set(allowed):raise RuntimeError('EXACT_AUXILIARY_FILE_LIST_REQUIRED')
# First validate every existing target before creating any new target.
for rel,record in files.items():
 p=PurePosixPath(rel);target=(root/rel)
 if p.is_absolute() or '..' in p.parts or target.is_symlink() or not target.resolve().is_relative_to(root):
  raise RuntimeError('AUXILIARY_PATH_NOT_ALLOWED')
 raw=base64.b64decode(record['base64'],validate=True)
 if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
  raise RuntimeError('STAGED_BYTES_CHANGED')
 if target.exists() and target.read_bytes()!=raw:raise RuntimeError('EXISTING_OPERATIONAL_SOURCE_CONFLICT '+rel)
created=[]
for rel,record in files.items():
 target=root/rel
 if not target.exists():
  with target.open('xb') as stream:stream.write(base64.b64decode(record['base64'],validate=True))
  created.append(rel)
 if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('REMOTE_AUXILIARY_HASH_CHANGED')
print(json.dumps(dict(passed=True,files={{rel:rec['sha256'] for rel,rec in files.items()}},
 created=created,scientific_source_modified=False,gate_modified=False,
 controller_launched=False,scientific_signals=0)))
'''
    result = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                            input=code, text=True, encoding='utf8', capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr+result.stdout)
    receipt = json.loads(result.stdout)
    expected = {k: v['sha256'] for k, v in files.items()}
    if not receipt['passed'] or receipt['files'] != expected:
        raise RuntimeError('ACTUAL_AUXILIARY_STAGING_IDENTITY_MISMATCH')
    path.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
