"""Hash-check auxiliary two-contract artifact helpers without launching them."""
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
    'analysis/export_frozen_allocations_guarded.py',
    'analysis/verify_timing_allocation_artifacts.py',
    'deployment/frozen_allocation_controller.py',
    'deployment/approved_allocation_policy.py',
    'deployment/approved_frozen_allocation_controller.py',
    'evidence/frozen_allocation_controller_known_errors.json',
    'evidence/timing_allocation_readback_known_errors.json',
    'evidence/approved_allocation_policy_controls.json',
    'evidence/approved_allocation_controller_controls.json',
    'evidence/approved_allocation_controller_build.json',
]

TWO_ROOT_FILES = FILES[:2]


def main():
    folder = ROOT/'outputs/approved_scope_operations_transport'
    path = folder/'actual_remote_delivery_staging.json'
    if path.exists():
        raise RuntimeError('ACTUAL_DELIVERY_HELPERS_ALREADY_STAGED_REVIEW_NOT_RESTART')
    proof = json.loads((ROOT/'evidence/approved_allocation_controller_build.json').read_text(encoding='utf8'))
    controls = json.loads((ROOT/'evidence/approved_allocation_controller_controls.json').read_text(encoding='utf8'))
    files = {}
    for relative in FILES:
        raw = (ROOT/relative).read_bytes()
        files[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                               base64=base64.b64encode(raw).decode('ascii'))
    if (not proof['passed'] or not controls['passed']
            or proof['base_sha256'] != files['deployment/frozen_allocation_controller.py']['sha256']
            or proof['controller_sha256'] != files['deployment/approved_frozen_allocation_controller.py']['sha256']
            or controls['source_sha256'] != proof['controller_sha256']):
        raise RuntimeError('ACTUAL_CURRENT_ARTIFACT_CONTROLLER_CONTROLS_REQUIRED')
    code = f'''
import base64,hashlib,json
from pathlib import Path,PurePosixPath
root=Path({REMOTE_ROOT!r});files={files!r}
targets=[(root,rel) for rel in files]
targets += [(root.with_name('20261008_1_maize_scope_v2'),rel) for rel in {TWO_ROOT_FILES!r}]
for parent,rel in targets:
 record=files[rel];path=PurePosixPath(rel);target=parent/rel;raw=base64.b64decode(record['base64'],validate=True)
 if path.is_absolute() or '..' in path.parts or target.is_symlink() or not target.resolve().is_relative_to(parent):
  raise RuntimeError('ONLY_AUXILIARY_DELIVERY_TARGETS_ALLOWED')
 if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
  raise RuntimeError('AUXILIARY_DELIVERY_BYTES_CHANGED')
 if rel in {TWO_ROOT_FILES!r}:
  gate=json.loads((parent/'evidence/formal_gate.json').read_text())
  if rel in gate['identities']:raise RuntimeError('AUXILIARY_MUST_NOT_REPLACE_SEALED_DEPENDENCY')
 if target.exists() and target.read_bytes()!=raw:raise RuntimeError('EXISTING_DELIVERY_SOURCE_CONFLICT '+str(target))
created=[]
for parent,rel in targets:
 record=files[rel];target=parent/rel
 if not target.exists():
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(base64.b64decode(record['base64'],validate=True))
  created.append(str(target))
 if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('ACTUAL_DELIVERY_HASH_CHANGED')
print(json.dumps(dict(passed=True,files={{rel:r['sha256'] for rel,r in files.items()}},created=created,
 two_root_helpers={{rel:files[rel]['sha256'] for rel in {TWO_ROOT_FILES!r}}},
 controller_launched=False,artifact_jobs_dispatched=0,scientific_sources_modified=False)))
'''
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True, timeout=90)
    if run.returncode:
        raise RuntimeError(run.stderr+run.stdout)
    actual = json.loads(run.stdout)
    if not actual['passed'] or actual['files'] != {k:v['sha256'] for k,v in files.items()}:
        raise RuntimeError('ACTUAL_PRIVATE_DELIVERY_STAGING_MISMATCH')
    with path.open('x', encoding='utf8') as stream:
        json.dump(actual, stream, indent=2)
    print(json.dumps(actual))


if __name__ == '__main__':
    main()
