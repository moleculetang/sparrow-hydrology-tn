"""Run only the tiny staged Linux dispatcher fixture, with byte verification."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment'))
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT/'outputs/approved_scope_operations_transport'
    staged = json.loads((folder/'actual_remote_auxiliary_staging.json').read_text(encoding='utf8'))
    if not staged['passed']:
        raise RuntimeError('ACTUAL_STAGING_REQUIRED')
    target = ROOT/'evidence/approved_scope_native_controls.json'
    if target.exists():
        raise RuntimeError('ACTUAL_NATIVE_CONTROL_ALREADY_RETURNED_REVIEW')
    code = f'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r})
files={staged['files']!r}
for rel,digest in files.items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('STAGED_OPERATIONAL_SOURCE_CHANGED '+rel)
env=os.environ.copy()
env.update(PYTHONUTF8='1',D29_RESOURCE_STATE=str(root.parent/'.compute_coordination/state'))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
p=subprocess.run([sys.executable,'-X','utf8','-B',str(root/'deployment/test_approved_scope_native_handoff.py'),str(root)],
 cwd=root,env=env,capture_output=True,text=True,timeout=30)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
receipt=root/'evidence/approved_scope_native_controls.json'
raw=receipt.read_bytes()
if json.loads(p.stdout)!=json.loads(raw):raise RuntimeError('NATIVE_CONTROL_OUTPUT_RECEIPT_MISMATCH')
print(json.dumps(dict(receipt=json.loads(raw),remote_receipt_sha256=hashlib.sha256(raw).hexdigest())))
'''
    result = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                            input=code, text=True, encoding='utf8', capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr+result.stdout)
    returned = json.loads(result.stdout)
    receipt = returned['receipt']
    if (not receipt['passed'] or receipt['scientific_signals'] != 0
            or receipt['handoff_sha256'] != sha(ROOT/'deployment/handoff_approved_scope.py')
            or receipt['fixture_sha256'] != sha(ROOT/'deployment/test_approved_scope_native_handoff.py')):
        raise RuntimeError('ACTUAL_NATIVE_CONTROL_IDENTITY_FAILED')
    target.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    (folder/'actual_native_controls_return.json').write_text(json.dumps(returned, indent=2), encoding='utf8')
    print(json.dumps(returned))


if __name__ == '__main__':
    main()
