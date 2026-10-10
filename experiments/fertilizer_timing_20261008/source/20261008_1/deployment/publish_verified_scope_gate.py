"""Private SSH gate activation after actual same-point and restore receipts.

No model dispatch, no parent signals, no original science mutation. Unknown
outcomes are recovered by reading exact existing bytes on the next invocation.
"""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT.with_name('20261008_1_maize_scope_v2')
sys.path.insert(0, str(ROOT/'deployment'))
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate = CANDIDATE/'evidence/formal_gate.json'
    value = json.loads(gate.read_text(encoding='utf8'))
    actual = json.loads((ROOT/'outputs/maize_scope_revision_candidate/transport_v4/dualhost_comparison.json').read_text(encoding='utf8'))
    if not value.get('passed') or not value.get('sealed') or not actual.get('passed'):
        raise RuntimeError('ACTUAL_SEALED_AND_DUALHOST_GATE_REQUIRED')
    activation = ROOT/'deployment/activate_approved_scope_seal.py'
    raw = gate.read_bytes()
    source = activation.read_bytes()
    digest = sha(gate)
    source_hash = sha(activation)
    remote_candidate = str(Path(REMOTE_ROOT).with_name('20261008_1_maize_scope_v2')).replace('\\', '/')
    # Paths use known fixed POSIX strings, with only hex SHA suffixes.
    code = f'''
import base64,hashlib,importlib.util,json
from pathlib import Path
root=Path({remote_candidate!r});transfer=root/'transfer'
inputs=[('approved_gate_'+{digest!r}+'.json',{base64.b64encode(raw).decode('ascii')!r},{digest!r}),
        ('activate_approved_scope_seal_'+{source_hash!r}+'.py',{base64.b64encode(source).decode('ascii')!r},{source_hash!r})]
paths=[]
for name,encoded,digest in inputs:
 raw=base64.b64decode(encoded,validate=True);target=transfer/name
 if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError('GATE_TRANSPORT_BYTES_CHANGED')
 if target.exists() and target.read_bytes()!=raw:raise RuntimeError('PRIVATE_GATE_TRANSPORT_CONFLICT')
 if not target.exists():
  with target.open('xb') as stream:stream.write(raw)
 paths.append(target)
spec=importlib.util.spec_from_file_location('verified_scope_activation',paths[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main(str(paths[0]),{digest!r})
receipt=root/'evidence/remote_final_seal.json'
print(json.dumps(dict(gate_sha256=hashlib.sha256((root/'evidence/formal_gate.json').read_bytes()).hexdigest(),
 receipt=json.loads(receipt.read_bytes()),remote_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))
'''
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True, timeout=300)
    if run.returncode:
        raise RuntimeError(run.stderr+run.stdout)
    lines = run.stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError('EXACT_ACTIVATION_AND_READBACK_RECEIPTS_REQUIRED')
    activated, returned = map(json.loads, lines)
    if activated != returned['receipt'] or returned['gate_sha256'] != digest or activated['gate_sha256'] != digest:
        raise RuntimeError('REMOTE_GATE_OR_SEAL_READBACK_MISMATCH')
    target = CANDIDATE/'evidence/remote_final_seal.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8')) != activated:
        raise RuntimeError('LOCAL_REMOTE_SEAL_RECEIPT_CONFLICT')
    if not target.exists():
        with target.open('x', encoding='utf8') as stream:
            json.dump(activated, stream, indent=2)
    folder = ROOT/'outputs/approved_scope_operations_transport'
    path = folder/'actual_remote_final_seal_return.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != returned:
        raise RuntimeError('EXISTING_ACTIVATION_RETURN_CONFLICT')
    if not path.exists():
        with path.open('x', encoding='utf8') as stream:
            json.dump(returned, stream, indent=2)
    print(json.dumps(returned))


if __name__ == '__main__':
    main()
