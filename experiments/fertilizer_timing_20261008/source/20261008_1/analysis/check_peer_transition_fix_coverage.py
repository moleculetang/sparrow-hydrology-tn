"""Verify that the shared reverse-transfer defect is already covered locally.

This is static source/receipt accounting, never a final-point derivative audit.
"""
import ast
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
    gate = json.loads((ROOT/'evidence/formal_gate.json').read_text(encoding='utf8'))
    sources = ['d29_platform/timing_kernel.py', 'timing/adapter.py', 'timing/chain.py']
    proofs = ['evidence/transition_reverse_replay_consistency.json',
              'evidence/full_transition_derivative_requalification.json',
              'evidence/dual_host_transition_requalification.json']
    candidate = ROOT.with_name('20261008_1_maize_scope_v2')
    expected = {name: sha(ROOT/name) for name in sources}
    for name, digest in expected.items():
        if gate['identities'][name] != digest or sha(candidate/name) != digest:
            raise RuntimeError('ACTUAL_REPAIRED_SOURCE_IDENTITY_CHANGED '+name)
    proof_hashes = {name: sha(ROOT/name) for name in proofs}
    for name, digest in proof_hashes.items():
        if gate['proof_sha256'][name] != digest or not json.loads((ROOT/name).read_text(encoding='utf8'))['passed']:
            raise RuntimeError('ACTUAL_REPAIR_PROOF_NOT_IN_CURRENT_GATE '+name)
    tree = ast.parse((ROOT/sources[0]).read_text(encoding='utf8'))
    reverse = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_reverse')
    callnames = {node.func.id for node in ast.walk(reverse)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    if 'transfer_expansion' not in callnames or 'correction_history' not in [a.arg for a in reverse.args.args]:
        raise RuntimeError('CORRECTED_COMPENSATED_REVERSE_NOT_PRESENT')
    adapter = ast.parse((ROOT/'timing/adapter.py').read_text(encoding='utf8'))
    if not any(isinstance(node, ast.ImportFrom) and node.module == 'd29_platform'
               and any(alias.name == 'timing_kernel' for alias in node.names) for node in ast.walk(adapter)):
        raise RuntimeError('ACTUAL_ADAPTER_NOT_USING_REPAIRED_KERNEL')
    code = f'''
import hashlib,json
from pathlib import Path
roots=[Path({REMOTE_ROOT!r}),Path({REMOTE_ROOT!r}).with_name('20261008_1_maize_scope_v2')]
expected={expected!r}
records=[]
for root in roots:
 actual={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}}
 if actual!=expected:raise RuntimeError('ACTUAL_REMOTE_REPAIRED_SOURCE_CHANGED '+str(root))
 records.append(dict(root=str(root),actual_sha256=actual))
print(json.dumps(records))
'''
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr+run.stdout)
    peer_note = ROOT.parent/'20261003_1/delivery/数值缺陷跨对话交接待授权_20261009.md'
    result = dict(passed=True, scope='static_dispatch_and_existing_sealed_repair_proof_accounting',
                  current_source_sha256=expected, actual_remote_sources=json.loads(run.stdout),
                  current_gate_sha256=sha(ROOT/'evidence/formal_gate.json'),
                  repair_proof_sha256=proof_hashes, peer_note_sha256=sha(peer_note),
                  peer_note_read_from_shared_local_directory=True,
                  peer_evaluation_scores_used=False, physical_oracle_calls=0,
                  final_points_derivative_qualified_by_this_check=False,
                  scientific_sources_modified=False, scientific_processes_signaled=0)
    target = ROOT/'evidence/peer_transition_defect_coverage_20261009.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8')) != result:
        raise RuntimeError('EXISTING_STATIC_COVERAGE_RECEIPT_CONFLICT')
    if not target.exists():
        with target.open('x', encoding='utf8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
