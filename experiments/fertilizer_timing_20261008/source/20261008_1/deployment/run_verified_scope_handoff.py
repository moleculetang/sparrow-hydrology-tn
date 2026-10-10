"""Launch or reattach one verified parent-only handoff after actual gate return.

A broken SSH observation never restarts the remote operation. No scientific
child is signaled by this coordinator; the native helper only replaces its
identified dispatcher parent after preserving all scientific PIDs and tokens.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'deployment')]
from controller_lock import acquire
from d29_platform.runtime import sha, write_json
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def main():
    folder = ROOT/'outputs/approved_scope_operations_transport'
    lifetime = acquire(folder/'handoff_coordinator.lock')
    candidate = ROOT.with_name('20261008_1_maize_scope_v2')
    gate_path = candidate/'evidence/formal_gate.json'
    gate = json.loads(gate_path.read_text(encoding='utf8'))
    native = json.loads((ROOT/'evidence/approved_scope_native_controls.json').read_text(encoding='utf8'))
    stage = json.loads((folder/'actual_remote_auxiliary_staging.json').read_text(encoding='utf8'))
    seal = json.loads((candidate/'evidence/remote_final_seal.json').read_text(encoding='utf8'))
    if (not gate.get('passed') or not gate.get('sealed') or not native['passed'] or not stage['passed']
            or not seal['passed'] or seal['gate_sha256'] != sha(gate_path)):
        raise RuntimeError('ACTUAL_CURRENT_GATE_SEAL_AND_NATIVE_CONTROLS_REQUIRED')
    source_hash = sha(ROOT/'deployment/handoff_approved_scope.py')
    if source_hash != native['handoff_sha256'] or stage['files']['deployment/handoff_approved_scope.py'] != source_hash:
        raise RuntimeError('EXACT_NATIVE_VERIFIED_HANDOFF_SOURCE_REQUIRED')
    templates = json.loads((ROOT/'evidence/approved_scope_remote_template_controls_v2.json').read_text(encoding='utf8'))
    if not templates['passed'] or templates['source_sha256']['run_verified_scope_handoff.py'] != sha(Path(__file__)):
        raise RuntimeError('CURRENT_COMPILED_REMOTE_HANDOFF_TEMPLATE_REQUIRED')
    target = folder/'actual_remote_handoff_return.json'
    if target.exists():
        raise RuntimeError('ACTUAL_HANDOFF_ALREADY_RETURNED_NO_REPEAT')
    intent_path = folder/'handoff_operation_intent.json'
    intent = dict(original_gate_sha256=sha(ROOT/'evidence/formal_gate.json'),
                  corrected_gate_sha256=sha(gate_path), handoff_sha256=source_hash,
                  old_owner=dict(pid=2026897, created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1357287288',
                                 host='pc', role='workstation'),
                  scientific_signals=0, max_active=13)
    allow_launch = not intent_path.exists()
    if not allow_launch and json.loads(intent_path.read_text(encoding='utf8')) != intent:
        raise RuntimeError('EXISTING_HANDOFF_OPERATION_IDENTITY_CHANGED')
    if allow_launch:
        with intent_path.open('x', encoding='utf8') as stream:
            json.dump(intent, stream, indent=2)
    code = f'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import write_json,sha
files={stage['files']!r};intent={intent!r}
for rel,digest in files.items():
 if sha(root/rel)!=digest:raise RuntimeError('EXACT_STAGED_OPERATIONAL_SOURCE_CHANGED '+rel)
candidate=root.with_name('20261008_1_maize_scope_v2')
if sha(root/'evidence/formal_gate.json')!=intent['original_gate_sha256'] or sha(candidate/'evidence/formal_gate.json')!=intent['corrected_gate_sha256']:
 raise RuntimeError('EXACT_TWO_GATE_HANDOFF_IDENTITY_REQUIRED')
folder=root/'outputs/approved_scope_operations_transport';folder.mkdir(exist_ok=True)
lock=acquire(folder/'handoff_launcher.lock');launchfile=folder/'handoff_launch.json'
helper=root/'deployment/handoff_approved_scope.py'
if launchfile.exists():
 launch=json.loads(launchfile.read_text())
 if launch['intent']!=intent:raise RuntimeError('EXISTING_REMOTE_HANDOFF_INTENT_CHANGED')
else:
 if not {allow_launch!r}:raise RuntimeError('UNKNOWN_LAUNCH_OUTCOME_REQUIRES_NATIVE_REVIEW_NO_RESTART')
 if (root/'outputs/approved_scope_handoff_v1/owner.json').exists():raise RuntimeError('EXISTING_NATIVE_HANDOFF_NO_RESTART')
 env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()))
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 with (folder/'handoff_launch.log').open('ab') as log:
  child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),str(root)],cwd=root,env=env,
   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(owner=dict(pid=child.pid,created=registry.identity(child.pid)),intent=intent)
 write_json(launchfile,launch)
owner=launch['owner']
while registry.identity(owner['pid'])==owner['created']:
 args=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().split(b'\\0')
 if str(helper).encode() not in args:raise RuntimeError('ACTUAL_HANDOFF_COMMAND_CHANGED')
 time.sleep(2)
actual=root/'outputs/approved_scope_handoff_v1/complete.json'
if not actual.exists():
 failed=root/'outputs/approved_scope_handoff_v1/status.json'
 raise RuntimeError('ACTUAL_HANDOFF_FAILED_NO_RESTART '+(failed.read_text() if failed.exists() else 'no terminal receipt'))
receipt=json.loads(actual.read_text())
if not receipt['passed'] or receipt['old_owner']!=intent['old_owner'] or receipt['controller_sha256']!=files['deployment/approved_scope_controller.py']:
 raise RuntimeError('ACTUAL_HANDOFF_RESULT_IDENTITY_CHANGED')
current=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if current['owner']!=receipt['new_owner']:raise RuntimeError('ACTUAL_NEW_PARENT_CHANGED')
live=registry.identity(current['owner']['pid'])==current['owner']['created']
if not live and current['phase'] not in ('all_owned_paths_frozen','remaining_paths_require_same_point_review','safely_drained'):
 raise RuntimeError('NEW_PARENT_NOT_LIVE_OR_TERMINAL')
print(json.dumps(dict(receipt=receipt,receipt_sha256=sha(actual),launch=launch,
 current_phase=current['phase'],current_parent_live=live)))
'''
    # No timeout that could terminate the handoff: this is one event wait.
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True)
    if run.returncode:
        write_json(folder/'handoff_observation_review.json', dict(error=run.stderr+run.stdout,
                   operation_intent=intent, remote_operation_restarted=False))
        raise RuntimeError(run.stderr+run.stdout)
    result = json.loads(run.stdout)
    receipt = result['receipt']
    if receipt['scientific_signals'] != 0 or receipt['optimizer_edits'] != 0 or receipt['max_active'] != 13:
        raise RuntimeError('ACTUAL_PARENT_ONLY_HANDOFF_REQUIRED')
    write_json(folder/'handoff_return_transport.json', result)
    write_json(target, receipt)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
