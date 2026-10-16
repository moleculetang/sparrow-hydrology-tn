"""Finite Windows 1/3-worker complete-target check; never owns a science path."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'deployment')]
from host_resources import registry, registry_root
from controller_lock import acquire
from train_job import validate_admission
from d29_platform.runtime import write_json, sha
from parallel_target_controller import compare


def main():
    if os.name != 'nt': raise RuntimeError('WINDOWS_ONLY')
    folder = ROOT/'outputs/windows_capacity_probe_v1'
    folder.mkdir(parents=True, exist_ok=True)
    lifetime = acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists(): raise RuntimeError('EXISTING_PROBE_REVIEW_REQUIRED')
    _, gate = validate_admission()
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    write_json(folder/'owner.json', owner)
    state = dict(owner=owner, phase='finite_capacity_check', rounds=[],
                 scientific_paths_dispatched=0, ownership_changed=False,
                 no_evaluation_labels_read=True, gate_sha256=gate)
    reference = None
    for count in (1, 3):
        rid = f'windows_capacity_v1_workers{count}'
        barrier = ROOT/'outputs/parallel_target_pilot'/rid
        barrier.mkdir(parents=True, exist_ok=False)
        children = []; denial = None; samples = []
        try:
            for index in range(count):
                jid = f'{rid}_{index}'
                lease = registry.acquire('20261008_1', 'parallel_target/'+jid,
                                         threads=1, peak_bytes=6500000000, root=registry_root())
                if not lease.get('granted'):
                    denial = lease; break
                log = (folder/f'{rid}_{index}.log').open('ab')
                env = os.environ.copy(); env.update(PYTHONUTF8='1', D29_WORKER_THREADS='1')
                child = None
                try:
                    child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B',
                        str(ROOT/'scripts/guarded_worker.py'), lease['token'],
                        'deployment/measure_windows_parallel_targets_v1.py', rid, str(index)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children.append(dict(process=child, log=log, lease=lease, index=index))
                    registry.bind(lease['token'], child.pid, root=registry_root())
                except Exception:
                    if child is None:
                        log.close(); registry.release(lease['token'], root=registry_root())
                    raise
                target = ROOT/'outputs/jobs'/('parallel_target_'+jid)
                deadline = time.monotonic()+1800
                while not (target/'ready.json').exists():
                    if child.poll() is not None: raise RuntimeError('CONSTRUCTOR_FAILED')
                    if time.monotonic() > deadline: raise RuntimeError('CONSTRUCTOR_WAIT_EXPIRED')
                    time.sleep(.5)
                ready = json.loads((target/'ready.json').read_text())
                if ready['pid'] != child.pid or ready['created'] != registry.identity(child.pid):
                    raise RuntimeError('EXACT_READY_CHILD_REQUIRED')
                state.update(workers=count, phase='cohort_ready', running=[dict(
                    pid=c['process'].pid, created=registry.identity(c['process'].pid),
                    lease=c['lease']) for c in children])
                write_json(folder/'status.json', state)
                time.sleep(3)
            full = len(children)==count and all(c['process'].poll() is None for c in children)
            write_json(barrier/('go.json' if full else 'cancel.json'), dict(owner=owner, denial=denial))
            started = time.monotonic()
            while any(c['process'].poll() is None for c in children):
                samples.append(dict(elapsed=time.monotonic()-started, memory=registry.memory(),
                    children=[dict(pid=c['process'].pid, **registry.process_memory(c['process'].pid))
                              for c in children if c['process'].poll() is None]))
                time.sleep(5)
            elapsed = time.monotonic()-started
            checks = []; receipts = []; events = []
            for c in children:
                target = ROOT/'outputs/jobs'/f'parallel_target_{rid}_{c["index"]}'
                events.append(dict(pid=c['process'].pid, exit_code=c['process'].returncode))
                if full and c['process'].returncode==0:
                    receipt = json.loads((target/'parallel_target.json').read_text())
                    receipts.append(receipt)
                    if reference is None: reference=target
                    checks.append(compare(reference, target))
                elif full: checks.append(dict(passed=False, reason='TARGET_EXIT_FAILURE'))
            write_json(folder/f'{rid}_resources.json', samples)
            passed = full and len(receipts)==count and all(c['passed'] for c in checks)
            result = dict(passed=passed, workers=count, denial=denial, children=events,
                capacity_measured=full, wall_seconds=elapsed, calls=2*count if full else 0,
                calls_per_second=2*count/elapsed if passed else None, comparisons=checks,
                receipts=receipts, peak_private_commit=max((r['peak_bytes'] for r in receipts), default=0),
                scope='two_native_requests_per_worker_not_training_convergence',
                shared_load_not_isolated=True, resource_sha256=sha(folder/f'{rid}_resources.json'))
            write_json(folder/f'{rid}.json', result)
            state['rounds'].append(result)
            state.update(phase='round_returned', running=[])
            write_json(folder/'status.json', state)
            if not passed: break
        finally:
            if any(c['process'].poll() is None for c in children) and not (barrier/'go.json').exists():
                write_json(barrier/'cancel.json', dict(reason='controller_pre_oracle_failure'))
                for c in children: c['process'].wait(timeout=30)
            for c in children:
                c['log'].close()
                if c['process'].poll() is not None:
                    registry.release(c['lease']['token'], root=registry_root())
    state.update(phase='finite_capacity_returned', passed=len(state['rounds'])==2 and
                 all(r['passed'] for r in state['rounds']), running=[])
    write_json(folder/'complete.json', state); write_json(folder/'status.json', state)
    print(json.dumps(dict(passed=state['passed'], rounds=[dict(workers=r['workers'],
        passed=r['passed'], calls_per_second=r['calls_per_second']) for r in state['rounds']])))
    return 0 if state['passed'] else 2


if __name__=='__main__': sys.exit(main())
