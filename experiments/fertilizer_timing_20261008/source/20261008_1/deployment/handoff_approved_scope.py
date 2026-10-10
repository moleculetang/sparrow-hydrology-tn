"""Pinned Linux parent-only handoff after actual two-gate qualification.

Preflight is read-only. Actual handoff preserves scientific PIDs and tokens.
An owned dispatch hold prevents the replacement from dispatching before its
adoption is verified. Failure after old-parent exit preserves that hold.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PINNED = dict(pid=2026897, created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1357287288',
              host='pc', role='workstation')


def stopped(pid):
    path = Path('/proc') / str(pid) / 'status'
    return path.exists() and any(row.startswith('State:') and row.split()[1] in ('T', 't')
                                 for row in path.read_text().splitlines())


def main(root, preflight_only):
    root = Path(root).resolve()
    sys.path[:0] = [str(root), str(root/'scripts'), str(root/'deployment')]
    from host_resources import registry, registry_root
    from d29_platform.runtime import write_json, sha
    from controller_lock import acquire
    from approved_scope_policy import load_qualified
    from approved_scope_handoff_policy import verify_snapshot, verify_adoption
    import approved_scope_controller as controller
    if os.name != 'posix':
        raise RuntimeError('LINUX_NATIVE_PARENT_HANDOFF_ONLY')
    scope = load_qualified(root)
    controller.SCIENCE_SCOPE = scope
    _, gate = controller.validate_admission()
    proof = json.loads((root/'evidence/approved_scope_controller_build.json').read_text())
    target = root/'deployment/approved_scope_controller.py'
    if (not proof['passed'] or proof['controller_sha256'] != sha(target)
            or proof['policy_sha256'] != sha(root/'deployment/approved_scope_policy.py')
            or proof['base_sha256'] != sha(root/'deployment/contract_hold_controller_v3.py')):
        raise RuntimeError('EXACT_VERIFIED_CONTROLLER_BUILD_REQUIRED')
    for name in ['approved_scope_controller_controls.json', 'approved_scope_policy_controls.json',
                 'approved_scope_handoff_controls_v2.json']:
        if not json.loads((root/'evidence'/name).read_text()).get('passed'):
            raise RuntimeError('PRE_DISPATCH_CONTROLS_REQUIRED '+name)
    controls = json.loads((root/'evidence/approved_scope_handoff_controls_v2.json').read_text())
    if (controls['handoff_sha256'] != sha(Path(__file__))
            or controls['policy_sha256'] != sha(root/'deployment/approved_scope_handoff_policy.py')):
        raise RuntimeError('CURRENT_HANDOFF_CONTROL_SOURCE_REQUIRED')
    native = json.loads((root/'evidence/approved_scope_native_controls.json').read_text())
    if (not native.get('passed') or native.get('handoff_sha256') != sha(Path(__file__))
            or native.get('fixture_sha256') != sha(root/'deployment/test_approved_scope_native_handoff.py')
            or native.get('scientific_signals') != 0):
        raise RuntimeError('ACTUAL_CURRENT_NATIVE_PARENT_CONTROL_REQUIRED')
    campaign = root/'outputs/campaign_workstation'

    def read_current():
        if registry.identity(PINNED['pid']) != PINNED['created']:
            raise RuntimeError('PINNED_OLD_DISPATCHER_NOT_LIVE')
        args = (Path('/proc')/str(PINNED['pid'])/'cmdline').read_bytes().split(b'\0')
        if str(root/'deployment/contract_hold_controller_v3.py').encode() not in args:
            raise RuntimeError('PINNED_OLD_COMMAND_CHANGED')
        state = json.loads((campaign/'status.json').read_text())
        children = (Path('/proc')/str(PINNED['pid'])/'task'/str(PINNED['pid'])/'children').read_text().split()
        leases = registry.status(root=registry_root())['leases']
        verified = verify_snapshot(state, PINNED, gate, scope, leases, registry.identity,
                                   controller.check_child_command, map(int, children))
        return state, verified

    before, verified = read_current()
    if (campaign/'pause_requested.json').exists():
        raise RuntimeError('EXISTING_DISPATCH_HOLD_MUST_FINISH_BEFORE_HANDOFF')
    if preflight_only:
        print(json.dumps(dict(passed=True, preflight_only=True, old_owner=PINNED,
                              verified_live_children=verified, scope=scope,
                              scientific_signals=0, dispatch_changed=False)))
        return 0
    folder = root/'outputs/approved_scope_handoff_v1'
    folder.mkdir(exist_ok=True)
    lifetime = acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():
        raise RuntimeError('EXISTING_HANDOFF_REVIEW_NO_BLIND_RETRY')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    write_json(folder/'owner.json', owner)
    pause = campaign/'pause_requested.json'
    hold = dict(reason='approved_scope_parent_only_handoff_hold', owner=owner,
                original_controller=PINNED, scientific_children_signaled=False)
    parent_stopped = False
    terminated = False
    new = None
    try:
        with registry.transaction(registry_root()) as resources:
            if pause.exists():
                raise RuntimeError('CONCURRENT_DISPATCH_HOLD')
            if registry.identity(PINNED['pid']) != PINNED['created']:
                raise RuntimeError('PINNED_PARENT_CHANGED_UNDER_RESOURCE_LOCK')
            with pause.open('x', encoding='utf8') as stream:
                json.dump(hold, stream, indent=2)
            os.kill(PINNED['pid'], signal.SIGSTOP)
            parent_stopped = True
            for _ in range(100):
                if stopped(PINNED['pid']):
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('PINNED_PARENT_STOP_NOT_ACKNOWLEDGED')
            before = json.loads((campaign/'status.json').read_text())
            children = map(int, (Path('/proc')/str(PINNED['pid'])/'task'/str(PINNED['pid'])/'children').read_text().split())
            verified = verify_snapshot(before, PINNED, gate, scope, resources['leases'],
                                       registry.identity, controller.check_child_command, children)
            write_json(folder/'old_status.json', before)
            write_json(folder/'verified_children.json', verified)
            resources['events'].append(dict(time=time.time(), action='approved_scope_parent_handoff',
                                            project='20261008_1', old_owner=PINNED,
                                            scientific_signals=0))
        os.kill(PINNED['pid'], signal.SIGTERM)
        os.kill(PINNED['pid'], signal.SIGCONT)
        parent_stopped = False
        for _ in range(100):
            if registry.identity(PINNED['pid']) != PINNED['created']:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('OLD_DISPATCHER_DID_NOT_EXIT')
        terminated = True
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', D29_WORKER_THREADS='1', D29_RESOURCE_STATE=str(registry_root()))
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            env[key] = '1'
        with (folder/'controller.log').open('ab') as log:
            new = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(target),
                                    '--role', 'workstation', '--max-active', '13'],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        for _ in range(300):
            if new.poll() is not None:
                raise RuntimeError('APPROVED_CONTROLLER_FAILED_KEEP_DISPATCH_HOLD')
            after = json.loads((campaign/'status.json').read_text())
            if after['owner']['pid'] == new.pid:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('NEW_CONTROLLER_STATUS_TIMEOUT_KEEP_HOLD')
        if registry.identity(new.pid) != after['owner']['created']:
            raise RuntimeError('NEW_CONTROLLER_NATIVE_IDENTITY_CHANGED')
        verify_adoption(verified, after, scope, registry.identity)
        if json.loads(pause.read_text()) != hold:
            raise RuntimeError('HANDOFF_HOLD_OWNER_CHANGED_NO_REMOVAL')
        result = dict(passed=True, old_owner=PINNED, new_owner=after['owner'],
                      verified_children=verified, scope=scope, controller_sha256=sha(target),
                      scientific_signals=0, optimizer_edits=0, max_active=13,
                      phase='verified_two_epoch_adoption_dispatch_resumed')
        write_json(folder/'adoption_verified.json', result)
        pause.unlink()
        write_json(folder/'complete.json', result)
        write_json(folder/'status.json', result)
        print(json.dumps(result))
        return 0
    except BaseException as exc:
        if parent_stopped and not terminated and registry.identity(PINNED['pid']) == PINNED['created']:
            os.kill(PINNED['pid'], signal.SIGCONT)
        # Before old-parent exit only our own pause can be undone safely.
        if not terminated and new is None and registry.identity(PINNED['pid']) == PINNED['created']:
            if pause.exists() and json.loads(pause.read_text()) == hold:
                pause.unlink()
        write_json(folder/'status.json', dict(passed=False, error=repr(exc),
                   old_owner=PINNED, old_dispatcher_terminated=terminated,
                   new_pid=None if new is None else new.pid, scientific_signals=0,
                   automatic_restart=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    sys.exit(main(args.root, args.preflight_only))
