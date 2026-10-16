"""Adversarial identity, epoch, command and lease controls for parent handoff."""
import copy
import hashlib
import json
from pathlib import Path
from approved_scope_handoff_policy import verify_snapshot, verify_adoption

ROOT = Path(__file__).resolve().parents[1]


def main():
    owner = dict(pid=11, created='parent')
    scope = dict(epochs={'A': 'original_nonK', 'K': 'approved_maize_K'},
                 gates={'A': 'oldgate', 'K': 'newgate'})
    lease = dict(pid=22, created='child', project='20261008_1', job='train/A',
                 cpus=[2], threads=1, reserved_bytes=100, gpu=None)
    child = dict(job='A', pid=22, created='child', stage='train', lease=copy.deepcopy(lease))
    state = dict(owner=owner, formal_gate_sha256='oldgate', running=[child], max_active=13)
    resources = {'token': lease}
    identity = lambda pid: {11: 'parent', 22: 'child'}.get(pid)
    checks = []

    def verify(s=state, r=resources, children=(22,), command=lambda record, lease: None):
        return verify_snapshot(s, owner, 'oldgate', scope, r, identity, command, children)

    before = verify()
    assert before[0]['token'] == 'token'
    checks.append('legacy_nonK_keyed_token_retained')

    def reject(name, mutate, target='state'):
        s, r = copy.deepcopy(state), copy.deepcopy(resources)
        mutate(s if target == 'state' else r)
        try:
            verify(s, r)
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError('ACCEPTED '+name)

    reject('changed_parent_rejected', lambda s: s['owner'].update(created='other'))
    reject('changed_parent_gate_rejected', lambda s: s.update(formal_gate_sha256='newgate'))
    reject('unmeasured_capacity_rejected', lambda s: s.update(max_active=14))
    reject('duplicate_children_rejected', lambda s: s['running'].append(copy.deepcopy(child)))
    reject('old_live_K_rejected', lambda s: s['running'][0].update(job='K'))
    reject('wrong_epoch_rejected', lambda s: s['running'][0].update(epoch='approved_maize_K', job_gate_sha256='newgate'))
    reject('foreign_project_lease_rejected', lambda r: r['token'].update(project='peer'), 'resources')
    reject('changed_reservation_rejected', lambda r: r['token'].update(reserved_bytes=101), 'resources')
    reject('unbound_dispatch_lease_rejected', lambda r: r.update(unbound=dict(pid=11)), 'resources')
    try:
        verify(children=(22, 33))
    except RuntimeError:
        checks.append('unrecorded_OS_child_rejected')
    else:
        raise AssertionError('UNRECORDED_CHILD_ACCEPTED')
    try:
        verify(command=lambda record, lease: (_ for _ in ()).throw(RuntimeError('wrong root')))
    except RuntimeError:
        checks.append('wrong_live_command_rejected')
    else:
        raise AssertionError('WRONG_COMMAND_ACCEPTED')
    adopted = dict(child, epoch='original_nonK', job_gate_sha256='oldgate',
                   lease=dict(lease, token='token'))
    after = dict(approved_scope=scope, max_active=13, reviews=[], running=[adopted])
    assert verify_adoption(before, after, scope, identity)
    checks.append('actual_PID_and_token_adoption_required')
    changed = copy.deepcopy(after)
    changed['running'][0]['lease']['token'] = 'replacement'
    try:
        verify_adoption(before, changed, scope, identity)
    except RuntimeError:
        checks.append('new_lease_for_old_live_child_rejected')
    else:
        raise AssertionError('TOKEN_REPLACEMENT_ACCEPTED')
    changed = copy.deepcopy(after)
    changed['reviews'] = [dict(job='K', reason='narrow_K_epoch_requires_approved_range_revision')]
    try:
        verify_adoption(before, changed, scope, identity)
    except RuntimeError:
        checks.append('stale_approved_K_hold_rejected')
    else:
        raise AssertionError('STALE_K_HOLD_ACCEPTED')
    result = dict(passed=True, count=len(checks), checks=checks,
                  policy_sha256=hashlib.sha256((ROOT/'deployment/approved_scope_handoff_policy.py').read_bytes()).hexdigest(),
                  handoff_sha256=hashlib.sha256((ROOT/'deployment/handoff_approved_scope.py').read_bytes()).hexdigest(),
                  control_scope='offline_identity_and_routing_only',
                  actual_native_handoff_passed=False, scientific_signals=0)
    path = ROOT/'evidence/approved_scope_handoff_controls_v2.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != result:
        raise RuntimeError('EXISTING_CONTROL_RECEIPT_CHANGED_REVIEW')
    path.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
