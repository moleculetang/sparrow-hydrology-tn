"""Pure preconditions for dispatcher-only transition; never signals a child."""
from controller_recovery_policy import resolve_record_lease
from approved_scope_policy import qualify_record


def verify_snapshot(current, expected_owner, original_gate, scope, leases,
                    identity, command_check, os_children):
    if current['owner'] != expected_owner or current['formal_gate_sha256'] != original_gate:
        raise RuntimeError('PINNED_PARENT_OR_ORIGINAL_GATE_CHANGED')
    records = current['running']
    if current['max_active'] != 13 or not 0 <= len(records) <= 13:
        raise RuntimeError('MEASURED13_CONCURRENCY_REQUIRED')
    if len({r['job'] for r in records}) != len(records) or len({r['pid'] for r in records}) != len(records):
        raise RuntimeError('DUPLICATE_SCIENTIFIC_CHILD_RECORD')
    if not set(os_children) <= {r['pid'] for r in records}:
        raise RuntimeError('UNRECORDED_OS_CHILD_REQUIRES_REVIEW')
    if any(v.get('pid') == expected_owner['pid'] for v in leases.values()):
        raise RuntimeError('UNBOUND_DISPATCH_LEASE_DEFER_HANDOFF')
    verified = []
    for record in records:
        epoch, gate = qualify_record(record, scope)
        if epoch != 'original_nonK':
            raise RuntimeError('NO_LIVE_K_EPOCH_REPLACEMENT_ALLOWED')
        if identity(record['pid']) != record['created']:
            continue  # Terminal receipt is resolved by the replacement controller.
        lease = resolve_record_lease(record, leases)
        command_check(record, lease)
        verified.append(dict(job=record['job'], pid=record['pid'], created=record['created'],
                             token=lease['token'], epoch=epoch, job_gate_sha256=gate))
    return verified


def verify_adoption(before, after, scope, identity):
    from approved_scope_policy import HOLD_REASON
    if after.get('approved_scope') != scope or after['max_active'] != 13:
        raise RuntimeError('NEW_CONTROLLER_SCOPE_OR_CAPACITY_CHANGED')
    if any(r.get('reason') == HOLD_REASON for r in after['reviews']):
        raise RuntimeError('APPROVED_K_SCOPE_HOLD_NOT_RELEASED')
    adopted = {r['job']: r for r in after['running']}
    for record in before:
        if identity(record['pid']) != record['created']:
            continue
        new = adopted.get(record['job'])
        if not new or any(new[k] != record[k] for k in ('pid', 'created', 'epoch', 'job_gate_sha256')):
            raise RuntimeError('LIVE_SCIENTIFIC_CHILD_NOT_ADOPTED')
        if new['lease'].get('token') != record['token']:
            raise RuntimeError('LIVE_SCIENTIFIC_LEASE_TOKEN_CHANGED')
    return True
