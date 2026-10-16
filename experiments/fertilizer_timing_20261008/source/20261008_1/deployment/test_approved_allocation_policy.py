"""Reject single-contract, stale, forged or wrongly routed import admission."""
import copy
import hashlib
import json
from pathlib import Path
from approved_allocation_policy import qualify_ack, owner_key

ROOT = Path(__file__).resolve().parents[1]


def main():
    owner = dict(pid=3, created='fixture')
    jobs = ['A', 'K']
    scope = dict(epochs={'A': 'original_nonK', 'K': 'approved_maize_K'},
                 gates={'A': 'a'*64, 'K': 'b'*64})
    ack = dict(version=2, transport_passed=True, remote_owner=owner,
               job_contracts={j: dict(epoch=scope['epochs'][j], gate_sha256=scope['gates'][j]) for j in jobs},
               ownership_sha256='c'*64, current_host_snapshot_imported=True,
               archive_relative='transfer/approved_scope_terminal_'+'d'*24+'.tar',
               archive_sha256='e'*64, manifest_sha256='d'*64)
    checks = []
    assert not qualify_ack(None, owner, jobs, scope, 'c'*64)
    assert not qualify_ack(dict(ack, remote_owner=dict(owner, created='old')), owner, jobs, scope, 'c'*64)
    assert qualify_ack(ack, owner, jobs, scope, 'c'*64)
    assert owner_key(owner) != owner_key(dict(owner, created='old'))
    checks.extend(['missing_import_waits', 'stale_parent_import_not_reused', 'actual_two_contract_import_admitted',
                   'creation_identity_is_part_of_ack_key'])

    def reject(name, mutate):
        value = copy.deepcopy(ack)
        mutate(value)
        try:
            qualify_ack(value, owner, jobs, scope, 'c'*64)
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError('ACCEPTED '+name)

    reject('single_gate_format_rejected', lambda v: v.update(version=1))
    reject('unchecked_import_rejected', lambda v: v.update(current_host_snapshot_imported=False))
    reject('failed_transport_rejected', lambda v: v.update(transport_passed=False))
    reject('changed_ownership_rejected', lambda v: v.update(ownership_sha256='f'*64))
    reject('narrow_K_gate_rejected', lambda v: v['job_contracts']['K'].update(gate_sha256='a'*64))
    reject('missing_K_job_rejected', lambda v: v['job_contracts'].pop('K'))
    reject('unowned_extra_job_rejected', lambda v: v['job_contracts'].update(EXTRA=dict(epoch='original_nonK',gate_sha256='a'*64)))
    reject('archive_traversal_rejected', lambda v: v.update(archive_relative='transfer/../bad.tar'))
    reject('unregistered_archive_name_rejected', lambda v: v.update(archive_relative='transfer/legacy.tar'))
    reject('unknown_archive_bytes_rejected', lambda v: v.update(archive_sha256=None))
    result = dict(passed=True, count=len(checks), checks=checks,
                  source_sha256=hashlib.sha256((ROOT/'deployment/approved_allocation_policy.py').read_bytes()).hexdigest(),
                  actual_campaign_imported=False, scientific_dispatch=False)
    path = ROOT/'evidence/approved_allocation_policy_controls.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != result:
        raise RuntimeError('EXISTING_ALLOCATION_POLICY_PROOF_CHANGED')
    path.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
