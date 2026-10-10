"""44-job ownership and correct-epoch progress controls, without model calls."""
import hashlib
import json
from pathlib import Path
from approved_scope_progress import collect, combine, sha

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def main():
    receipt = ROOT/'evidence/approved_scope_progress_controls.json'
    source_hash = sha(ROOT/'analysis/approved_scope_progress.py')
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf8'))
        if saved['passed'] and saved['source_sha256'] == source_hash:
            print(json.dumps(saved))
            return
        raise RuntimeError('EXISTING_PROGRESS_CONTROL_SOURCE_CHANGED')
    folder = ROOT/'outputs/approved_scope_progress_controls_v1'
    folder.mkdir(exist_ok=False)
    root = folder/'20261008_1'
    candidate = root.with_name('20261008_1_maize_scope_v2')
    write(root/'evidence/formal_gate.json', dict(passed=True, sealed=True))
    original_gate = sha(root/'evidence/formal_gate.json')
    jobs = [dict(id=f'J{i:02}', calendar='K' if i < 14 else 'C0') for i in range(44)]
    owners = dict(owners={j['id']: 'local' if j['id'] in ('J14', 'J15') else 'workstation' for j in jobs})
    # Old K deliberately contains invalid JSON: correct progress never reads it.
    old_k = root/'outputs/jobs/J00/prediction_freeze.json'
    old_k.parent.mkdir(parents=True, exist_ok=True)
    old_k.write_text('FORBIDDEN OLD K RESULT', encoding='utf8')
    local = collect(root, jobs, owners, 'local', original_gate, None)
    remote = collect(root, jobs, owners, 'workstation', original_gate, None)
    initial = combine(local, remote, jobs)
    assert len(local) == 2 and len(remote) == 42
    assert initial['counts']['qualified_frozen'] == 0 and initial['counts']['approved_K_started'] == 0
    checks = ['exact2local42remote_one_owner', 'old_narrow_K_not_read_or_counted']
    f = root/'outputs/jobs/J14'
    write(f/'status.json', dict(status='PG_met_full_audit_pending', calls=3))
    (f/'frozen_parameters.npy').write_bytes(b'fixture parameters, never loaded')
    phash = sha(f/'frozen_parameters.npy')
    write(f/'independent_audit.json', dict(numerically_sufficient=True,
          formal_gate_sha256=original_gate, source_multiplier=1., parameter_sha256=phash))
    write(f/'prediction_freeze.json', dict(numerically_sufficient=True,
          formal_gate_sha256=original_gate, source_multiplier=1., parameter_sha256=phash,
          audit_sha256=sha(f/'independent_audit.json'), job=dict(id='J14')))
    local = collect(root, jobs, owners, 'local', original_gate, None)
    assert combine(local, remote, jobs)['counts']['qualified_frozen'] == 1
    checks.append('freeze_requires_matching_point_and_audit_bytes')
    k = candidate/'outputs/jobs/J00'
    write(k/'status.json', dict(status='running', calls=1))
    try:
        collect(root, jobs, owners, 'workstation', original_gate, None)
    except RuntimeError:
        checks.append('candidate_start_without_gate_rejected')
    else:
        raise AssertionError('UNGATED_CANDIDATE_ACCEPTED')
    write(candidate/'evidence/formal_gate.json', dict(passed=True, sealed=True, corrected=True))
    corrected_gate = sha(candidate/'evidence/formal_gate.json')
    remote = collect(root, jobs, owners, 'workstation', original_gate, corrected_gate)
    assert combine(local, remote, jobs)['counts']['approved_K_started'] == 1
    checks.append('corrected_remote_K_is_counted')
    write(k/'independent_audit.json', dict(numerically_sufficient=True,
          formal_gate_sha256=original_gate, source_multiplier=1.))
    try:
        collect(root, jobs, owners, 'workstation', original_gate, corrected_gate)
    except RuntimeError:
        checks.append('candidate_audit_using_old_gate_rejected')
    else:
        raise AssertionError('OLD_GATE_CANDIDATE_AUDIT_ACCEPTED')
    try:
        combine(local, remote+[local[0]], jobs)
    except RuntimeError:
        checks.append('duplicate_host_job_not_merged_by_last_write')
    else:
        raise AssertionError('DUPLICATE_OWNER_ACCEPTED')
    try:
        combine(local, remote[:-1], jobs)
    except RuntimeError:
        checks.append('missing_registered_job_rejected')
    else:
        raise AssertionError('MISSING_JOB_ACCEPTED')
    (f/'frozen_parameters.npy').write_bytes(b'changed fixture parameters')
    try:
        collect(root, jobs, owners, 'local', original_gate, corrected_gate)
    except RuntimeError:
        checks.append('changed_frozen_parameters_rejected')
    else:
        raise AssertionError('CHANGED_PARAMETERS_ACCEPTED')
    try:
        collect(root, jobs, owners, 'local', 'f'*64, corrected_gate)
    except RuntimeError:
        checks.append('current_source_gate_identity_required')
    else:
        raise AssertionError('CHANGED_GATE_ACCEPTED')
    result = dict(passed=True, count=len(checks), checks=checks, source_sha256=source_hash,
                  evaluation_labels_read=False, model_calls=0, production_changed=False)
    receipt.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
