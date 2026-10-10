"""Read only the registered owner and correct scientific root for every job."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else None


def collect(root, jobs, ownership, role, original_gate, corrected_gate):
    root = Path(root)
    if len(jobs) != 44 or len({j['id'] for j in jobs}) != 44:
        raise RuntimeError('EXACT44_MATRIX_REQUIRED_FOR_PROGRESS')
    if set(ownership['owners']) != {j['id'] for j in jobs}:
        raise RuntimeError('EXACT44_OWNER_MAP_REQUIRED')
    if sha(root/'evidence/formal_gate.json') != original_gate:
        raise RuntimeError('ORIGINAL_PROGRESS_GATE_CHANGED')
    candidate = root.with_name('20261008_1_maize_scope_v2')
    if corrected_gate is not None and sha(candidate/'evidence/formal_gate.json') != corrected_gate:
        raise RuntimeError('APPROVED_PROGRESS_GATE_CHANGED')
    records = []
    for job in jobs:
        jid = job['id']
        if ownership['owners'][jid] != role:
            continue
        is_k = job['calendar'] == 'K'
        science = candidate if is_k else root
        gate = corrected_gate if is_k else original_gate
        folder = science/'outputs/jobs'/jid
        status = read(folder/'status.json') or {}
        owner = read(folder/'owner.json') or {}
        audit = read(folder/'independent_audit.json') or {}
        freeze = read(folder/'prediction_freeze.json') or {}
        started = bool(status or (owner.get('project') == '20261008_1' and owner.get('job') == jid))
        if started and is_k and gate is None:
            raise RuntimeError('APPROVED_K_STARTED_WITHOUT_ACTUAL_GATE '+jid)
        audited = audit.get('numerically_sufficient') is True
        frozen = freeze.get('numerically_sufficient') is True
        for proof in [audit, freeze]:
            if proof.get('numerically_sufficient') is True:
                if proof.get('formal_gate_sha256') != gate or proof.get('source_multiplier') != 1.:
                    raise RuntimeError('PROGRESS_RECEIPT_CONTRACT_MISMATCH '+jid)
        if frozen:
            if (not audited or sha(folder/'independent_audit.json') != freeze.get('audit_sha256')
                    or audit.get('parameter_sha256') != freeze.get('parameter_sha256')
                    or sha(folder/'frozen_parameters.npy') != freeze.get('parameter_sha256')
                    or freeze.get('job', {}).get('id') != jid):
                raise RuntimeError('ACTUAL_QUALIFIED_FREEZE_IDENTITY_REQUIRED '+jid)
        progress = read(folder/'solution_audit_progress.json') or {}
        records.append(dict(job=jid, calendar=job['calendar'], assigned_host=role,
                            epoch='approved_maize_K' if is_k else 'original_nonK',
                            science_root=str(science), formal_gate_sha256=gate,
                            started=started, status=status.get('status', 'not_dispatched'),
                            calls=status.get('calls'), objective=status.get('best_objective'),
                            projected_gradient=status.get('projected_gradient'),
                            active_seconds=status.get('active_seconds'),
                            audited=audited, frozen=frozen,
                            completed_audit_coordinates=len(progress.get('checks', []))))
    return records


def combine(local, remote, jobs):
    records = local + remote
    ids = [r['job'] for r in records]
    if len(records) != 44 or len(set(ids)) != 44 or set(ids) != {j['id'] for j in jobs}:
        raise RuntimeError('EXACT44_ONE_OWNER_PROGRESS_RECORDS_REQUIRED')
    if sum(r['epoch'] == 'approved_maize_K' for r in records) != 14:
        raise RuntimeError('EXACT14_CORRECTED_K_PROGRESS_REQUIRED')
    return dict(records=records, counts=dict(registered=44,
                nonK_started=sum(r['started'] and r['epoch']=='original_nonK' for r in records),
                approved_K_started=sum(r['started'] and r['epoch']=='approved_maize_K' for r in records),
                independent_audits_passed=sum(r['audited'] for r in records),
                qualified_frozen=sum(r['frozen'] for r in records)),
                evaluation_labels_read=False, physical_oracle_calls=0,
                narrow_K_results_selected=False)
