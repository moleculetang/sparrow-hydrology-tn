"""Rearm only archived, qualified same-point epochs after the host controller exits.

Operational status changes never certify a solution or start a fit. The existing
measured controller and normal stage lease must be used for any later dispatch.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from d29_platform.runtime import sha, write_json


def qualified_epoch(root, jid, gate_hash):
    folder = root / 'outputs/jobs' / jid
    history_path = folder / 'continuation_history.json'
    history = json.loads(history_path.read_text())
    attempts = history.get('attempts', [])
    if not attempts:
        raise RuntimeError('ACTUAL_PREPARED_EPOCH_REQUIRED')
    record = attempts[-1]
    if (record.get('continuation_prepared') is not True or
            record.get('continuation_started') is not False or
            record.get('training_started') is not False or
            record.get('same_point') is not True or
            record.get('new_canonical_memory') is not True or
            record.get('old_workspace_reencoded') is not False):
        raise RuntimeError('EXACT_UNSTARTED_CANONICAL_EPOCH_REQUIRED')
    if record.get('formal_gate_sha256') != gate_hash:
        raise RuntimeError('REENTRY_GATE_CHANGED')
    if sha(folder / 'best.npy') != record['parameter_sha256']:
        raise RuntimeError('REENTRY_POINT_CHANGED')
    if sha(folder / 'optimizer.pkl') != record['new_checkpoint_sha256']:
        raise RuntimeError('PREPARED_NATIVE_CHECKPOINT_CHANGED')
    archive_source = folder / record['archive']
    archive = archive_source.resolve()
    if not archive.is_relative_to(folder.resolve()) or archive_source.is_symlink():
        raise RuntimeError('ORIGINAL_JOB_ARCHIVE_REQUIRED')
    if json.loads((archive / 'prepared.json').read_text()) != record:
        raise RuntimeError('PREPARED_JOURNAL_MISMATCH')
    journal = json.loads((archive / 'preparation_journal.json').read_text())
    if journal != dict(phase='prepared_not_dispatched', record=record):
        raise RuntimeError('COMPLETE_PREPARATION_TRANSACTION_REQUIRED')
    manifest = json.loads((archive / 'archive_manifest.json').read_text())
    if (manifest.get('job') != jid or manifest.get('gate_sha256') != gate_hash or
            manifest.get('parameter_sha256') != record['parameter_sha256'] or
            manifest.get('diagnostic_sha256') != record['diagnostic_sha256'] or
            manifest.get('qualification', {}).get('qualified_for_reviewed_same_point_reentry') is not True):
        raise RuntimeError('ORIGINAL_QUALIFIED_DIAGNOSTIC_REQUIRED')
    required = {'optimizer.pkl', 'status.json', 'best.npy', 'best_gradient.npy',
                'training_contract.json', 'independent_audit.json'}
    if not required <= set(manifest.get('files', {})):
        raise RuntimeError('COMPLETE_ORIGINAL_NATIVE_ARCHIVE_REQUIRED')
    for rel, item in manifest['files'].items():
        source = archive / rel
        p = source.resolve()
        if (not p.is_relative_to(archive) or source.is_symlink() or
                p.stat().st_size != item['bytes'] or sha(p) != item['sha256']):
            raise RuntimeError('ORIGINAL_ARCHIVE_CHANGED ' + rel)
    diagnostic_path = (folder / 'continuation_diagnostics' /
                       record['parameter_sha256'] / 'complete.json')
    if sha(diagnostic_path) != record['diagnostic_sha256']:
        raise RuntimeError('ACTUAL_DESCENT_DIAGNOSTIC_CHANGED')
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get('source_sha256') != sha(root / 'analysis/probe_same_point_descent.py'):
        raise RuntimeError('CURRENT_ACTUAL_DESCENT_PROBE_REQUIRED')
    from continuation_review_policy import review
    audit = json.loads((archive / 'independent_audit.json').read_text())
    contract = json.loads((archive / 'training_contract.json').read_text())
    qualification = review(audit, contract, diagnostic.get('witness'), attempts[:-1])
    if (not diagnostic.get('passed') or diagnostic.get('new_solver_started') is not False or
            not qualification['qualified_for_reviewed_same_point_reentry']):
        raise RuntimeError('ACTUAL_DESCENT_AND_FULL_AUDIT_NOT_QUALIFIED')
    status = json.loads((folder / 'status.json').read_text())
    if (status.get('status') != 'safe_checkpoint_paused' or
            status.get('reviewed_reentry_prepared') is not True or
            status.get('audit_completed') is not False or
            status.get('numerically_sufficient') is not False or
            status.get('source_multiplier') != 1. or
            status.get('reentry_history_sha256') != sha(history_path)):
        raise RuntimeError('UNSTARTED_REENTRY_STATUS_REQUIRED')
    if (folder / 'independent_audit.json').exists() or (folder / 'prediction_freeze.json').exists():
        raise RuntimeError('OLD_TERMINAL_RECEIPTS_NOT_ARCHIVED')
    return dict(job=jid, point_sha256=record['parameter_sha256'],
                checkpoint_sha256=record['new_checkpoint_sha256'],
                history_sha256=sha(history_path), archive=str(archive))


def rearmed_state(state, qualified):
    names = {r['job'] for r in qualified}
    if not names or len(names) != len(qualified):
        raise RuntimeError('NONEMPTY_UNIQUE_REENTRY_LIST_REQUIRED')
    if state.get('running'):
        raise RuntimeError('NO_LIVE_OR_UNREAPED_CHILD_RECORDS_REQUIRED')
    if state.get('phase') != 'remaining_paths_require_same_point_review':
        raise RuntimeError('ACTUAL_INSUFFICIENT_CAMPAIGN_TERMINAL_REQUIRED')
    if not names <= set(state['owned_paths']) or names & set(state.get('completed', [])):
        raise RuntimeError('ORIGINAL_UNFINISHED_OWNERSHIP_REQUIRED')
    for jid in names:
        rows = [r for r in state.get('reviews', []) if r['job'] == jid]
        if not rows or any(r['reason'] != 'same_point_review_required' for r in rows):
            raise RuntimeError('ONLY_REVIEWED_INSUFFICIENT_STOPS_REARMED')
    result = dict(state)
    result['reviews'] = [r for r in state.get('reviews', []) if r['job'] not in names]
    result['events'] = [*state.get('events', []), dict(
        operational_reviewed_reentry_ready=True, qualified_epochs=qualified,
        training_dispatched=False, numerically_sufficient_certified=False)]
    result['phase'] = 'reviewed_epochs_prepared_for_existing_controller'
    return result


def main(jids):
    from controller_lock import acquire
    from host_resources import registry, registry_root
    from train_job import validate_admission
    _, gate_hash = validate_admission()
    controls = json.loads((ROOT / 'evidence/rearm_reviewed_campaign_known_errors.json').read_text())
    if (controls.get('passed') is not True or
            controls.get('source_sha256') != sha(ROOT / 'analysis/rearm_reviewed_campaign.py') or
            not controls.get('negative_controls') or
            any(x.get('passed') is not True for x in controls['negative_controls'])):
        raise RuntimeError('CURRENT_REARM_NEGATIVE_CONTROLS_REQUIRED')
    role = 'local' if os.name == 'nt' else 'workstation'
    folder = ROOT / 'outputs' / ('campaign_' + role)
    lifetime = acquire(folder / 'lifetime.lock')
    owner = json.loads((folder / 'owner.json').read_text())
    if registry.identity(owner['pid']) == owner['created']:
        raise RuntimeError('EXACT_ORIGINAL_CONTROLLER_MUST_HAVE_EXITED')
    state = json.loads((folder / 'status.json').read_text())
    if state['owner'] != owner or state['formal_gate_sha256'] != gate_hash:
        raise RuntimeError('CAMPAIGN_OWNER_OR_GATE_CHANGED')
    ownership = json.loads((ROOT / 'deployment/path_ownership.json').read_text())
    if any(ownership['owners'].get(jid) != role for jid in jids):
        raise RuntimeError('ORIGINAL_MACHINE_OWNERSHIP_REQUIRED')
    for lease in registry.status(root=registry_root())['leases'].values():
        if lease['project'] == '20261008_1' and lease['job'].split('/', 1)[-1] in state['owned_paths']:
            raise RuntimeError('HOST_CAMPAIGN_STAGE_LEASE_STILL_LIVE')
    qualified = [qualified_epoch(ROOT, jid, gate_hash) for jid in jids]
    replacement = rearmed_state(state, qualified)
    fingerprint = sha(folder / 'status.json')
    archive = folder / 'reviewed_reentry_campaign_states' / fingerprint
    if archive.exists():
        raise RuntimeError('EXISTING_REARM_JOURNAL_REQUIRES_REVIEW')
    archive.mkdir(parents=True)
    (archive / 'status_before.json').write_bytes((folder / 'status.json').read_bytes())
    if sha(archive / 'status_before.json') != fingerprint:
        raise RuntimeError('ORIGINAL_CAMPAIGN_STATUS_ARCHIVE_MISMATCH')
    write_json(archive / 'qualification.json', dict(
        passed=True, qualified_epochs=qualified, original_owner=owner,
        gate_sha256=gate_hash, scientific_checkpoint_mutated=False,
        training_dispatched=False, original_ownership_unchanged=True))
    write_json(folder / 'status.json', replacement)
    write_json(archive / 'complete.json', dict(
        passed=True, updated_status_sha256=sha(folder / 'status.json'),
        removed_review_jobs=sorted(jids), training_dispatched=False))
    print(json.dumps(dict(prepared_for_guarded_controller=True, jobs=sorted(jids),
                          training_dispatched=False)))


if __name__ == '__main__':
    main(sys.argv[1:])
