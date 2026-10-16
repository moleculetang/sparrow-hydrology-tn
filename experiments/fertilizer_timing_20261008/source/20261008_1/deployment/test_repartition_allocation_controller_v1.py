"""Exercise actual generated stage routing, restore and ACK functions offline."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'deployment')]
from d29_platform.runtime import sha, write_json
import repartition_allocation_controller_v1 as controller
from approved_allocation_policy import owner_key


def fixture(root, jid, gate):
    f = root/'outputs/jobs'/jid
    f.mkdir(parents=True)
    (f/'frozen_parameters.npy').write_bytes(('fixture '+jid).encode())
    verifier = root/'analysis/verify_timing_allocation_artifacts.py'
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text('# fixture verifier '+gate, encoding='utf8')
    point = sha(f/'frozen_parameters.npy')
    freeze = dict(numerically_sufficient=True, formal_gate_sha256=gate, parameter_sha256=point)
    write_json(f/'prediction_freeze.json', freeze)
    return f, freeze


def main():
    receipt = ROOT/'evidence/repartition_allocation_controller_controls_v1.json'
    source_hash = sha(ROOT/'deployment/repartition_allocation_controller_v1.py')
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf8'))
        if not saved['passed'] or saved['source_sha256'] != source_hash:
            raise RuntimeError('EXISTING_ARTIFACT_CONTROLLER_CONTROL_SOURCE_CHANGED')
        print(json.dumps(saved))
        return
    folder = ROOT/'outputs/repartition_allocation_controller_controls_v1'
    folder.mkdir(exist_ok=False)
    root = folder/'20261008_1'
    candidate = root.with_name('20261008_1_maize_scope_v2')
    scope = dict(epochs={'A': 'original_nonK', 'K': 'approved_maize_K'},
                 gates={'A': 'a'*64, 'K': 'b'*64})
    controller.ROOT = root
    controller.SCIENCE_SCOPE = scope
    checks = []
    assert controller.stage_for_job('A') == 'await_scientific_freeze'
    a, af = fixture(root, 'A', scope['gates']['A'])
    k, kf = fixture(candidate, 'K', scope['gates']['K'])
    old = root/'outputs/jobs/K/prediction_freeze.json'
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text('old K must never be read', encoding='utf8')
    assert controller.stage_for_job('A') == controller.stage_for_job('K') == 'allocation_export'
    checks.extend(['awaits_actual_scientific_freeze', 'corrected_K_uses_candidate_not_original_folder'])
    allocation = dict(passed=True, source_multiplier=1., formal_gate_sha256=scope['gates']['K'],
                      parameter_sha256=kf['parameter_sha256'])
    write_json(k/'input_allocation/complete.json', allocation)
    assert controller.stage_for_job('K') == 'allocation_review'
    review = dict(passed=True, years=64, reaches=230, crops=21,
                  allocation_receipt_sha256=sha(k/'input_allocation/complete.json'),
                  parameter_sha256=kf['parameter_sha256'],
                  verifier_sha256=sha(candidate/'analysis/verify_timing_allocation_artifacts.py'))
    write_json(k/'input_allocation/independent_artifact_review.json', review)
    assert controller.stage_for_job('K') == 'complete'
    checks.append('candidate_readback_uses_candidate_verifier_identity')

    def reject_stage(name, path, changed):
        original = path.read_bytes()
        write_json(path, changed)
        try:
            controller.stage_for_job('K')
        except RuntimeError:
            checks.append(name)
        else:
            raise AssertionError('ACCEPTED '+name)
        path.write_bytes(original)

    reject_stage('old_K_freeze_gate_rejected', k/'prediction_freeze.json', dict(kf, formal_gate_sha256=scope['gates']['A']))
    reject_stage('allocation_old_multiplier_rejected', k/'input_allocation/complete.json', dict(allocation, source_multiplier=2.))
    reject_stage('partial_history_readback_rejected', k/'input_allocation/independent_artifact_review.json', dict(review, years=9))
    reject_stage('original_verifier_for_candidate_rejected', k/'input_allocation/independent_artifact_review.json',
                 dict(review, verifier_sha256=sha(root/'analysis/verify_timing_allocation_artifacts.py')))
    assert controller.recovered_exit('allocation_export', 'allocation_review') == 0
    assert controller.recovered_exit('allocation_review', 'complete') == 0
    assert controller.recovered_exit('allocation_review', 'allocation_review') == 2
    checks.append('native_stop_without_review_not_certified_complete')
    ownership = dict(owners={'A':'workstation', 'K':'workstation'})
    write_json(root/'deployment/path_ownership.json', ownership)
    owner = dict(pid=0, created='synthetic terminal parent')
    write_json(root/'outputs/campaign_workstation/owner.json', owner)
    assert not controller.transport_admitted(root, 'workstation', scope['gates']['A'])
    archive = root/'transfer'/('approved_scope_terminal_'+'d'*24+'.tar')
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'fixture archive bytes; this is not a real campaign import')
    ack = dict(version=2, transport_passed=True, remote_owner=owner,
               job_contracts={j:dict(epoch=scope['epochs'][j],gate_sha256=scope['gates'][j]) for j in ['A','K']},
               ownership_sha256=sha(root/'deployment/path_ownership.json'),
               current_host_snapshot_imported=True, archive_relative=archive.relative_to(root).as_posix(),
               archive_sha256=sha(archive), manifest_sha256='d'*64)
    ack_path = root/'evidence'/('approved_scope_terminal_import_ack_'+owner_key(owner)+'.json')
    write_json(ack_path, ack)
    assert controller.transport_admitted(root, 'workstation', scope['gates']['A'])
    checks.append('actual_archive_bytes_required_before_artifact_dispatch')
    write_json(root/'outputs/campaign_workstation/owner.json', dict(owner, created='new parent'))
    assert not controller.transport_admitted(root, 'workstation', scope['gates']['A'])
    checks.append('new_parent_requires_new_two_contract_ack')
    write_json(root/'outputs/campaign_workstation/owner.json', owner)
    write_json(ack_path, dict(ack, archive_sha256='f'*64))
    try:
        controller.transport_admitted(root, 'workstation', scope['gates']['A'])
    except RuntimeError:
        checks.append('unverified_archive_rejected')
    else:
        raise AssertionError('UNVERIFIED_ARCHIVE_ACCEPTED')
    assert controller.transport_admitted(root, 'local', scope['gates']['A'])
    checks.append('local_needs_no_remote_archive_but_still_requires_independent_scientific_freeze')
    result = dict(passed=True, count=len(checks), checks=checks, source_sha256=source_hash,
                  scientific_dispatch=False, actual_paths_exported=0,
                  scope='generated_controller_functions_offline_not_native_artifact_controller_launch')
    write_json(receipt, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
