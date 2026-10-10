"""Two-contract selection, immutable transport and malicious-manifest controls."""
import copy
import hashlib
import json
from pathlib import Path
from approved_scope_return import digest, import_archive, make_archive, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'outputs/approved_scope_return_controls_v1'
    receipt = ROOT/'evidence/approved_scope_return_controls.json'
    source_hash = digest(ROOT/'deployment/approved_scope_return.py')
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf8'))
        if not saved['passed'] or saved['module_sha256'] != source_hash:
            raise RuntimeError('EXISTING_TRANSPORT_CONTROL_SOURCE_CHANGED')
        print(json.dumps(saved))
        return
    folder.mkdir(exist_ok=False)
    original = folder/'science/20261008_1'
    corrected = original.with_name('20261008_1_maize_scope_v2')
    scope = dict(epochs={'A': 'original_nonK', 'K': 'approved_maize_K'},
                 gates={'A': 'a'*64, 'K': 'b'*64})
    jobs = ['A', 'K']
    ownership_hash = 'c'*64
    data = {
        'outputs/campaign_workstation/status.json': b'{"phase":"remaining_paths_require_same_point_review"}',
        'outputs/campaign_workstation/owner.json': b'{"pid":1,"created":"toy"}',
        'outputs/jobs/A/result.json': b'{"epoch":"original_nonK"}',
        'outputs/jobs/K/result.json': b'{"epoch":"approved_maize_K"}',
    }
    records = []
    for name, raw in data.items():
        jid = name.split('/')[2] if name.startswith('outputs/jobs/') else None
        root = corrected if jid == 'K' else original
        path = root/name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
        records.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            epoch=scope['epochs'][jid] if jid else 'controller',
                            gate_sha256=scope['gates'][jid] if jid else None))
    stale = original/'outputs/jobs/K/result.json'
    stale.parent.mkdir(exist_ok=True, parents=True)
    stale.write_bytes(b'{"epoch":"forbidden_narrow_K"}')
    manifest = dict(version=2, host_role='workstation', owned_paths=jobs,
                    job_contracts={j: dict(epoch=scope['epochs'][j], gate_sha256=scope['gates'][j]) for j in jobs},
                    ownership_sha256=ownership_hash, selection='fixed_contract_epoch_without_score_selection',
                    files=records, actual_bytes=sum(r['bytes'] for r in records),
                    terminal_phase='remaining_paths_require_same_point_review')
    checks = []
    validate_manifest(manifest, jobs, scope, ownership_hash)
    packed = make_archive(original, manifest, scope, folder/'archives')
    imported = import_archive(packed['archive'], packed['sha256'], folder/'imported', jobs, scope, ownership_hash)
    assert imported['passed'] and not imported['numerical_sufficiency_certified']
    assert (folder/'imported/outputs/jobs/K/result.json').read_bytes() == data['outputs/jobs/K/result.json']
    checks.extend(['roundtrip_actual_candidate_K_selected', 'narrow_K_not_packaged',
                   'review_terminal_not_convergence_certificate'])
    repeated = import_archive(packed['archive'], packed['sha256'], folder/'imported', jobs, scope, ownership_hash)
    assert repeated == imported
    checks.append('identical_byte_import_resume_allowed')

    def reject(name, mutation):
        changed = copy.deepcopy(manifest)
        mutation(changed)
        try:
            validate_manifest(changed, jobs, scope, ownership_hash)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError('ACCEPTED '+name)

    reject('single_gate_v1_rejected', lambda m: m.update(version=1))
    reject('old_K_contract_rejected', lambda m: m['job_contracts']['K'].update(gate_sha256='a'*64))
    reject('file_epoch_mismatch_rejected', lambda m: m['files'][-1].update(epoch='original_nonK'))
    reject('file_gate_mismatch_rejected', lambda m: m['files'][-1].update(gate_sha256='a'*64))
    reject('score_based_epoch_selection_rejected', lambda m: m.update(selection='lowest_NSE'))
    reject('duplicate_owned_ID_rejected', lambda m: m['owned_paths'].append('K'))
    reject('duplicate_file_rejected', lambda m: m['files'].append(copy.deepcopy(m['files'][-1])))
    reject('raw_source_path_rejected', lambda m: m['files'][-1].update(path='inputs/raw_observations.csv'))
    reject('path_traversal_rejected', lambda m: m['files'][-1].update(path='outputs/jobs/K/../../escape'))
    reject('byte_total_mismatch_rejected', lambda m: m.update(actual_bytes=m['actual_bytes']+1))
    reject('ownership_identity_change_rejected', lambda m: m.update(ownership_sha256='d'*64))
    try:
        import_archive(packed['archive'], 'd'*64, folder/'wrong_hash', jobs, scope, ownership_hash)
    except RuntimeError:
        checks.append('archive_byte_identity_required')
    else:
        raise AssertionError('WRONG_ARCHIVE_HASH_ACCEPTED')
    (folder/'imported/outputs/jobs/K/result.json').write_bytes(b'changed fixture only')
    try:
        import_archive(packed['archive'], packed['sha256'], folder/'imported', jobs, scope, ownership_hash)
    except RuntimeError:
        checks.append('immutable_import_conflict_rejected')
    else:
        raise AssertionError('IMPORTED_RESULT_OVERWRITTEN')
    result = dict(passed=True, count=len(checks), checks=checks, module_sha256=source_hash,
                  fixture_archive_sha256=packed['sha256'],
                  actual_remote_campaign_imported=False, model_calls=0, pickle_loaded=False,
                  scientific_directory_overwritten=False)
    receipt.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
