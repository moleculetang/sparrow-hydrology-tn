"""Prepare an immutable independent candidate; never edit live science."""
import datetime
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT.with_name('20261008_1_maize_scope_v2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    original = ROOT / 'evidence/formal_gate.json'
    gate = json.loads(original.read_text(encoding='utf8'))
    revision = json.loads((CANDIDATE / 'evidence/constructor_access_epoch_v3/source_revision.json').read_text(encoding='utf8'))
    strict_path = CANDIDATE / 'outputs/scope_strict_point/complete.json'
    strict = json.loads(strict_path.read_text(encoding='utf8'))
    if not gate['passed'] or not gate['sealed'] or not strict['passed']:
        raise RuntimeError('ACTUAL_SEALED_ORIGINAL_AND_STRICT_ORACLE_REQUIRED')
    if not strict['objective_bitwise_equal_to_pre_access_epoch'] or not strict['gradient_bitwise_equal_to_pre_access_epoch']:
        raise RuntimeError('STRICT_ORACLE_COMPATIBILITY_REQUIRED')
    if any(r['evaluation_year_rows_returned'] for r in strict['constructor_reads']):
        raise RuntimeError('STRICT_CONSTRUCTOR_EXPOSED_EVALUATION_ROWS')
    changed = set()
    files = []
    for rel, old in gate['identities'].items():
        p = CANDIDATE / rel
        h = sha(p)
        if h != old:
            changed.add(rel)
        files.append(dict(path=rel, bytes=p.stat().st_size, sha256=h,
                          original_sha256=old, transport='payload' if h != old else 'verified_remote_original'))
    if changed != {'config/timing_evidence.json', 'scripts/train_job.py', 'scripts/audit_solution.py'}:
        raise RuntimeError('UNREGISTERED_CANDIDATE_SCIENCE_CHANGE ' + repr(changed))
    old_config = json.loads((ROOT / 'config/timing_evidence.json').read_text(encoding='utf8'))
    new_config = json.loads((CANDIDATE / 'config/timing_evidence.json').read_text(encoding='utf8'))
    reconstructed = json.loads(json.dumps(new_config))
    profile = reconstructed['fertilizer_profiles'][1]
    if profile['basal_delta_bounds'] != [-.1, 0.]:
        raise RuntimeError('APPROVED_MAIZE_RANGE_REQUIRED')
    profile['basal_delta_bounds'] = old_config['fertilizer_profiles'][1]['basal_delta_bounds']
    profile['bounds_basis'] = old_config['fertilizer_profiles'][1]['bounds_basis']
    if reconstructed != old_config:
        raise RuntimeError('CONFIG_HAS_UNREGISTERED_DIFFERENCES')
    for r in revision['changes']:
        if gate['identities'][r['path']] != r['old_sha256'] or sha(CANDIDATE/r['path']) != r['new_sha256']:
            raise RuntimeError('CONSTRUCTOR_REVISION_CHANGED')
    helper = 'scripts/training_only_constructor.py'
    if sha(CANDIDATE/helper) != revision['helper_sha256']:
        raise RuntimeError('STRICT_CONSTRUCTOR_HELPER_CHANGED')
    additions = [helper, 'analysis/qualify_scope_remote_point.py',
                 'evidence/constructor_access_epoch_v3/source_revision.json',
                 'evidence/constructor_access_epoch_v3/completed_diagnostic_access_clarification.json',
                 'outputs/scope_strict_point/complete.json', 'outputs/scope_strict_point/point.npy']
    for rel in additions:
        p = CANDIDATE / rel
        files.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha(p), transport='payload'))
    manifest = dict(scope='independent_candidate_staging_not_production_gate',
        project='20261008_1', candidate_epoch='maize_scope_v3_strict_access',
        remote_root='/home/pc/TWY/SPARROW/Test/20261008_1_maize_scope_v2',
        original_remote_root='/home/pc/TWY/SPARROW/Test/20261008_1',
        original_gate_sha256=sha(original), files=files,
        nonK_scientific_kernel_and_inputs_unchanged=True,
        strict_oracle_actual_sha256=sha(strict_path),
        h1_contract=json.loads((CANDIDATE/'vendor/legacy22/data/domains/FULL24C/arrays.json').read_text(encoding='utf8')),
        production_gate_passed=False, scientific_dispatch=0,
        owner_or_optimizer_transferred=False,
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    folder = ROOT / 'outputs/maize_scope_revision_candidate/transport_v4'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/'manifest.json'
    archive_path = folder/'candidate_payload.tar'
    if manifest_path.exists() or archive_path.exists():
        raise RuntimeError('EXISTING_TRANSPORT_REVIEW_REQUIRED_NO_OVERWRITE')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    with tarfile.open(archive_path, 'w:') as tar:
        for row in files:
            if row['transport'] == 'payload':
                tar.add(CANDIDATE/row['path'], arcname=row['path'], recursive=False)
        tar.add(manifest_path, arcname='deployment/candidate_manifest.json', recursive=False)
    receipt = dict(passed=True,manifest_sha256=sha(manifest_path),archive_sha256=sha(archive_path),
        files=len(files), payload_files=sum(r['transport']=='payload' for r in files),
        byte_compatibility_verified=True, dualhost_pending=True, production_gate_passed=False)
    (folder/'prepared.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
