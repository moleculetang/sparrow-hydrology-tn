"""Private, partial, hash-bound physical replay inputs. Never admits evaluation.

These copies do not transfer training ownership or replace the terminal42 import.
The complete44 final admission must still verify its own actual return receipt.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Additive imports preserve the original six-path manifest and receipts.
# Only independently frozen, explicitly named non-K paths may enter here.
IMPORTS = {
    'staged_frozen_import': {'TIM_F23_C1_V0_s0', 'TIM_F23_C2_V0_s0',
        'TIM_F24_C1_V0_s0', 'TIM_F24_C2_V0_s0',
        'TIM_F23_C1_V1_s0', 'TIM_F23_C2_V1_s0'},
    'staged_frozen_import_002': {'TIM_F23_C0_V0_s0', 'TIM_F23_C0_V1_s0'},
}


def resolve(jid):
    from d29_platform.runtime import sha
    from diagnose_approved_waveforms import accepted_point
    selected = [name for name, ids in IMPORTS.items() if jid in ids]
    if len(selected) != 1:
        raise RuntimeError('EXPLICIT_UNIQUE_STAGED_NONK_ID_REQUIRED')
    namespace = selected[0]
    proof_path = ROOT/'outputs'/namespace/'complete.json'
    proof = json.loads(proof_path.read_text(encoding='utf8'))
    if (proof.get('passed') is not True or proof.get('partial_only') is not True or
            proof.get('evaluation_admitted') is not False or
            proof.get('training_ownership_changed') is not False):
        raise RuntimeError('PARTIAL_PHYSICAL_ONLY_IMPORT_REQUIRED')
    target = Path(proof['target']).resolve()
    if target != (ROOT/'outputs'/namespace/'data').resolve():
        raise RuntimeError('EXACT_PRIVATE_STAGED_TARGET_REQUIRED')
    manifest_path = target/'manifest.json'
    if sha(manifest_path) != proof['manifest_sha256']:
        raise RuntimeError('STAGED_MANIFEST_CHANGED')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    records = manifest['records']
    ids = [r['job']['id'] for r in records]
    if (len(ids) != len(set(ids)) or set(ids) != IMPORTS[namespace]
            or any(r['job']['calendar'] == 'K' for r in records)):
        raise RuntimeError('EXACT_IMMUTABLE_STAGED_BATCH_REQUIRED')
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    job = next(j for j in jobs if j['id']==jid)
    record = next(r for r in manifest['records'] if r['job']==job)
    folder = target/'outputs/jobs'/jid
    support = json.loads((ROOT/'evidence/expected_station_support.json').read_text(encoding='utf8'))
    _, _, bound = accepted_point(folder,job,support)
    if bound != record['bound_files']:
        raise RuntimeError('EXACT_FROZEN_BINDING_CHANGED')
    for rel,digest in record['extra_files'].items():
        path=(folder/rel).resolve()
        if not path.is_relative_to(folder.resolve()) or path.is_symlink() or sha(path)!=digest:
            raise RuntimeError('STAGED_EXTRA_BYTES_CHANGED')
    return folder
