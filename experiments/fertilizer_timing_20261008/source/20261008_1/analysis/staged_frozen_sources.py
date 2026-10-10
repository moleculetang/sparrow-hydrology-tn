"""Private, partial, hash-bound physical replay inputs. Never admits evaluation.

These copies do not transfer training ownership or replace the terminal42 import.
The complete44 final admission must still verify its own actual return receipt.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(jid):
    from d29_platform.runtime import sha
    from diagnose_approved_waveforms import accepted_point
    proof_path = ROOT/'outputs/staged_frozen_import/complete.json'
    proof = json.loads(proof_path.read_text(encoding='utf8'))
    if (proof.get('passed') is not True or proof.get('partial_only') is not True or
            proof.get('evaluation_admitted') is not False or
            proof.get('training_ownership_changed') is not False):
        raise RuntimeError('PARTIAL_PHYSICAL_ONLY_IMPORT_REQUIRED')
    target = Path(proof['target']).resolve()
    if target != (ROOT/'outputs/staged_frozen_import/data').resolve():
        raise RuntimeError('EXACT_PRIVATE_STAGED_TARGET_REQUIRED')
    manifest_path = target/'manifest.json'
    if sha(manifest_path) != proof['manifest_sha256']:
        raise RuntimeError('STAGED_MANIFEST_CHANGED')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
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
