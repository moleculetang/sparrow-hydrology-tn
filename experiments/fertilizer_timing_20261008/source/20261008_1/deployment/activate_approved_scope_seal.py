"""Publish a verified corrected gate in the independent Linux candidate only."""
import hashlib
import json
import os
import sys
from pathlib import Path,PurePosixPath

ROOT=Path('/home/pc/TWY/SPARROW/Test/20261008_1_maize_scope_v2')
ORIGINAL=ROOT.with_name('20261008_1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(source_path,expected_hash):
    source=Path(source_path).resolve()
    if os.name!='posix' or source.parent!=ROOT/'transfer' or sha(source)!=expected_hash:
        raise RuntimeError('EXACT_CANDIDATE_GATE_TRANSPORT_REQUIRED')
    gate=json.loads(source.read_text(encoding='utf8'))
    needed=('full_history_derivatives','source_network_ledger','independent_objective','source_multiplier_one',
        'training_prediction_isolation','checkpoint_restore','dual_host_consistency','known_error_auditor','immutable_input_identity')
    if (not gate.get('passed') or not gate.get('sealed') or gate.get('pending') or gate.get('errors')
            or any(not gate.get('requirements',{}).get(k) for k in needed)
            or gate.get('source_multiplier')!=1. or gate.get('K_coordinate_count')!=24):
        raise RuntimeError('ACTUAL_COMPLETE_APPROVED_K_GATE_REQUIRED')
    if sha(ORIGINAL/'evidence/formal_gate.json')!=gate['original_gate_sha256']:
        raise RuntimeError('ORIGINAL_GATE_CHANGED')
    if sha(ROOT/'deployment/candidate_manifest.json')!=gate['candidate_manifest_sha256']:
        raise RuntimeError('CANDIDATE_MANIFEST_CHANGED')
    for rel,digest in gate['identities'].items():
        path=PurePosixPath(rel);p=(ROOT/rel).resolve()
        if path.is_absolute() or '..' in path.parts or not p.is_relative_to(ROOT) or sha(p)!=digest:
            raise RuntimeError('APPROVED_SOURCE_INPUT_CHANGED '+rel)
    config=json.loads((ROOT/'config/timing_evidence.json').read_text(encoding='utf8'))
    maize=next(s for s in config['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    if maize['basal_delta_bounds']!=[-.1,0.]:raise RuntimeError('APPROVED_MAIZE_RANGE_REQUIRED')
    status=json.loads((ROOT/'outputs/scope_remote_point_launcher/status.json').read_text(encoding='utf8'))
    actual=json.loads((ROOT/'outputs/scope_remote_point/complete.json').read_text(encoding='utf8'))
    if (status.get('exit_code')!=0 or status.get('phase')!='qualification_finished_dispatch_restored'
            or not actual.get('passed') or not actual.get('fresh_canonical_restore_passed')
            or any(r['evaluation_year_rows_returned'] for r in actual['constructor_reads'])):
        raise RuntimeError('ACTUAL_FINISHED_STRICT_REMOTE_QUALIFICATION_REQUIRED')
    for name,digest in actual['array_sha256'].items():
        if sha(ROOT/'outputs/scope_remote_point'/(name+'.npy'))!=digest:raise RuntimeError('ACTUAL_REMOTE_ARRAY_CHANGED')
    h1=ROOT.parent/'20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'
    arrays=json.loads((ROOT/'vendor/legacy22/data/domains/FULL24C/arrays.json').read_text(encoding='utf8'))
    if len(arrays)!=26:raise RuntimeError('FULL26_H1_ARRAY_CONTRACT_REQUIRED')
    for name,spec in arrays.items():
        p=(h1/spec['file']).resolve()
        if not p.is_relative_to(h1) or sha(p)!=spec['sha256']:raise RuntimeError('ACTUAL_H1_ARRAY_CHANGED '+name)
    target=ROOT/'evidence/formal_gate.json';receipt_path=ROOT/'evidence/remote_final_seal.json'
    if target.exists():
        if sha(target)!=expected_hash:raise RuntimeError('EXISTING_GATE_CONFLICT_NO_OVERWRITE')
    else:
        with target.open('xb') as f:f.write(source.read_bytes())
    receipt=dict(passed=True,scope='independent_corrected_K_actual_remote_source_input_H1_seal',
        gate_sha256=expected_hash,h1_arrays_verified=len(arrays),verified_files=len(gate['identities']),
        candidate_manifest_sha256=gate['candidate_manifest_sha256'],
        actual_remote_oracle_sha256=sha(ROOT/'outputs/scope_remote_point/complete.json'),
        source_multiplier=1.,original_scientific_sources_changed=False,scientific_dispatch=0)
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding='utf8'))!=receipt:raise RuntimeError('EXISTING_REMOTE_SEAL_CONFLICT')
    else:
        with receipt_path.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main(*sys.argv[1:])
