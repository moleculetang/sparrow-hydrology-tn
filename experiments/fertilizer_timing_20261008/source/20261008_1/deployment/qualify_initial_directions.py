"""Merge five admissible families; the off-area diagnostic is not admission."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json


def main():
    original=ROOT/'evidence/full_initial_feedback.json'
    restricted=ROOT/'evidence/supported_zero_initial_feedback.json'
    support=ROOT/'evidence/initial_direction_support_failure_and_fix.json'
    snapshot=ROOT/'evidence/source_snapshots/initial_feedback_physical_prefix.json'
    r=json.loads((snapshot if snapshot.exists() else original).read_text(encoding='utf8'))
    s=json.loads(restricted.read_text(encoding='utf8')) if restricted.exists() else None
    f=json.loads(support.read_text(encoding='utf8'))
    positive={c['state']:c for c in r['cases'] if c['state'] in ('P','Sa','Sp')}
    valid=len(positive)==3 and all(c['result']['passed'] and c['result'].get('branch_consistent_pairs') for c in positive.values())
    valid=bool(valid and s and s['passed'] and {c['state'] for c in s['cases']}=={'N_available','L'}
        and f['passed'] and not any(f['positive_pool_off_area_cells'].values())
        and all(c['unsupported_perturbed_cells']==0 for c in s['cases']))
    if valid and not snapshot.exists():
        # Preserve the exact original prefix bytes before the wider diagnostic
        # appends further cases. The original live receipt is never rewritten.
        snapshot.write_bytes(original.read_bytes())
    if valid:
        for rel,h in r['identities'].items():
            if sha(ROOT/rel)!=h:raise RuntimeError('POSITIVE_INITIAL_PROOF_IDENTITY_CHANGED')
        for rel,h in s['identities'].items():
            if r['identities'].get(rel)!=h:raise RuntimeError('INITIAL_DIRECTION_PROOF_MODELS_DIFFER')
    receipt=dict(passed=valid,scope='full1961_2024_five_admissible_initial_family_directions_not_every_cell_coordinate',
        cases=[positive[k] for k in ('P','Sa','Sp') if k in positive]+(s['cases'] if s else []),
        positive_prefix_snapshot_sha256=sha(snapshot) if snapshot.exists() else None,
        positive_prefix_path='evidence/source_snapshots/initial_feedback_physical_prefix.json',
        zero_pool_proof_sha256=sha(restricted) if restricted.exists() else None,
        support_failure_and_fix_sha256=sha(support),
        original_off_area_fixture_is_admission=False,source_multiplier=1.,formal_gate_issued=False,
        identities=r['identities'])
    write_json(ROOT/'evidence/admissible_initial_feedback.json',receipt)
    print(json.dumps(dict(passed=valid,positive_families=list(positive),zero_pool_support_passed=bool(s and s['passed']))))
    return 0 if valid else 2


if __name__=='__main__':sys.exit(main())
