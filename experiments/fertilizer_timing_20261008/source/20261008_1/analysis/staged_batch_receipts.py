"""Known additive batch receipts, never a glob that admits arbitrary results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(kind):
    if kind not in ('internal', 'bnf'):
        raise ValueError('DECLARED_STAGED_DIAGNOSTIC_KIND_REQUIRED')
    base = 'staged_internal_trace' if kind == 'internal' else 'staged_bnf_trace'
    rows = []
    from staged_frozen_sources import IMPORTS
    from d29_platform.runtime import sha
    for suffix, namespace in (('', 'staged_frozen_import'), ('_002', 'staged_frozen_import_002')):
        folder = ROOT/'outputs'/(base+suffix)
        if not folder.exists():
            if suffix: continue
            raise RuntimeError('ORIGINAL_SIX_STAGE_RECEIPT_REQUIRED')
        path = folder/'complete.json'
        if not path.exists():
            raise RuntimeError('LIVE_OR_PARTIAL_STAGED_BATCH_NOT_ADMISSIBLE')
        value = json.loads(path.read_text(encoding='utf8'))
        ids = [r['job'] for r in value['completed']]
        if (value.get('passed') is not True or value.get('partial_only') is not True
                or value.get('evaluation_admitted') is not False
                or value.get('full44_completed_claimed') is not False
                or value.get('training_ownership_changed') is not False
                or len(ids) != len(set(ids)) or set(ids) != IMPORTS[namespace]):
            raise RuntimeError('EXACT_COMPLETE_KNOWN_STAGE_BATCH_REQUIRED')
        worker_namespace = 'internal_trace_approved_staged_v1' if kind == 'internal' else 'bnf_trace_approved_staged_v1'
        for row in value['completed']:
            receipt = Path(row['complete'])
            expected = ROOT/'outputs/jobs'/(worker_namespace+'_'+row['job'])/'complete.json'
            if (receipt.is_symlink() or receipt.resolve() != expected.resolve()
                    or not receipt.is_file() or sha(receipt) != row['sha256']):
                raise RuntimeError('EXACT_KNOWN_STAGE_RECEIPT_BYTES_REQUIRED')
        rows.extend(value['completed'])
    ids = [r['job'] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError('DUPLICATE_PHYSICAL_EXECUTION_NOT_ADMISSIBLE')
    return rows
