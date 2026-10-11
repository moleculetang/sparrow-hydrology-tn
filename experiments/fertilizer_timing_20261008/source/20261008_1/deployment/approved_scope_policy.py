"""Route the fixed matrix to30original non-K and14corrected K, never by score."""
import json
from pathlib import Path

HOLD_REASON='narrow_K_epoch_requires_approved_range_revision'


def plan(jobs, original_gate, corrected_gate):
    if len(jobs)!=44 or len({j['id'] for j in jobs})!=44:
        raise RuntimeError('UNIQUE_EXACT44_REGISTERED_JOBS_REQUIRED')
    if not original_gate or not corrected_gate or original_gate==corrected_gate:
        raise RuntimeError('DISTINCT_ORIGINAL_AND_APPROVED_GATE_REQUIRED')
    groups={j['id']:('approved_maize_K' if j['calendar']=='K' else 'original_nonK') for j in jobs}
    if any(j['calendar'] not in ('C0','C1','C2','K') or j['fixed_log_source_multiplier']!=0. for j in jobs):
        raise RuntimeError('REGISTERED_FIXED_MASS_CALENDAR_REQUIRED')
    if list(groups.values()).count('approved_maize_K')!=14 or list(groups.values()).count('original_nonK')!=30:
        raise RuntimeError('EXACT30NONK_AND14APPROVED_K_REQUIRED')
    return dict(epochs=groups,gates={j:corrected_gate if e=='approved_maize_K' else original_gate for j,e in groups.items()})


def route(root,jid,scope):
    root=Path(root)
    if jid not in scope['epochs']:raise RuntimeError('UNREGISTERED_JOB_ID')
    return root.with_name('20261008_1_maize_scope_v2') if scope['epochs'][jid]=='approved_maize_K' else root


def qualify_record(record,scope):
    jid=record['job']
    if jid not in scope['epochs']:raise RuntimeError('UNREGISTERED_ADOPTED_JOB')
    epoch=scope['epochs'][jid];gate=scope['gates'][jid]
    # Only old non-K children can lack the new provenance fields.
    if 'epoch' not in record:
        if epoch!='original_nonK':raise RuntimeError('LIVE_OLD_K_CHILD_CANNOT_CHANGE_CONTRACT')
    elif record['epoch']!=epoch or record.get('job_gate_sha256')!=gate:
        raise RuntimeError('ADOPTED_SCIENTIFIC_EPOCH_OR_GATE_CHANGED')
    return epoch,gate


def corrected_reviews(reviews,scope):
    result=[]
    for record in reviews:
        jid=record['job']
        if jid not in scope['epochs']:raise RuntimeError('UNREGISTERED_REVIEW_JOB')
        if record.get('reason')==HOLD_REASON:
            if scope['epochs'][jid]!='approved_maize_K':raise RuntimeError('NONK_CANNOT_RELEASE_K_SCOPE_HOLD')
            continue
        result.append(dict(record))
    return result


def load_qualified(root):
    """Actual files only; a pending gate cannot construct dispatch routing."""
    root=Path(root)
    from d29_platform.runtime import sha
    corrected=root.with_name('20261008_1_maize_scope_v2')
    original_path=root/'evidence/formal_gate.json';corrected_path=corrected/'evidence/formal_gate.json'
    original=json.loads(original_path.read_text(encoding='utf8'))
    revised=json.loads(corrected_path.read_text(encoding='utf8'))
    needed=('full_history_derivatives','source_network_ledger','independent_objective','source_multiplier_one',
            'training_prediction_isolation','checkpoint_restore','dual_host_consistency','known_error_auditor','immutable_input_identity')
    for gate,location in [(original,root),(revised,corrected)]:
        if not gate.get('passed') or not gate.get('sealed') or any(not gate.get('requirements',{}).get(k) for k in needed):
            raise RuntimeError('TWO_ACTUAL_COMPLETE_SCIENTIFIC_GATES_REQUIRED')
        if gate.get('source_multiplier')!=1. or not gate.get('identities'):
            raise RuntimeError('FIXED_MASS_DEPENDENCY_IDENTITY_REQUIRED')
        for rel,digest in gate['identities'].items():
            if sha(location/rel)!=digest:raise RuntimeError('FROZEN_EPOCH_DEPENDENCY_CHANGED '+rel)
    if revised.get('original_gate_sha256')!=sha(original_path):raise RuntimeError('REVISED_GATE_PARENT_CHANGED')
    config=json.loads((corrected/'config/timing_evidence.json').read_text(encoding='utf8'))
    maize=next(s for s in config['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    if maize['basal_delta_bounds']!=[-.1,0.]:raise RuntimeError('EXACT_APPROVED_MAIZE_RANGE_REQUIRED')
    remote=corrected/'evidence/remote_final_seal.json'
    if not remote.exists():raise RuntimeError('ACTUAL_REMOTE_APPROVED_SEAL_REQUIRED')
    q=json.loads(remote.read_text(encoding='utf8'))
    if not q.get('passed') or q['gate_sha256']!=sha(corrected_path) or q['h1_arrays_verified']!=26:
        raise RuntimeError('ACTUAL_REMOTE_APPROVED_SEAL_CHANGED')
    jobs=json.loads((root/'config/jobs.json').read_text(encoding='utf8'))
    if jobs!=json.loads((corrected/'config/jobs.json').read_text(encoding='utf8')):
        raise RuntimeError('TWO_EPOCH_MATRIX_CONTRACTS_DIFFER')
    return plan(jobs,sha(original_path),sha(corrected_path))
