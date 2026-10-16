"""Final 44 paths = 30 compatible original non-K + 14 approved-range K.

Selection is fixed by contract identity, never an NSE or training-score choice
between the narrow and expanded epochs. Actual imports remain immutable.
"""
import json
from pathlib import Path


def validate_rows(jobs, records, original_gate, corrected_gate):
    registered={j['id']:j for j in jobs}
    if len(jobs)!=44 or len(registered)!=44 or len(records)!=44:
        raise RuntimeError('EXACT44APPROVED_PATH_ACCOUNTING_REQUIRED')
    ids=[r['job'] for r in records]
    if len(set(ids))!=44 or set(ids)!=set(registered):
        raise RuntimeError('UNIQUE_EXACT_APPROVED_IDS_REQUIRED')
    if not corrected_gate or corrected_gate==original_gate:
        raise RuntimeError('DISTINCT_CORRECTED_SCIENTIFIC_GATE_REQUIRED')
    counts={'original_nonK':0,'approved_maize_K':0}
    for record in records:
        job=registered[record['job']]
        epoch='approved_maize_K' if job['calendar']=='K' else 'original_nonK'
        gate=corrected_gate if job['calendar']=='K' else original_gate
        if record.get('epoch')!=epoch or record.get('formal_gate_sha256')!=gate:
            raise RuntimeError('APPROVED_EPOCH_REQUIRED '+job['id'])
        if record.get('source_multiplier')!=1. or record.get('selection')!='fixed_contract_epoch_without_score_selection':
            raise RuntimeError('FIXED_MASS_AND_NON_SCORE_EPOCH_SELECTION_REQUIRED')
        if record.get('evaluation_labels_used') is not False:
            raise RuntimeError('EVALUATION_EPOCH_SELECTION_FORBIDDEN')
        counts[epoch]+=1
    if counts!={'original_nonK':30,'approved_maize_K':14}:
        raise RuntimeError('EXACT30PLUS14REQUIRED')
    return registered


def validate_delivery(root, delivery):
    """Validate an in-memory assembly before publishing its immutable receipt."""
    root=Path(root).resolve()
    from d29_platform.runtime import sha
    corrected=root.with_name('20261008_1_maize_scope_v2')
    original_gate=sha(root/'evidence/formal_gate.json')
    corrected_gate=sha(corrected/'evidence/formal_gate.json')
    jobs=json.loads((root/'config/jobs.json').read_text(encoding='utf8'))
    records=delivery.get('records',[])
    validate_rows(jobs,records,original_gate,corrected_gate)
    new_gate=json.loads((corrected/'evidence/formal_gate.json').read_text(encoding='utf8'))
    if not new_gate.get('passed') or not new_gate.get('sealed'):
        raise RuntimeError('CORRECTED_K_GATE_NOT_PASSED')
    required=['full_history_derivatives','source_network_ledger','independent_objective','source_multiplier_one',
              'training_prediction_isolation','checkpoint_restore','dual_host_consistency']
    if any(not new_gate.get('requirements',{}).get(name) for name in required):
        raise RuntimeError('CORRECTED_K_COMMON_REQUIREMENTS_INCOMPLETE')
    config=json.loads((corrected/'config/timing_evidence.json').read_text(encoding='utf8'))
    maize=next(s for s in config['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    if maize['basal_delta_bounds']!=[-.1,0.] or new_gate['identities'].get('config/timing_evidence.json')!=sha(corrected/'config/timing_evidence.json'):
        raise RuntimeError('APPROVED_MAIZE_BOUND_CONFIG_REQUIRED')
    qualification=corrected/'outputs/maize_scope_history/complete.json'
    q=json.loads(qualification.read_text(encoding='utf8'))
    if not q.get('passed') or q.get('config_sha256')!=sha(corrected/'config/timing_evidence.json'):
        raise RuntimeError('ACTUAL_EXPANDED_HISTORY_QUALIFICATION_REQUIRED')
    if delivery.get('expanded_history_sha256')!=sha(qualification):
        raise RuntimeError('EXPANDED_HISTORY_PROOF_CHANGED')
    if delivery.get('selection_rule')!='fixed_contract_epoch_without_score_selection':
        raise RuntimeError('REGISTERED_EPOCH_SELECTION_REQUIRED')
    roots=[root,corrected]
    result={}
    for record in records:
        source=Path(record['folder'])
        folder=source.resolve()
        if source.is_symlink() or not any(folder.is_relative_to(r/'outputs') for r in roots):
            raise RuntimeError('IMMUTABLE_LOCAL_RESULT_NAMESPACE_REQUIRED')
        audit_path=folder/'independent_audit.json'
        freeze_path=folder/'prediction_freeze.json'
        audit=json.loads(audit_path.read_text(encoding='utf8'))
        freeze=json.loads(freeze_path.read_text(encoding='utf8'))
        if not audit.get('numerically_sufficient') or not freeze.get('numerically_sufficient'):
            raise RuntimeError('FINAL_PATH_STILL_UNQUALIFIED '+record['job'])
        for proof in (audit,freeze):
            if proof.get('formal_gate_sha256')!=record['formal_gate_sha256'] or proof.get('source_multiplier')!=1.:
                raise RuntimeError('FINAL_FREEZE_EPOCH_MISMATCH '+record['job'])
        if sha(audit_path)!=freeze.get('audit_sha256') or audit['parameter_sha256']!=freeze['parameter_sha256']:
            raise RuntimeError('FINAL_AUDIT_FREEZE_POINT_MISMATCH')
        if freeze.get('job',{}).get('id')!=record['job']:
            raise RuntimeError('FINAL_FREEZE_JOB_MISMATCH')
        if record.get('audit_sha256')!=sha(audit_path) or record.get('freeze_sha256')!=sha(freeze_path):
            raise RuntimeError('FINAL_DELIVERY_RECEIPT_BYTES_CHANGED')
        result[record['job']]=folder
    return result


def admitted_folders(root):
    root=Path(root).resolve()
    path=root/'evidence/approved_epoch_delivery.json'
    if not path.exists():raise RuntimeError('FINAL_CORRECTED44DELIVERY_NOT_READY')
    return validate_delivery(root, json.loads(path.read_text(encoding='utf8')))
