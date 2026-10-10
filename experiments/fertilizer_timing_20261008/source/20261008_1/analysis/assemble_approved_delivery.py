"""Assemble actual two-host frozen results; never invent a terminal certificate.

The 42 remote paths must come from the actual replacement parent's verified
private import. The two local baselines retain their registered owner. This
receipt admits frozen evaluation, not completion of the whole experiment or
of the later serialized annual/month/day allocation delivery.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'analysis'), str(ROOT/'deployment')]
from d29_platform.runtime import sha
from approved_scope_policy import load_qualified
from approved_scope_return import MANIFEST, validate_manifest
from approved_epoch_delivery import validate_rows, validate_delivery
from approved_parent_lineage import current


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def regular(path, parent):
    path=Path(path).absolute()
    parent=Path(parent).resolve()
    if not path.resolve().is_relative_to(parent):
        raise RuntimeError('EXACT_PRIVATE_RESULT_NAMESPACE_REQUIRED')
    for entry in [path, *path.parents]:
        if entry==parent.parent:break
        if entry.is_symlink():raise RuntimeError('RESULT_LINK_REJECTED')
    if not path.is_file():raise RuntimeError('MISSING_REGULAR_RESULT_FILE '+str(path))
    return path


def validate_frozen(folder, job, gate, support):
    """Check frozen parameter, predictions and all 66 ledger artifacts by bytes."""
    folder=Path(folder)
    status=read(regular(folder/'status.json', folder))
    audit_path=regular(folder/'independent_audit.json', folder)
    freeze_path=regular(folder/'prediction_freeze.json', folder)
    audit=read(audit_path);freeze=read(freeze_path)
    if status.get('audit_completed') is not True or status.get('numerically_sufficient') is not True:
        raise RuntimeError('ACTUAL_PATH_AUDIT_NOT_FINAL '+job['id'])
    if (audit.get('job')!=job['id'] or freeze.get('job')!=job or
            audit.get('numerically_sufficient') is not True or
            freeze.get('numerically_sufficient') is not True):
        raise RuntimeError('EXACT_SUFFICIENT_JOB_AND_FREEZE_REQUIRED '+job['id'])
    for proof in [audit,freeze]:
        if proof.get('source_multiplier')!=1. or proof.get('formal_gate_sha256')!=gate:
            raise RuntimeError('FROZEN_MASS_OR_EPOCH_CHANGED '+job['id'])
    needed=('objective_passed','physical_passed','gradient_recovery_passed','derivative_passed')
    if (any(audit.get(k) is not True for k in needed) or
            not isinstance(audit.get('projected_gradient'), (int,float)) or
            not 0<=audit['projected_gradient']<=1e-5 or
            audit.get('measured_feasible_descent') is not False):
        raise RuntimeError('COMPLETE_NUMERICAL_QUALIFICATION_REQUIRED '+job['id'])
    if (audit.get('full_history')!=[1961,2024] or audit.get('reaches')!=230 or
            freeze.get('station_count')!=len(support['station_keys']) or
            freeze.get('simulation_reaches')!=230 or freeze.get('history_start')!=1961 or
            freeze.get('no_evaluation_outcomes_read') is not True or
            freeze.get('evaluation_labels_used') is not False):
        raise RuntimeError('FULL_HISTORY_SUPPORT_AND_LABEL_FREEZE_REQUIRED '+job['id'])
    checks=audit.get('checks',[])
    contract=read(regular(folder/'training_contract.json', folder))
    if contract.get('job')!=job:
        raise RuntimeError('EXACT_TRAINING_JOB_CONTRACT_REQUIRED '+job['id'])
    names=contract['names']
    if (not checks or len(checks)!=len(names) or
            len({c['coordinate'] for c in checks})!=len(names) or
            {c['coordinate'] for c in checks}!=set(names) or
            any(c.get('derivative_qualified') is not True or c.get('measured_feasible_descent') is not False for c in checks)):
        raise RuntimeError('EXACT_FULL_COORDINATE_AUDIT_REQUIRED '+job['id'])
    if sha(audit_path)!=freeze.get('audit_sha256'):
        raise RuntimeError('AUDIT_BYTES_CHANGED_AFTER_FREEZE '+job['id'])
    if audit.get('parameter_sha256')!=freeze.get('parameter_sha256'):
        raise RuntimeError('AUDIT_FREEZE_PARAMETER_MISMATCH '+job['id'])
    files={
        'independent_audit.json':sha(audit_path),
        'prediction_freeze.json':sha(freeze_path),
        'status.json':sha(folder/'status.json'),
        'training_contract.json':sha(folder/'training_contract.json'),
        'frozen_parameters.npy':freeze['parameter_sha256'],
        'frozen_station_days.parquet':freeze['days_sha256'],
        'frozen_station_months.parquet':freeze['months_sha256'],
        'frozen_local_kg_day.npy':freeze['local_sha256'],
        'physical_ledger.json':freeze['physical_ledger_receipt_sha256'],
    }
    artifacts=freeze.get('physical_artifact_sha256',{})
    expected={'full_history_physical_source_ledger.npz','station_source_ledger.npz'}
    if len(artifacts)!=66 or not expected<=set(artifacts):
        raise RuntimeError('ALL_66_PHYSICAL_SOURCE_PLANT_ARTIFACTS_REQUIRED '+job['id'])
    physical=read(regular(folder/'physical_ledger.json', folder))
    years=physical.get('year_ledger',[])
    plant_files={r.get('plant_activity',{}).get('reach_land_file') for r in years}
    if (physical.get('passed') is not True or len(years)!=64 or
            sorted(r['year'] for r in years)!=list(range(1961,2025)) or
            len(plant_files)!=64 or set(artifacts)!=expected|plant_files):
        raise RuntimeError('EXACT_64_YEAR_PLANT_LEDGER_REQUIRED '+job['id'])
    # Tie the physical ledger to the actual little-endian float64 vector,
    # rather than relying only on its file-level digest.
    import hashlib
    import numpy as np
    x=np.load(regular(folder/'frozen_parameters.npy', folder),allow_pickle=False)
    if x.shape!=(len(names),) or not np.isfinite(x).all():
        raise RuntimeError('FINITE_COMPLETE_FROZEN_PARAMETER_VECTOR_REQUIRED')
    digest=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
    if physical.get('parameter_vector_sha256')!=digest:
        raise RuntimeError('PHYSICAL_PARAMETER_VECTOR_CHANGED '+job['id'])
    for r in years:
        p=r['plant_activity']
        if artifacts[p['reach_land_file']]!=p['reach_land_sha256']:
            raise RuntimeError('PLANT_LEDGER_HASH_IDENTITY_MISMATCH')
    if (artifacts['full_history_physical_source_ledger.npz']!=physical.get('full_history_ledger_sha256') or
            artifacts['station_source_ledger.npz']!=physical.get('station_source_ledger_sha256')):
        raise RuntimeError('SOURCE_LEDGER_HASH_IDENTITY_MISMATCH')
    files.update(artifacts)
    for rel,digest in files.items():
        if sha(regular(folder/rel, folder))!=digest:
            raise RuntimeError('FROZEN_ARTIFACT_BYTES_CHANGED '+job['id']+'/'+rel)
    # Every claimed dependency must be a member of its sealed scientific gate.
    if contract.get('formal_gate_sha256') is not None and contract['formal_gate_sha256']!=gate:
        raise RuntimeError('TRAINING_CONTRACT_GATE_CHANGED')
    return files


def assemble(root, return_receipt):
    root=Path(root).resolve();candidate=root.with_name('20261008_1_maize_scope_v2')
    scope=load_qualified(root)
    jobs=read(root/'config/jobs.json');ownership_path=root/'deployment/path_ownership.json'
    ownership=read(ownership_path)
    if set(ownership['owners'])!={j['id'] for j in jobs}:
        raise RuntimeError('EXACT_REGISTERED_OWNERSHIP_REQUIRED')
    local_ids={j for j,h in ownership['owners'].items() if h=='local'}
    if local_ids!={'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'}:
        raise RuntimeError('REGISTERED_TWO_LOCAL_BASELINES_REQUIRED')
    remote_ids=sorted(j for j,h in ownership['owners'].items() if h=='workstation')
    if len(remote_ids)!=42:raise RuntimeError('EXACT_REMOTE42REQUIRED')
    handoff=current(root)
    returned=read(regular(return_receipt, root/'outputs/approved_scope_terminal_return'))
    if (not handoff.get('passed') or returned.get('transport_passed') is not True or
            returned.get('remote_owner')!=handoff['new_owner'] or
            returned.get('scope')!=scope or returned.get('ownership_sha256')!=sha(ownership_path)):
        raise RuntimeError('ACTUAL_CURRENT_PARENT_TWO_CONTRACT_IMPORT_REQUIRED')
    imported=returned['imported'];packed=returned['returned']
    target=Path(imported['target']).resolve()
    expected=root/'outputs'/('imported_workstation_approved_campaign_'+packed['manifest_sha256'][:20])
    if target!=expected or imported.get('passed') is not True:
        raise RuntimeError('EXACT_PRIVATE_TERMINAL_IMPORT_REQUIRED')
    manifest_path=regular(target/MANIFEST,target)
    if sha(manifest_path)!=packed['manifest_sha256'] or imported['manifest_sha256']!=packed['manifest_sha256']:
        raise RuntimeError('ACTUAL_IMPORTED_MANIFEST_CHANGED')
    manifest=read(manifest_path)
    indexed=validate_manifest(manifest, remote_ids, scope, sha(ownership_path))
    if manifest.get('expected_owner')!=handoff['new_owner'] or manifest.get('terminal_phase')!='all_owned_paths_frozen':
        raise RuntimeError('ACTUAL_PARENT_NOT_ALL_PATHS_FROZEN')
    archive=Path(return_receipt).parent/Path(packed['archive']).name
    if sha(regular(archive,Path(return_receipt).parent))!=packed['sha256'] or imported['archive_sha256']!=packed['sha256']:
        raise RuntimeError('ACTUAL_TERMINAL_ARCHIVE_CHANGED')
    support=read(root/'evidence/expected_station_support.json')
    records=[]
    for job in jobs:
        jid=job['id'];epoch=scope['epochs'][jid];gate=scope['gates'][jid]
        folder=(root if jid in local_ids else target)/'outputs/jobs'/jid
        bound=validate_frozen(folder,job,gate,support)
        if jid not in local_ids:
            for rel,digest in bound.items():
                member='outputs/jobs/'+jid+'/'+rel
                if member not in indexed or indexed[member]['sha256']!=digest:
                    raise RuntimeError('FROZEN_FILE_NOT_FROM_EXACT_TERMINAL_IMPORT '+member)
        records.append(dict(job=jid,epoch=epoch,formal_gate_sha256=gate,
            source_multiplier=1.,selection='fixed_contract_epoch_without_score_selection',
            evaluation_labels_used=False,folder=str(folder),host_role=ownership['owners'][jid],
            audit_sha256=bound['independent_audit.json'],freeze_sha256=bound['prediction_freeze.json'],
            bound_files=bound))
    result=dict(version=1,records=records,selection_rule='fixed_contract_epoch_without_score_selection',
        expanded_history_sha256=sha(candidate/'outputs/maize_scope_history/complete.json'),
        ownership_sha256=sha(ownership_path),actual_return_receipt=str(Path(return_receipt).resolve()),
        actual_return_sha256=sha(return_receipt),terminal_manifest_sha256=sha(manifest_path),
        source_multiplier=1.,evaluation_labels_read=False,frozen_evaluation_admitted=True,
        entire_experiment_completed=False,input_allocation_delivery_complete=False)
    validate_rows(jobs,records,sha(root/'evidence/formal_gate.json'),sha(candidate/'evidence/formal_gate.json'))
    validate_delivery(root,result)
    return result


def publish(root, result):
    """Create once; an identical retry verifies bytes, never replaces a result."""
    path=Path(root)/'evidence/approved_epoch_delivery.json'
    raw=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf8')
    if path.exists():
        if path.read_bytes()!=raw:raise RuntimeError('EXISTING_IMMUTABLE_DELIVERY_CONFLICT')
    else:
        with path.open('xb') as stream:stream.write(raw)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--return-receipt',type=Path,required=True)
    args=p.parse_args();result=assemble(ROOT,args.return_receipt);path=publish(ROOT,result)
    print(json.dumps(dict(passed=True,receipt=str(path),paths=len(result['records']),
        frozen_evaluation_admitted=True,entire_experiment_completed=False)))


if __name__=='__main__':main()
