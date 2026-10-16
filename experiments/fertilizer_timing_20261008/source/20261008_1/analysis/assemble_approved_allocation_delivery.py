"""Bind full-history input allocations to the actual 44 frozen fit points.

This is a separate deliverable from prediction admission. Each path retains its
fixed contract epoch and immutable source folder; there is no score selection.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'deployment')]
from d29_platform.runtime import sha
from approved_epoch_delivery import admitted_folders
from approved_scope_policy import load_qualified
from approved_scope_return import MANIFEST, validate_manifest
from assemble_approved_delivery import regular,read
from approved_parent_lineage import current


def validate_allocation(folder,job,gate,freeze,science):
    folder=Path(folder);science=Path(science);directory=folder/'input_allocation'
    a_path=regular(directory/'complete.json',folder)
    r_path=regular(directory/'independent_artifact_review.json',folder)
    allocation=read(a_path);review=read(r_path)
    expected={f'{kind}_{year}.parquet' for kind in ['budgets','profiles'] for year in range(1961,2025)}
    if (allocation.get('passed') is not True or allocation.get('job')!=job['id'] or
            allocation.get('calendar')!=job['calendar'] or allocation.get('source_multiplier')!=1. or
            allocation.get('formal_gate_sha256')!=gate or allocation.get('parameter_sha256')!=freeze['parameter_sha256'] or
            allocation.get('observation_data_read') is not False or allocation.get('BNF_and_deposition_unchanged') is not True or
            set(allocation.get('artifact_sha256',{}))!=expected):
        raise RuntimeError('EXACT_FROZEN_FULL_HISTORY_INPUT_ALLOCATION_REQUIRED '+job['id'])
    required_sources={'timing/inputs.py','timing/profiles.py','scripts/export_timing_allocations.py'}
    if set(allocation.get('identities',{}))!=required_sources:
        raise RuntimeError('EXACT_GENERATOR_IDENTITY_ACCOUNTING_REQUIRED')
    for relative,digest in allocation['identities'].items():
        if sha(regular(science/relative,science))!=digest:
            raise RuntimeError('ALLOCATION_GENERATOR_IDENTITY_CHANGED '+relative)
    if (review.get('passed') is not True or review.get('job')!=job['id'] or review.get('years')!=64 or
            review.get('reaches')!=230 or review.get('crops')!=21 or
            review.get('formal_gate_sha256')!=gate or review.get('parameter_sha256')!=freeze['parameter_sha256'] or
            review.get('no_TN_or_prediction_values_read') is not True or
            review.get('allocation_receipt_sha256')!=sha(a_path) or
            review.get('verifier_sha256')!=sha(regular(science/'analysis/verify_timing_allocation_artifacts.py',science)) or
            review.get('annual_source_sha256')!=sha(regular(science/'inputs/reference/agriculture_reference/annual_crop_source_reference.parquet',science))):
        raise RuntimeError('ACTUAL_CURRENT_INDEPENDENT_READBACK_REQUIRED '+job['id'])
    for receipt in [allocation,review]:
        records=receipt.get('records',[])
        if len(records)!=64 or sorted(r['year'] for r in records)!=list(range(1961,2025)):
            raise RuntimeError('EXACT64_UNIQUE_YEAR_RECORDS_REQUIRED')
    if any(r.get('crops')!=21 or r.get('records')!=230*21*2 for r in allocation['records']):
        raise RuntimeError('ALL_CROP_REACH_SOURCE_ALLOCATION_SUPPORT_REQUIRED')
    if any(r.get('rows')!=230*21*2 for r in review['records']):
        raise RuntimeError('ALL_CROP_REACH_SOURCE_READBACK_SUPPORT_REQUIRED')
    import math
    maxima=allocation.get('maxima_kg',{})
    if set(maxima)!={'annual','month_sum','reconstruction'} or any(not math.isfinite(v) or not 0<=v<=1e-6 for v in maxima.values()):
        raise RuntimeError('ABSOLUTE_ALLOCATION_MASS_TOLERANCE_REQUIRED')
    if any(set(r.get('maxima_kg',{}))!={'annual','month_sum'} or
        any(not math.isfinite(v) or not 0<=v<=1e-6 for v in r['maxima_kg'].values()) for r in review['records']):
        raise RuntimeError('ABSOLUTE_READBACK_MASS_TOLERANCE_REQUIRED')
    bound={'input_allocation/complete.json':sha(a_path),'input_allocation/independent_artifact_review.json':sha(r_path)}
    for name,digest in allocation['artifact_sha256'].items():
        if sha(regular(directory/name,folder))!=digest:
            raise RuntimeError('REVIEWED_ALLOCATION_BYTES_CHANGED '+job['id']+'/'+name)
        bound['input_allocation/'+name]=digest
    return bound


def assemble(root,return_receipt):
    root=Path(root).resolve();candidate=root.with_name('20261008_1_maize_scope_v2')
    freezes=admitted_folders(root);scope=load_qualified(root)
    jobs=read(root/'config/jobs.json');ownership_path=root/'deployment/path_ownership.json'
    from repartition_policy_v1 import read_ownership
    ownership=read_ownership(root)
    remote_ids=sorted(j for j,h in ownership['owners'].items() if h=='workstation')
    receipt_path=regular(return_receipt,root/'outputs/approved_allocation_terminal_return')
    returned=read(receipt_path)
    handoff=current(root)
    actual=returned['actual'];packed=actual['packed'];imported=returned['imported']
    if (returned.get('transport_passed') is not True or returned.get('scope')!=scope or
            returned.get('science_owner')!=handoff['new_owner'] or actual.get('science_owner')!=handoff['new_owner'] or
            actual.get('artifact_owner')!=returned.get('artifact_owner') or
            returned.get('ownership_sha256')!=sha(ownership_path) or
            set(actual.get('reviews',{}))!=set(remote_ids)):
        raise RuntimeError('ACTUAL_EXACT_APPROVED_REMOTE_ALLOCATION_RETURN_REQUIRED')
    status=actual['artifact_controller_status']
    if (status.get('owner')!=returned['artifact_owner'] or status.get('approved_scope')!=scope or
            status.get('phase')!='all_owned_allocations_independently_reviewed' or status.get('running') is not None or
            sorted(status.get('completed',[]))!=remote_ids or status.get('blocked')):
        raise RuntimeError('ACTUAL_ALLOCATION_TERMINAL_PARENT_REQUIRED')
    target=Path(imported['target']).resolve()
    if target!=root/'outputs'/('imported_workstation_approved_allocations_'+packed['manifest_sha256'][:20]) or imported.get('passed') is not True:
        raise RuntimeError('EXACT_IMMUTABLE_ALLOCATION_IMPORT_REQUIRED')
    manifest_path=regular(target/MANIFEST,target)
    if sha(manifest_path)!=packed['manifest_sha256'] or imported['manifest_sha256']!=packed['manifest_sha256']:
        raise RuntimeError('ACTUAL_ALLOCATION_IMPORT_MANIFEST_CHANGED')
    manifest=read(manifest_path)
    indexed=validate_manifest(manifest,remote_ids,scope,sha(ownership_path))
    if manifest.get('expected_owner')!=handoff['new_owner'] or manifest.get('terminal_phase')!='all_owned_paths_frozen':
        raise RuntimeError('EXACT_FULLY_FROZEN_SCIENTIFIC_IMPORT_REQUIRED')
    archive=regular(returned['archive'],receipt_path.parent)
    if sha(archive)!=packed['sha256'] or imported['archive_sha256']!=packed['sha256']:
        raise RuntimeError('ACTUAL_FINAL_ALLOCATION_ARCHIVE_CHANGED')
    records=[]
    for job in jobs:
        jid=job['id'];science=candidate if job['calendar']=='K' else root
        folder=(target if ownership['owners'][jid]=='workstation' else root)/'outputs/jobs'/jid
        freeze=read(freezes[jid]/'prediction_freeze.json')
        # The later allocation archive must preserve the first frozen point,
        # including its entire prediction identity, rather than merely its J.
        if sha(regular(folder/'prediction_freeze.json',folder))!=sha(freezes[jid]/'prediction_freeze.json'):
            raise RuntimeError('PREDICTION_FREEZE_CHANGED_BETWEEN_RETURNS '+jid)
        bound=validate_allocation(folder,job,scope['gates'][jid],freeze,science)
        if ownership['owners'][jid]=='workstation':
            for rel,digest in bound.items():
                name='outputs/jobs/'+jid+'/'+rel
                if name not in indexed or indexed[name]['sha256']!=digest:
                    raise RuntimeError('EXACT_IMPORTED_ALLOCATION_FILE_REQUIRED '+name)
            proof=actual['reviews'][jid]
            if proof['allocation_sha256']!=bound['input_allocation/complete.json'] or proof['review_sha256']!=bound['input_allocation/independent_artifact_review.json']:
                raise RuntimeError('REMOTE_LOCAL_READBACK_PROOF_CHANGED')
        records.append(dict(job=jid,epoch=scope['epochs'][jid],formal_gate_sha256=scope['gates'][jid],
          source_multiplier=1.,host_role=ownership['owners'][jid],folder=str(folder),
          prediction_freeze_sha256=sha(freezes[jid]/'prediction_freeze.json'),parameter_sha256=freeze['parameter_sha256'],
          bound_files=bound))
    if len(records)!=44 or len({r['job'] for r in records})!=44:
        raise RuntimeError('EXACT44_ALLOCATION_DELIVERY_REQUIRED')
    return dict(passed=True,records=records,years=[1961,2024],reaches=230,crops=21,
      allocation_artifacts=44*128,independent_readbacks=44,source_multiplier=1.,
      prediction_delivery_sha256=sha(root/'evidence/approved_epoch_delivery.json'),
      actual_allocation_return_sha256=sha(receipt_path),evaluation_labels_read=False,
      entire_experiment_completed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--return-receipt',type=Path,required=True);args=p.parse_args()
    result=assemble(ROOT,args.return_receipt)
    path=ROOT/'evidence/approved_allocation_delivery.json'
    raw=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf8')
    if path.exists():
        if path.read_bytes()!=raw:raise RuntimeError('EXISTING_IMMUTABLE_ALLOCATION_DELIVERY_CONFLICT')
    else:
        with path.open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(passed=True,paths=len(result['records']),allocation_artifacts=result['allocation_artifacts'],
      entire_experiment_completed=False)))


if __name__=='__main__':main()
