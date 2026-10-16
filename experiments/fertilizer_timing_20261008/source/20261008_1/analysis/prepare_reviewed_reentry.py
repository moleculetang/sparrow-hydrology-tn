"""Archive a qualified stopped point and create a fresh canonical solver epoch.

Only the original host may use this entry, after its campaign controller has
actually exited. It never edits sealed science or reencodes an old workspace.
No training is started here; a normal guarded train lease is required later.
"""
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis'),str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
import numpy as np
from d29_platform.runtime import write_json,sha
from native_bound_adapter_v2 import CanonicalLBFGSB,canonical_bounds
from continuation_review_policy import review
from timing.projected_gradient import original_point


def create_epoch(saved,contract,point):
    """Create new memory at the reviewed point; retain cumulative fit counters."""
    if saved['status']!='numerical_review_required':raise RuntimeError('REVIEW_REQUIRED_NATIVE_STOP_REQUIRED')
    if saved['contract_sha256']!=contract['contract_sha256']:raise RuntimeError('CHECKPOINT_CONTRACT_MISMATCH')
    if saved['backend']!=contract['backend']:raise RuntimeError('CHECKPOINT_BACKEND_MISMATCH')
    point=np.asarray(point,dtype=np.float64);scale=np.asarray(contract['scale'],dtype=np.float64)
    if point.shape!=scale.shape or not np.isfinite(point).all() or np.any(scale<=0) or not np.isfinite(scale).all():
        raise RuntimeError('VALID_POINT_AND_SCALE_REQUIRED')
    if saved['best'] is None or not np.array_equal(saved['best']['x'],point):raise RuntimeError('BEST_POINT_CHECKPOINT_MISMATCH')
    if saved.get('source_multiplier')!=1. or contract.get('source_multiplier')!=1.:raise RuntimeError('SOURCE_MULTIPLIER_NOT_ONE')
    b=np.array([[float('-inf') if lo is None else lo,float('inf') if hi is None else hi]
                for lo,hi in canonical_bounds(contract['bounds'])],dtype=np.float64)
    native=CanonicalLBFGSB(point/scale,(b/scale[:,None]).tolist(),stage=1,maxiter=np.iinfo(np.int32).max)
    if not np.array_equal(original_point(native.s['x'],scale,b),point):
        raise RuntimeError('SAME_POINT_SCALED_ROUNDTRIP_NOT_EXACT')
    native.s['pgtol']=min(native.s['pgtol'],float(scale.min())*1e-5)
    return dict(saved,solver=native.s,status='safe_checkpoint_paused')


def prepare(jid):
    from train_job import validate_admission,backend,save_pickle
    from job_lease import require
    from host_resources import registry,registry_root
    from controller_lock import acquire
    _,gate_hash=validate_admission();lease=require(jid,'reentry_prepare')
    if lease['threads']!=1:raise RuntimeError('SINGLE_THREAD_PREPARATION_REQUIRED')
    role='local' if os.name=='nt' else 'workstation'
    controller=json.loads((ROOT/'outputs'/('campaign_'+role)/'owner.json').read_text())
    if registry.identity(controller['pid'])==controller['created']:
        raise RuntimeError('ORIGINAL_HOST_CAMPAIGN_CONTROLLER_STILL_LIVE')
    jobs=json.loads((ROOT/'config/jobs.json').read_text());job=next(j for j in jobs if j['id']==jid)
    owners=json.loads((ROOT/'deployment/path_ownership.json').read_text())['owners']
    if owners[jid]!=role:raise RuntimeError('ORIGINAL_MACHINE_OWNERSHIP_REQUIRED')
    folder=ROOT/'outputs/jobs'/jid;lock=acquire(folder/'reentry_prepare.lock')
    owner=json.loads((folder/'owner.json').read_text());contract=json.loads((folder/'training_contract.json').read_text())
    if owner['host']!=platform.node() or contract['host']!=platform.node():raise RuntimeError('ORIGINAL_HOST_REQUIRED')
    if registry.identity(owner['pid'])==owner['created']:raise RuntimeError('OLD_JOB_OWNER_STILL_LIVE')
    for token,item in registry.status(root=registry_root())['leases'].items():
        if token!=lease['token'] and item['project']=='20261008_1' and item['job'].endswith('/'+jid):
            raise RuntimeError('OTHER_JOB_STAGE_STILL_HAS_LEASE')
    if backend()!=contract['backend']:raise RuntimeError('BACKEND_CHANGED_NO_REENTRY')
    rawcontract={k:v for k,v in contract.items() if k!='contract_sha256'}
    if hashlib.sha256(json.dumps(rawcontract,sort_keys=True).encode()).hexdigest()!=contract['contract_sha256']:
        raise RuntimeError('TRAINING_CONTRACT_DIGEST_CHANGED')
    if contract['job']!=job or contract['formal_gate_sha256']!=gate_hash:raise RuntimeError('JOB_OR_GATE_CHANGED')
    point_hash=sha(folder/'best.npy');probe=folder/'continuation_diagnostics'/point_hash/'complete.json'
    diagnostic=json.loads(probe.read_text());audit=json.loads((folder/'independent_audit.json').read_text())
    if diagnostic.get('source_sha256')!=sha(ROOT/'analysis/probe_same_point_descent.py'):
        raise RuntimeError('DESCENT_PROBE_SOURCE_NOT_CURRENT')
    if not diagnostic.get('passed') or diagnostic.get('new_solver_started') is not False:
        raise RuntimeError('ACTUAL_PASSED_READ_ONLY_DESCENT_PROBE_REQUIRED')
    for rel,digest in diagnostic['fixed_scientific_artifacts'].items():
        p=(folder/rel).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p)!=digest:raise RuntimeError('PROBED_ARTIFACT_CHANGED '+rel)
    historypath=folder/'continuation_history.json'
    history=json.loads(historypath.read_text()) if historypath.exists() else dict(attempts=[])
    qualification=review(audit,contract,diagnostic['witness'],history['attempts'])
    if not qualification['qualified_for_reviewed_same_point_reentry']:raise RuntimeError(str(qualification['reasons']))
    policy=json.loads((ROOT/'evidence/continuation_review_policy_known_errors.json').read_text())
    if not policy.get('passed') or policy['source_sha256']!=sha(ROOT/'analysis/continuation_review_policy.py'):
        raise RuntimeError('CURRENT_REENTRY_POLICY_CONTROLS_REQUIRED')
    # These are trusted original-host native checkpoints, not imported pickle.
    saved=pickle.loads((folder/'optimizer.pkl').read_bytes());point=np.load(folder/'best.npy',allow_pickle=False)
    replacement=create_epoch(saved,contract,point)
    archive=folder/'reentry_epochs'/f"{len(history['attempts'])+1:03d}_{point_hash}"
    if archive.exists():raise RuntimeError('PREVIOUS_PREPARATION_EXISTS_INSPECT_JOURNAL_NO_BLIND_RETRY')
    archive.mkdir(parents=True)
    files=['optimizer.pkl','status.json','best.npy','best_gradient.npy','best_full_parameters.npy','best_management.npy',
        'training_contract.json','independent_audit.json','solution_audit_progress.json','audit_current_probe.json',
        'audited_gradient.npy','audited_training_prediction.npy','physical_ledger.json','full_history_physical_source_ledger.npz',
        'station_source_ledger.npz','same_point_diagnostic_request.json','prediction_freeze.json']
    paths=[folder/rel for rel in files if (folder/rel).is_file()]
    if (folder/'plant_activity_yearly').exists():paths.extend(p for p in (folder/'plant_activity_yearly').rglob('*') if p.is_file())
    manifest={}
    for p in paths:
        if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(folder.resolve()):
            raise RuntimeError('REGULAR_ORIGINAL_JOB_ARTIFACTS_REQUIRED')
        rel=p.relative_to(folder).as_posix();dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(p);shutil.copy2(p,dest)
        if sha(p)!=digest or sha(dest)!=digest:raise RuntimeError('ARCHIVE_HASH_MISMATCH '+rel)
        manifest[rel]=dict(sha256=digest,bytes=p.stat().st_size)
    write_json(archive/'archive_manifest.json',dict(job=jid,files=manifest,parameter_sha256=point_hash,
        gate_sha256=gate_hash,diagnostic_sha256=sha(probe),qualification=qualification,
        cumulative_training_calls=saved['calls'],cumulative_active_seconds=saved['elapsed']))
    # Do not expose a ready re-entry until the full old scientific archive is verified.
    write_json(archive/'preparation_journal.json',dict(phase='archive_verified_before_new_memory'))
    save_pickle(folder/'optimizer.pkl',replacement)
    for rel in ['independent_audit.json','solution_audit_progress.json','audit_current_probe.json',
                'physical_ledger.json','full_history_physical_source_ledger.npz','station_source_ledger.npz',
                'prediction_freeze.json']:
        p=folder/rel
        if p.exists():
            if sha(p)!=manifest[rel]['sha256']:raise RuntimeError('OLD_RECEIPT_CHANGED '+rel)
            p.unlink()
    if (folder/'plant_activity_yearly').exists():
        plant_root=(folder/'plant_activity_yearly').resolve()
        if not plant_root.is_relative_to(folder.resolve()) or (folder/'plant_activity_yearly').is_symlink():
            raise RuntimeError('ORIGINAL_PLANT_LEDGER_ROOT_REQUIRED')
        removals=sorted((folder/'plant_activity_yearly').rglob('*'),key=lambda p:len(p.parts),reverse=True)
        if any(p.is_symlink() or not p.resolve().is_relative_to(plant_root) for p in removals):
            raise RuntimeError('ALL_PLANT_REMOVALS_MUST_STAY_IN_VERIFIED_ROOT')
        # Archive proved each leaf first. All removals stay inside this explicit job.
        for p in removals:
            if p.is_file():
                rel=p.relative_to(folder).as_posix()
                if sha(p)!=manifest[rel]['sha256']:raise RuntimeError('OLD_PLANT_LEDGER_CHANGED')
                p.unlink()
            elif p.is_dir():p.rmdir()
        (folder/'plant_activity_yearly').rmdir()
    record=dict(parameter_sha256=point_hash,formal_gate_sha256=gate_hash,continuation_started=False,
        continuation_prepared=True,archive=archive.relative_to(folder).as_posix(),diagnostic_sha256=sha(probe),
        new_checkpoint_sha256=sha(folder/'optimizer.pkl'),cumulative_training_calls=saved['calls'],
        cumulative_active_seconds=saved['elapsed'],same_point=True,new_canonical_memory=True,
        old_workspace_reencoded=False,training_started=False)
    history['attempts'].append(record);write_json(historypath,history)
    status=json.loads((archive/'status.json').read_text());status.update(status='safe_checkpoint_paused',
        audit_completed=False,numerically_sufficient=False,reviewed_reentry_prepared=True,
        reentry_history_sha256=sha(historypath))
    write_json(folder/'status.json',status)
    write_json(archive/'preparation_journal.json',dict(phase='prepared_not_dispatched',record=record))
    write_json(archive/'prepared.json',record);print(json.dumps(record));return record


if __name__=='__main__':prepare(sys.argv[1])
