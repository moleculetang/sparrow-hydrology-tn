"""Synthetic file transaction controls, not a real physical audit or restart."""
import copy
import hashlib
import json
import pickle
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'scripts'),str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
import numpy as np
import prepare_reviewed_reentry as prepare_module
from native_bound_adapter_v2 import CanonicalLBFGSB
from d29_platform.runtime import write_json,sha
from host_resources import registry
import train_job,job_lease


def fixture(root):
    jid='SYNTHETIC_REENTRY';folder=root/'outputs/jobs'/jid;folder.mkdir(parents=True)
    job=dict(id=jid);host=platform.node()
    for name,value in [('config/jobs.json',[job]),('deployment/path_ownership.json',dict(owners={jid:'local'})),
                       ('outputs/campaign_local/owner.json',dict(pid=99999998,created='fixture'))]:
        (root/name).parent.mkdir(parents=True,exist_ok=True);write_json(root/name,value)
    contract=dict(job=job,names=['x','y'],scale=[1.,1.],bounds=[[None,None],[0.,None]],
        formal_gate_sha256='gate',backend={'synthetic':True},source_multiplier=1.,host=host)
    contract['contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    write_json(folder/'training_contract.json',contract);write_json(folder/'owner.json',dict(host=host,pid=99999999,created='fixture'))
    point=np.array([2.,1.]);np.save(folder/'best.npy',point);np.save(folder/'best_gradient.npy',[.02,0.])
    point_hash=sha(folder/'best.npy')
    native=CanonicalLBFGSB(point,[(None,None),(0.,None)],stage=1);native.s['done']=True
    saved=dict(status='numerical_review_required',contract_sha256=contract['contract_sha256'],backend=contract['backend'],
        source_multiplier=1.,solver=native.s,best=dict(x=point.copy(),value=3.,gradient=np.array([.02,0.]),pg=.02),calls=113,elapsed=1000.)
    (folder/'optimizer.pkl').write_bytes(pickle.dumps(saved,protocol=5))
    write_json(folder/'status.json',dict(status='numerical_review_required',calls=113))
    audit=dict(source_multiplier=1.,formal_gate_sha256='gate',parameter_sha256=point_hash,full_history=[1961,2024],reaches=230,
        objective_passed=True,physical_passed=True,gradient_recovery_passed=True,derivative_passed=True,
        checks=[dict(coordinate=n,derivative_qualified=True) for n in contract['names']],objective=3.,projected_gradient=.02,
        scaled_projected_gradient=.02,numerically_sufficient=False,synthetic_control=True)
    write_json(folder/'independent_audit.json',audit);write_json(folder/'physical_ledger.json',dict(synthetic_control=True))
    plant=folder/'plant_activity_yearly/1961.npz';plant.parent.mkdir();np.savez(plant,synthetic_control=np.array([1]))
    for name in ['continuation_review_policy.py','probe_same_point_descent.py']:
        p=root/'analysis'/name;p.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'analysis'/name,p)
    p=root/'evidence/continuation_review_policy_known_errors.json';p.parent.mkdir()
    write_json(p,dict(passed=True,source_sha256=sha(root/'analysis/continuation_review_policy.py')))
    witness=dict(parameter_sha256=point_hash,formal_gate_sha256='gate',source_multiplier=1.,evaluation_scores_used=False,
        base_objective=3.,direction=[-1.,0.],probes=[dict(step=h,slope=-.02,analytic=-.02,objective=3.-.02*h,
        feasible=True,branch_consistent=True,branch_signature='fixture_branch') for h in [.01,.005]])
    probe=folder/'continuation_diagnostics'/point_hash/'complete.json';probe.parent.mkdir(parents=True)
    write_json(probe,dict(passed=True,new_solver_started=False,witness=witness,synthetic_control=True,
        source_sha256=sha(root/'analysis/probe_same_point_descent.py'),
        fixed_scientific_artifacts={rel:sha(folder/rel) for rel in ['best.npy','independent_audit.json','optimizer.pkl','training_contract.json']}))
    return jid,folder,point_hash


def main():
    parent=ROOT/'outputs/test_fixtures';parent.mkdir(exist_ok=True);controls=[]
    for mode in ('accepted_transaction','archive_corruption'):
        with tempfile.TemporaryDirectory(prefix='reentry_transaction_',dir=parent) as directory:
            root=Path(directory).resolve();assert root.is_relative_to(parent.resolve())
            jid,folder,point_hash=fixture(root);old_digest=sha(folder/'optimizer.pkl');copier=shutil.copy2
            def corrupt(src,dest):
                result=copier(src,dest)
                if Path(src).name=='optimizer.pkl':Path(dest).write_bytes(b'corrupt synthetic archive')
                return result
            with patch.object(prepare_module,'ROOT',root),patch.object(train_job,'validate_admission',return_value=({},'gate')),\
                 patch.object(train_job,'backend',return_value={'synthetic':True}),\
                 patch.object(job_lease,'require',return_value=dict(token='fixture',threads=1)),\
                 patch.object(registry,'identity',return_value=None),patch.object(registry,'status',return_value={'leases':{}}):
                if mode=='accepted_transaction':
                    result=prepare_module.prepare(jid)
                    archive=folder/result['archive'];manifest=json.loads((archive/'archive_manifest.json').read_text())
                    assert sha(archive/'optimizer.pkl')==old_digest
                    assert all(sha(archive/rel)==r['sha256'] for rel,r in manifest['files'].items())
                    assert (archive/'independent_audit.json').exists() and (archive/'plant_activity_yearly/1961.npz').exists()
                    assert not (folder/'independent_audit.json').exists() and not (folder/'plant_activity_yearly').exists()
                    new=pickle.loads((folder/'optimizer.pkl').read_bytes())
                    assert new['calls']==113 and new['elapsed']==1000. and not new['solver']['done']
                    assert sha(folder/'best.npy')==point_hash and result['training_started'] is False
                    controls.append(dict(control=mode,passed=True,original_optimizer_archive_verified=True,
                        point_and_cumulative_counters_preserved=True,training_started=False))
                else:
                    with patch.object(prepare_module.shutil,'copy2',side_effect=corrupt):
                        try:prepare_module.prepare(jid)
                        except RuntimeError as exc:assert 'ARCHIVE_HASH_MISMATCH' in str(exc)
                        else:raise AssertionError('corrupt archive was accepted')
                    assert sha(folder/'optimizer.pkl')==old_digest and (folder/'independent_audit.json').exists()
                    assert (folder/'plant_activity_yearly/1961.npz').exists()
                    controls.append(dict(control=mode,passed=True,original_checkpoint_and_audit_not_mutated=True))
    write_json(ROOT/'evidence/reentry_file_transaction_controls.json',dict(passed=True,controls=controls,
        scope='synthetic_transaction_only_not_actual_history_or_lease_qualification',actual_training_started=False,
        source_sha256=sha(ROOT/'analysis/prepare_reviewed_reentry.py')))
    print('passed: archived same-point transaction and corruption-before-mutation negative control')


if __name__=='__main__':main()
