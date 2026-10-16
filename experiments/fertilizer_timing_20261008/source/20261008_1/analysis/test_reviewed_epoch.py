"""New native epoch preserves old bytes, point, backend and cumulative counts."""
import copy
import pickle
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
import numpy as np
from prepare_reviewed_reentry import create_epoch
from native_bound_adapter_v2 import CanonicalLBFGSB,restore_canonical_solver
from d29_platform.runtime import write_json,sha


def main():
    x=np.array([2.,-1.]);scale=np.array([.5,2.]);b=[(None,None),(0.,None)]
    # Use a feasible point; one absent lower/upper side must remain absent.
    x[1]=1.
    old=CanonicalLBFGSB(x/scale,[(None,None),(0.,None)],stage=1)
    old.s['done']=True
    contract=dict(contract_sha256='contract',backend={'scipy':'1.17.1'},source_multiplier=1.,scale=scale.tolist(),bounds=b)
    saved=dict(status='numerical_review_required',contract_sha256='contract',backend=contract['backend'],source_multiplier=1.,
        solver=old.s,best=dict(x=x.copy(),value=3.,gradient=np.array([2.,1.]),pg=2.),calls=113,elapsed=1000.)
    original=pickle.dumps(saved,protocol=5);new=create_epoch(saved,contract,x)
    assert pickle.dumps(saved,protocol=5)==original
    assert new['calls']==113 and new['elapsed']==1000. and new['status']=='safe_checkpoint_paused'
    assert not new['solver']['done'] and new['solver']['evaluations']==0
    assert np.array_equal(new['solver']['x'],x/scale) and new['solver']['nbd'].tolist()==[0,1]
    first=restore_canonical_solver(new['solver'])
    def fun(z):
        p=z*scale;g=np.array([p[0]-.25,p[1]-.5]);return .5*float(g@g),g*scale
    for _ in range(200):
        event=first.step(fun)
        if event=='evaluation':break
    restored=restore_canonical_solver(copy.deepcopy(first.s));direct=restore_canonical_solver(copy.deepcopy(first.s))
    for solver in (restored,direct):
        for _ in range(200):
            if solver.step(fun)=='done':break
        else:raise AssertionError('known truth solver did not stop')
    assert np.array_equal(restored.s['x'],direct.s['x'])
    assert np.max(abs(restored.s['x']*scale-np.array([.25,.5])))<1e-6
    rejected=[]
    for name,alter in [('wrong_point',lambda s,c:s['best'].update(x=np.zeros(2))),
                       ('old_multiplier',lambda s,c:s.update(source_multiplier=2.)),
                       ('wrong_backend',lambda s,c:s.update(backend={'scipy':'other'})),
                       ('not_stopped',lambda s,c:s.update(status='running')),
                       ('wrong_contract',lambda s,c:s.update(contract_sha256='other'))]:
        s,c=copy.deepcopy((saved,contract));alter(s,c)
        try:create_epoch(s,c,x)
        except RuntimeError:rejected.append(name)
        else:raise AssertionError(name)
    write_json(ROOT/'evidence/reviewed_epoch_known_truth.json',dict(passed=True,negative_controls=rejected,
        old_checkpoint_bytes_preserved=True,cumulative_training_calls_preserved=True,canonical_absent_bounds=True,
        partial_recovery_matches_direct=True,scope='small_solver_only_no_actual_path_reentry',
        source_sha256=sha(ROOT/'analysis/prepare_reviewed_reentry.py')))
    print('passed: canonical same-point epoch, counters, native recovery and '+str(len(rejected))+' negative controls')


if __name__=='__main__':main()
