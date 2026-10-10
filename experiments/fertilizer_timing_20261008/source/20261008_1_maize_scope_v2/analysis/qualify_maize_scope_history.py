"""Expanded Maize range: actual complete-history derivative/ledger checks.

The old zero-management derivatives are reused only after bitwise forward,
current gradient and initial-cotangent regression at that identical point.
Expanded interior/boundary probes are newly computed. No fits or scores.
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.with_name('20261008_1')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(PARENT/'work')]
from d29_platform.runtime import configure, write_json, sha
configure()
import numpy as np
from timing.adapter import TimingTraining
from timing.chain import TimingChain
from timing.branch_trace import trace
from timing.independent_objective import recompute
from audit_solution import probe
from source_ledger import ledger


def independent(a, point, value):
    # Independently spell the approved prior, rather than read a.last's term.
    delta = point[22:]
    m = len(delta)
    spans = a.bounds[22:, 1] - a.bounds[22:, 0]
    prior = 0.1/(2*m)*float(np.sum((delta/(spans/2))**2)) if m else 0.
    j = sum(recompute(a.objective.rows, a.prediction, a.objective.scales,
                      a.objective.long_sites).values()) + a.last['prior'] + prior
    if abs(j-value) > 1e-8*(1+abs(value)):
        raise RuntimeError('INDEPENDENT_APPROVED_OBJECTIVE_FAILED')
    if abs(prior-a.last['management_prior']) > 1e-12*(1+abs(prior)):
        raise RuntimeError('INDEPENDENT_APPROVED_MANAGEMENT_PRIOR_FAILED')
    return dict(value=float(value), independently_recomputed=float(j),
                error=float(abs(j-value)), management_prior=prior)


def qualification():
    start = time.monotonic()
    out = ROOT/'outputs/maize_scope_history'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('EXISTING_HISTORY_RESULT_REVIEW_NO_BLIND_RESTART')
    gate = json.loads((PARENT/'evidence/formal_gate.json').read_text(encoding='utf8'))
    changed = []
    for rel, digest in gate['identities'].items():
        if sha(PARENT/rel) != digest:
            raise RuntimeError('LIVE_PARENT_SEAL_CHANGED '+rel)
        if sha(ROOT/rel) != digest:
            changed.append(rel)
    if changed != ['config/timing_evidence.json']:
        raise RuntimeError('ONLY_REGISTERED_RANGE_CONFIG_CHANGE_ALLOWED '+repr(changed))
    profile_proof = ROOT/'outputs/maize_scope_profiles/complete.json'
    profile = json.loads(profile_proof.read_text(encoding='utf8'))
    if not profile['passed'] or profile['identities']['config/timing_evidence.json'] != sha(ROOT/'config/timing_evidence.json'):
        raise RuntimeError('CURRENT_EXPANDED_PROFILE_PROOF_REQUIRED')
    derivatives = PARENT/'evidence/full_transition_derivative_requalification.json'
    proof = json.loads(derivatives.read_text(encoding='utf8'))
    if not proof['passed']:
        raise RuntimeError('PARENT_FULL_DERIVATIVE_QUALIFICATION_REQUIRED')
    for rel, digest in proof['artifact_sha256'].items():
        if sha(PARENT/rel) != digest:
            raise RuntimeError('PARENT_REUSED_PROOF_BYTES_CHANGED '+rel)
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    records = []
    a = None
    for jid in ('TIM_F23_C0_V0_s0', 'TIM_F23_K_V1_s0'):
        job = next(j.copy() for j in jobs if j['id']==jid)
        job['id']='maize_scope_regression_'+jid
        a = TimingTraining(job, acceptance=True)
        old = PARENT/'outputs/jobs'/('transition_requalification_'+jid)
        point = np.load(old/'point.npy', allow_pickle=False)
        if not np.array_equal(point, a.initial):
            raise RuntimeError('SAME_LEGAL_PARENT_POINT_REQUIRED')
        capture={}
        native=TimingChain.backward
        def captured(chain,*args,**kwargs):
            result=native(chain,*args,**kwargs)
            capture['initial']=result['initial'].copy()
            return result
        with patch.object(TimingChain, 'backward', captured):
            value, gradient=a.value_gradient(point)
        checks={}
        for name, actual in [('training_prediction.npy',a.prediction),
                             ('local_kg_day.npy',a.local_prediction),
                             ('final_state.npy',a.last_chain.final),
                             ('gradient.npy',gradient),
                             ('initial_cotangent.npy',capture['initial'])]:
            expected=np.load(old/name,allow_pickle=False)
            checks[name]=dict(bitwise_equal=bool(np.array_equal(actual,expected)),
                             max_abs_error=float(np.max(abs(actual-expected))))
            np.save(a.folder/name,actual)
        old_result=json.loads((old/'transition_requalification.json').read_text(encoding='utf8'))
        passed=all(v['bitwise_equal'] for v in checks.values()) and float(value)==old_result['objective']
        if not passed:
            raise RuntimeError('ZERO_MANAGEMENT_FULL_HISTORY_REGRESSION_FAILED '+jid)
        records.append(dict(job=jid,passed=passed,arrays=checks,
                            objective=independent(a,point,value),
                            point=point.tolist(),names=a.names))
        write_json(out/'progress.json',dict(phase='zero_management_regression',records=records,
                   full_history=[1961,2024],production_gate_passed=False))
    if a.names[-1]!='Maize_Nfer_basal' or len(a.names)!=24:
        raise RuntimeError('MAIZE_COORDINATE_IDENTITY')
    index=a.names.index('Maize_Nfer_basal')
    if a.bounds[index].tolist()!=[-.1,0.] or a.inputs.sigmas[-1]!=.05:
        raise RuntimeError('APPROVED_RANGE_AND_PRIOR_REQUIRED')
    base=a.initial.copy()
    cases=[]
    for delta,side in ((0.,'minus'),(-.075,'central'),(-.1,'plus')):
        point=base.copy();point[index]=delta
        a.folder=out/('case_'+str(delta).replace('-','minus').replace('.','p'))
        a.folder.mkdir(exist_ok=True)
        with trace() as reference:value,gradient=a.value_gradient(point)
        objective=independent(a,point,value)
        np.save(a.folder/'point.npy',point)
        np.save(a.folder/'gradient.npy',gradient)
        result=probe(a,point,value,gradient,reference,index,side,a.folder)
        physical=ledger(a,point) if delta==-.1 else None
        cases.append(dict(delta=delta,side=side,derivative=result,
                          independent_objective=objective,physical=physical,
                          passed=bool(result['passed'] and (physical is None or physical['passed']))))
        write_json(out/'progress.json',dict(phase='expanded_full_history_probes',
                   regression=records,cases=cases,production_gate_passed=False))
    # Existing FD slopes on all coordinates and five supported initial families
    # apply at delta=0 only, where actual forward, gradient and initial cotangent
    # were just reproduced. This is not every coordinate at every bound.
    reused=dict(scope='identical_zero_management_point_only',
                proof_path=str(derivatives),proof_sha256=sha(derivatives),
                coordinate_coverage=[dict(job=c['job'],count=len(c['checks']),passed=c['passed'])
                                     for c in proof['coordinate_cases']],
                initial_family_count=len(proof['initial_cases']),
                old_receipts_overwritten=False)
    result=dict(passed=all(c['passed'] for c in cases),
                scope='complete1961_2024_expanded_maize_range_qualification_not_fit_or_dualhost_seal',
                regression=records,cases=cases,reused=reused,seconds=time.monotonic()-start,
                profile_proof_sha256=sha(profile_proof),
                config_sha256=sha(ROOT/'config/timing_evidence.json'),
                source_multiplier=1.,scientific_training_dispatched=False,
                production_gate_passed=False,dual_host_passed=False,
                evaluation_labels_read=False)
    write_json(out/'complete.json',result)
    print(json.dumps(dict(passed=result['passed'],cases=[dict(delta=c['delta'],passed=c['passed']) for c in cases],
                         seconds=result['seconds'])))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    sys.exit(qualification())
