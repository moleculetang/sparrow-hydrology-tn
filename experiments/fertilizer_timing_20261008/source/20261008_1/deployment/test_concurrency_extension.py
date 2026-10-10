"""Missing requests, wrong numerics and unmeasured14worker cannot admit."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from concurrency_extension import qualify
from d29_platform.runtime import sha,write_json


def main():
    ownership=dict(formal_gate_sha256='gate',max_active=dict(local=2,workstation=12),peak_bytes=dict(local=6500000000,workstation=6500000000))
    checks={n:dict(passed=True) for n in ['point','gradient','final_state','prediction','objective','local_export_digest']}
    baseline=dict(controller_live_and_unchanged=True,rows=[dict(live=True,stage='train') for _ in range(12)])
    proof=dict(passed=True,formal_gate_sha256='gate',numerical_checks=[dict(call=i,checks=copy.deepcopy(checks),passed=True) for i in range(2)],
        baseline_stable=True,extra_calls=2,baseline_before=copy.deepcopy(baseline),baseline_after=copy.deepcopy(baseline),
        peak_extra_bytes=4896288768,source_multiplier=1.,new_campaign_paths_dispatched=0,training_interrupted=False)
    assert qualify(ownership,'workstation',13,'gate',proof)['admitted']
    assert qualify(ownership,'workstation',12,'gate',None)['admitted']
    rejected=[]
    for name,alter,role,cap in [('missing_probe',lambda p:p.clear(),'workstation',13),
        ('one_request',lambda p:p.update(numerical_checks=p['numerical_checks'][:1]),'workstation',13),
        ('wrong_gradient',lambda p:p['numerical_checks'][0]['checks']['gradient'].update(passed=False),'workstation',13),
        ('missing_local_ledger',lambda p:p['numerical_checks'][0]['checks'].pop('local_export_digest'),'workstation',13),
        ('peak_exceeded',lambda p:p.update(peak_extra_bytes=7000000000),'workstation',13),
        ('baseline_changed',lambda p:p.update(baseline_stable=False),'workstation',13),
        ('campaign_mutated',lambda p:p.update(new_campaign_paths_dispatched=1),'workstation',13),
        ('wrong_gate',lambda p:p.update(formal_gate_sha256='other'),'workstation',13),
        ('unmeasured14',lambda p:None,'workstation',14),('unmeasured_local',lambda p:None,'local',3)]:
        p=copy.deepcopy(proof);alter(p);r=qualify(ownership,role,cap,'gate',p)
        assert not r['admitted'],name;rejected.append(dict(control=name,reason=r['reason']))
    write_json(ROOT/'evidence/concurrency_extension_known_errors.json',dict(passed=True,negative_controls=rejected,
        scope='operational_admission_only_not_scientific_convergence',source_sha256=sha(ROOT/'deployment/concurrency_extension.py')))
    print('passed: bounded13worker admission and '+str(len(rejected))+' negative controls')


if __name__=='__main__':main()
