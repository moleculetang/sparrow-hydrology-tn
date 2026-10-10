"""Actual six-history readbacks and false final-admission controls."""
import json,sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'scripts')]
from d29_platform.runtime import sha,write_json
from run_approved_internal_trace_delivery import verified_case
from staged_frozen_sources import resolve
from reuse_staged_internal_trace import compare_bindings,verified_for_final


def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    delivery=json.loads((ROOT/'outputs/staged_internal_trace/complete.json').read_text())
    if not delivery.get('passed') or len(delivery['completed'])!=6:raise RuntimeError('ACTUAL_SIX_REPLAYS_REQUIRED')
    cases=[];controls=0
    for record in delivery['completed']:
        job=next(j for j in jobs if j['id']==record['job'])
        path=Path(record['complete']);destination=path.parent;source=resolve(job['id'])
        if sha(path)!=record['sha256']:raise RuntimeError('STAGE_CONTROLLER_BINDING_CHANGED')
        cases.append(verified_case(destination,source,job,'staged'));controls+=1
        try:verified_for_final(destination,source,job)
        except RuntimeError as exc:
            if 'FINAL_CORRECTED44DELIVERY_NOT_READY' not in str(exc):raise
            controls+=1
        else:raise AssertionError('partial results admitted before actual44')
    compare_bindings({'a':'hash'},{'a':'hash'},'prediction','prediction');controls+=1
    for args in [({'a':'hash'},{'a':'different'},'p','p'),
                 ({'a':'hash'},{},'p','p'),
                 ({'a':'hash'},{'a':'hash'},'p','different')]:
        try:compare_bindings(*args)
        except RuntimeError:controls+=1
        else:raise AssertionError('changed terminal binding accepted')
    job=next(j for j in jobs if j['id']==delivery['completed'][0]['job'])
    path=Path(delivery['completed'][0]['complete']);original=json.loads(path.read_text());reader=Path.read_text
    for key,value in [('source_sha256','wrong'),('mode','final'),('no_evaluation_outcomes_read',False),
        ('new_scientific_training_calls',1),('routing_trace_sha256','wrong')]:
        changed=dict(original);changed[key]=value
        def read_override(self,*args,**kwargs):
            return json.dumps(changed) if self==path else reader(self,*args,**kwargs)
        with patch.object(Path,'read_text',read_override):
            try:verified_case(path.parent,resolve(job['id']),job,'staged')
            except RuntimeError:controls+=1
            else:raise AssertionError('changed staged proof admitted '+key)
    proof=dict(passed=True,controls=controls,actual_full_history_cases=6,cases=cases,
        exact44_admission_preserved=True,evaluation_admitted=False,
        source_sha256=sha(ROOT/'analysis/reuse_staged_internal_trace.py'),
        launcher_sha256=sha(ROOT/'analysis/run_approved_internal_trace_delivery.py'),
        final_verifier_sha256=sha(ROOT/'analysis/audit_final_approved_delivery.py'))
    write_json(ROOT/'evidence/staged_trace_reuse_controls.json',proof)
    print(json.dumps(dict(passed=True,controls=controls,actual_full_history_cases=6)))


if __name__=='__main__':main()
