"""Real64year BNF readbacks, regrouped endpoints and false final admission."""
import json,sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'scripts')]
from d29_platform.runtime import sha,write_json
from run_approved_bnf_delivery import verified_case
from bnf_endpoint_readback import verify as endpoints
from staged_frozen_sources import resolve
from reuse_staged_bnf_trace import verified_for_final


def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    delivery=json.loads((ROOT/'outputs/staged_bnf_trace/complete.json').read_text())
    if not delivery.get('passed') or len(delivery['completed'])!=6:raise RuntimeError('ACTUAL_SIX_BNF_REPLAYS_REQUIRED')
    cases=[];endpoint_checks=[];controls=0
    for record in delivery['completed']:
        job=next(j for j in jobs if j['id']==record['job']);path=Path(record['complete'])
        source=resolve(job['id'])
        if sha(path)!=record['sha256']:raise RuntimeError('STAGED_BNF_CONTROLLER_BYTES_CHANGED')
        cases.append(verified_case(path.parent,source,job,'staged'));controls+=1
        endpoint_checks.append(dict(job=job['id'],**endpoints(path.parent,source)));controls+=1
        try:verified_for_final(path.parent,source,job)
        except RuntimeError as exc:
            if 'FINAL_CORRECTED44DELIVERY_NOT_READY' not in str(exc):raise
            controls+=1
        else:raise AssertionError('partial BNF results admitted as44')
    job=next(j for j in jobs if j['id']==delivery['completed'][0]['job'])
    path=Path(delivery['completed'][0]['complete']);original=json.loads(path.read_text());reader=Path.read_text
    for key,value in [('worker_sha256','wrong'),('mode','final'),('evaluation_labels_read',True),
        ('new_training_calls',1),('BNF_timing_changed',True),('local_source_sum_max_error_kg',2e-6)]:
        changed=dict(original);changed[key]=value
        def override(self,*args,**kwargs):return json.dumps(changed) if self==path else reader(self,*args,**kwargs)
        with patch.object(Path,'read_text',override):
            try:verified_case(path.parent,resolve(job['id']),job,'staged')
            except RuntimeError:controls+=1
            else:raise AssertionError('changed BNF receipt admitted '+key)
    result=dict(passed=True,controls=controls,actual_full_history_cases=6,cases=cases,endpoints=endpoint_checks,
        exact44_admission_preserved=True,evaluation_admitted=False,
        source_sha256=sha(ROOT/'analysis/reuse_staged_bnf_trace.py'),
        launcher_sha256=sha(ROOT/'analysis/run_approved_bnf_delivery.py'),
        final_verifier_sha256=sha(ROOT/'analysis/audit_final_approved_delivery.py'))
    write_json(ROOT/'evidence/staged_bnf_reuse_controls.json',result)
    print(json.dumps(dict(passed=True,controls=controls,actual_full_history_cases=6)))


if __name__=='__main__':main()
