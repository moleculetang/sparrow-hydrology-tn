"""Candidate-only full-history forward/adjoint comparison. No calibration."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np
import d29_training.annual_chain as chain
from d29_platform import precision_candidate as candidate
chain.run_land1=candidate.run_land1
chain.land1_adjoint=candidate.land1_adjoint
from d29_training.land1_adapter import Land1Training
ok,res=dispatch_allowed(reserve_bytes=7_000_000_000)
if not ok:raise RuntimeError('RESOURCE_GATE_NO_DISPATCH')
j=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T0_s0')
a=Land1Training(j,diagnostic=True);start=time.monotonic();v,g=a.value_gradient(a.initial)
old=json.loads((ROOT/'outputs/land1_full_history_gradient_v2.json').read_text(encoding='utf-8'))
receipt={'candidate_only':True,'no_calibration':True,'objective':v,'gradient':g.tolist(),'difference_from_original_objective':v-old['objective'],'max_gradient_difference':float(np.max(abs(g-np.asarray(old['analytic_gradient'])))),'elapsed_seconds':time.monotonic()-start,'last':a.last,'formal_gate_passed':False,'pending':'full source-tag represented-state audit and branch/gradient acceptance','identities':{str(p):sha(p) for p in [ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py']}}
write_json(ROOT/'outputs/precision_candidate_full_history.json',receipt)
print(json.dumps(receipt),flush=True)
