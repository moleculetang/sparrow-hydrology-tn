"""Independent station/year/month loops at the candidate preset, no fitting."""
import sys,json,ast,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np
import pandas as pd
import math
import d29_training.annual_chain as annual
from d29_platform import precision_candidate as precise
annual.run_land1=precise.run_land1;annual.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training
# Extract the independently written scalar-loop implementation without running
# the U-only command-line driver. It does not use StrategyObjective.evaluate.
tree=ast.parse((ROOT/'scripts/audit_u_solution.py').read_text(encoding='utf-8'))
fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='recompute')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<independent_scalar_objective>','exec'))
ok,resources=dispatch_allowed(reserve_bytes=9_000_000_000)
if not ok:raise RuntimeError('RESOURCE_GATE')
job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T0_s0')
a=Land1Training(job,diagnostic=True);evaluation=a.objective.evaluate;saved={}
def observe(p):
    saved['p']=p.copy();return evaluation(p)
a.objective.evaluate=observe
start=time.monotonic();value,_=a.value_gradient(a.initial,forward_only=True)
terms=recompute(a.objective.rows,saved['p'],a.objective.scales,job['strategy'],a.objective.long_sites)
independent=math.fsum(terms.values())+a.last['prior'];error=abs(value-independent)
write_json(ROOT/'outputs/land1_independent_objective_numerical_v2.json',{'passed':error<=1e-8*(1+abs(value)),'objective':value,'independent':independent,'error':error,'data_terms':terms,'prior':a.last['prior'],'elapsed_seconds':time.monotonic()-start,'candidate_sha256':sha(ROOT/'d29_platform/precision_candidate.py'),'adapter_sha256':sha(ROOT/'d29_training/land1_adapter.py'),'independent_implementation_sha256':sha(ROOT/'scripts/audit_u_solution.py'),'not_a_fitted_point':True,'NSE':'not applicable'})
print(value,independent,error,flush=True)

