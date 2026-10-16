"""One legal-row constructor/full-history oracle at the expanded interior point."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from d29_platform.runtime import configure,sha,write_json
configure()
import numpy as np
import pandas as pd
from training_only_constructor import build_training
from timing.independent_objective import recompute

def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    job=next(j.copy() for j in jobs if j['id']=='TIM_F23_K_V1_s0')
    job['id']='scope_strict_point_F23_K_V1'
    out=ROOT/'outputs/scope_strict_point';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('ACTUAL_STRICT_SNAPSHOT_EXISTS_NO_RESTART')
    pointfile=ROOT/'outputs/maize_scope_history/case_minus0p075/point.npy'
    expectedfile=ROOT/'outputs/maize_scope_history/case_minus0p075/gradient.npy'
    history=json.loads((ROOT/'outputs/maize_scope_history/complete.json').read_text(encoding='utf8'))
    if not history['passed'] or history['config_sha256']!=sha(ROOT/'config/timing_evidence.json'):
        raise RuntimeError('CURRENT_APPROVED_EXPANDED_HISTORY_REQUIRED')
    expected_case=next(c for c in history['cases'] if c['delta']==-.075)
    reader=pd.read_parquet;reads=[];start=time.monotonic()
    def observed(path,*args,**kwargs):
        result=reader(path,*args,**kwargs);p=Path(path).resolve()
        if p in [(ROOT/'data/monthly_tn.parquet').resolve(),(ROOT/'data/hf_daily.parquet').resolve()]:
            years=result.year if 'year' in result else result.date.dt.year
            invalid=int(years.gt(job['train_end']).sum())
            reads.append(dict(file=p.relative_to(ROOT).as_posix(),rows=len(result),
                              evaluation_year_rows_returned=invalid,filters=repr(kwargs.get('filters'))))
            if invalid:raise RuntimeError('EVALUATION_ROWS_EXPOSED')
        return result
    with patch.object(pd,'read_parquet',side_effect=observed):a=build_training(job,acceptance=True)
    if len(reads)!=2:raise RuntimeError('BOTH_LEGAL_TRAINING_TABLES_REQUIRED')
    point=np.load(pointfile,allow_pickle=False);expected=np.load(expectedfile,allow_pickle=False)
    value,gradient=a.value_gradient(point)
    independent=sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
    gradient_equal=bool(np.array_equal(gradient,expected))
    objective_equal=float(value)==expected_case['independent_objective']['value']
    if not gradient_equal or not objective_equal or abs(value-independent)>1e-8*(1+abs(value)):
        raise RuntimeError('STRICT_ACCESS_FULL_ORACLE_NOT_EQUIVALENT')
    for name,array in [('point',point),('gradient',gradient),('training_prediction',a.prediction),
                       ('local_kg_day',a.local_prediction),('final_state',a.last_chain.final)]:
        np.save(out/(name+'.npy'),array)
    from train_job import backend
    write_json(out/'complete.json',dict(passed=True,scope='full1961_2024_strict_access_expanded_interior_oracle_not_fit',
        value=float(value),independent_objective=float(independent),constructor_reads=reads,
        objective_bitwise_equal_to_pre_access_epoch=objective_equal,
        gradient_bitwise_equal_to_pre_access_epoch=gradient_equal,
        source_multiplier=1.,config_sha256=sha(ROOT/'config/timing_evidence.json'),
        array_sha256={n:sha(out/(n+'.npy')) for n in ['point','gradient','training_prediction','local_kg_day','final_state']},
        backend=backend(),seconds=time.monotonic()-start,evaluation_labels_read=False,
        formal_gate_passed=False,dualhost_passed=False))
    print(json.dumps(dict(passed=True,seconds=time.monotonic()-start,eval_rows_returned=0)))
    return 0

if __name__=='__main__':sys.exit(main())
