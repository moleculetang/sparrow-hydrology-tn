"""Compare new training/audit/export measurements before freezing dispatch."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from d29_platform.runtime import write_json,sha
from compare_hosts import arrays


def export_equal(left,right):
    import numpy as np
    import pandas as pd
    a=pd.read_parquet(left);b=pd.read_parquet(right);keys=['station_key','date']
    if a.duplicated(keys).any() or b.duplicated(keys).any():return dict(passed=False,reason='duplicate_station_day')
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]):return dict(passed=False,reason='different_station_day_support')
    comparisons={}
    for c,tol,relative in [('prediction_mg_l',1e-8,True),('mass_kg',1e-6,False),('water_m3',1e-10,True)]:
        x=a[c].to_numpy();y=b[c].to_numpy();missing=np.isnan(x)
        valid=np.array_equal(missing,np.isnan(y)) and not np.isinf(x).any() and not np.isinf(y).any()
        diff=abs(x[~missing]-y[~missing]);valid &= np.all(diff<=tol*(1+abs(x[~missing])) if relative else diff<=tol)
        comparisons[c]=dict(passed=bool(valid),max_abs_difference=float(diff.max(initial=0)))
    return dict(passed=all(v['passed'] for v in comparisons.values()),rows=len(a),stations=a.station_key.nunique(),values=comparisons)


def main():
    jid='TIM_F23_C0_V0_s0';base=ROOT/'outputs/jobs'/('measurement_threads1_'+jid)
    reference=json.loads((base/'stage_measurement.json').read_text(encoding='utf8'))
    cases=[];pending=[]
    for threads in (1,2,4):
        folder=ROOT/'outputs/imported_workstation_measurements/outputs/jobs'/f'measurement_threads{threads}_{jid}'
        p=folder/'stage_measurement.json'
        if not p.exists():pending.append(str(p.relative_to(ROOT)));continue
        result=json.loads(p.read_text(encoding='utf8'))
        valid=result['passed'] and reference['passed'] and result['formal_gate_sha256']==reference['formal_gate_sha256']
        compared={name:arrays(base/name,folder/name,tol,relative) for name,tol,relative in (
            ('point.npy',0.,False),('gradient.npy',1e-6,True),('training_prediction.npy',1e-8,True),
            ('local_kg_day.npy',1e-6,False),('final_state.npy',1e-6,False))}
        exported=export_equal(base/'frozen_station_days.parquet',folder/'frozen_station_days.parquet')
        lhs=reference['measurements']['native_training_two_requests']['target_requests']
        rhs=result['measurements']['native_training_two_requests']['target_requests']
        jpass=len(lhs)==len(rhs)==2 and all(abs(x['objective']-y['objective'])<=1e-8*(1+abs(x['objective'])) for x,y in zip(lhs,rhs))
        valid &= all(v['passed'] for v in compared.values()) and exported['passed'] and jpass
        cases.append(dict(threads=threads,passed=bool(valid),arrays=compared,station_export=exported,native_two_objectives_passed=jpass,
            measurements=result['measurements'],native_peak=result['native_peak'],receipt_sha256=sha(p)))
    receipt=dict(passed=not pending and len(cases)==3 and all(c['passed'] for c in cases),pending=pending,cases=cases,
        windows_measurements=reference['measurements'],windows_peak=reference['native_peak'],production_threads=1,
        scope='actual_full_history_native_requests_coordinate_audit_and_full_station_export_not_converged_training',
        different_shared_loads=True,controlled_parallel_speedup_claimed=False,
        comparator_sha256=sha(Path(__file__)))
    write_json(ROOT/'evidence/stage_measurement_consistency.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],pending=pending,cases=[dict(threads=c['threads'],passed=c['passed'],
        seconds={k:v['seconds'] for k,v in c['measurements'].items()},peak=c['native_peak']) for c in cases])))
    return 0 if receipt['passed'] else 2


if __name__=='__main__':sys.exit(main())
