"""Known internal-transfer and shared-reservoir negative controls."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from d29_platform.timing_kernel import run_land1
from d29_platform.runtime import write_json,sha
from internal_flux_observer import observe,routing_checks,stage_semantics


def case(initial,sources,out=None,cap=None,transitions=None):
    nt,nr,nl,_=sources.shape
    return run_land1(initial=initial,sources=sources,plant_target=np.zeros((nt,nr,nl)),
        plant_outflows=np.zeros((nt,nr,nl,3)) if out is None else out,
        probabilities=dict(mineralize_active=.25,mineralize_protected=.1,mobilize=.2,
            fast_fraction=.5,available_loss=.1,lower_release=.3),
        plant_growth_capacity=cap,transitions=transitions,compensated=True,
        plant_activity_mode='potential_with_shortfall')


def main():
    checks={};initial=np.zeros((1,2,5));source=np.zeros((3,1,2,4));source[0,0,0,1]=8.
    result=case(initial,source);arrays,receipt=observe(result)
    assert receipt['passed']
    release=arrays['mineralize_active_kg_daily_by_land']
    assert release[0,0]==0. and release[1,0]==2.
    checks['new_organic_next_day_release']=True
    assert arrays['plant_actual_uptake_kg_daily_by_land'].sum()==0.
    checks['plant_external_export_not_mislabeled_uptake']=True
    zero,receipt=observe(case(initial,np.zeros_like(source)))
    assert receipt['passed'] and all(np.count_nonzero(v)==0 for v in zero.values())
    checks['zero_source_stock_and_water_no_invented_transfer']=True
    s=source.copy();s[0,0,0,0]=1.;out=np.zeros((3,1,2,3));out[0,0,0]=[2.,1.,1.]
    r=case(initial,s,out=out,cap=np.zeros((3,1,2)));a,c=observe(r)
    assert c['passed'] and r.unmet_plant_outflows[0,0,0].sum()==3.
    assert a['plant_actual_uptake_kg_daily_by_land'].sum()==0.
    checks['capacity_zero_and_potential_output_shortfall']=True
    st=initial.copy();st[0,0]=[1.,8.,4.,2.,1.]
    matrices={0:np.array([[[0.,1.],[1.,0.]]])}
    a,c=observe(case(st,np.zeros_like(source),transitions=matrices))
    assert c['passed'] and a['mineralize_active_kg_daily_by_land'][0,1]==2.
    checks['land_transfer_precedes_release']=True
    local=np.array([[2.,3.,0.],[0.,0.,0.]])
    routing=dict(inlet=np.array([[0.,0.,2.5],[0.,0.,1.25]]),preout=np.zeros((2,3)),
        official=np.zeros((2,3)),channel_removed=np.zeros((2,3)),
        captures=np.array([[5.],[0.]]),releases=np.array([[2.5],[1.25]]),
        stocks=np.array([[2.5],[1.25]]),terminal=np.array([2.5,1.25]))
    _,c=routing_checks(local,routing)
    assert c['passed'] and c['reservoir_count']==1
    checks['shared_reservoir_once_and_zero_local_release_day']=True
    corrupted=dict(routing);corrupted['releases']=routing['releases']*2
    _,bad=routing_checks(local,corrupted)
    assert not bad['passed']
    checks['double_release_negative_control_rejected']=True
    s=stage_semantics('available_before_uptake_kg')
    assert s['quantity_kind']=='stock_snapshot' and s['sum_value_units']=='kg N day' and not s['annual_sum_is_external_mass']
    for stage in ('plant_potential_demand_kg','plant_capacity_limited_demand_kg','plant_supply_shortfall_kg'):
        assert stage_semantics(stage)['quantity_kind']=='potential_or_unmet_daily_request'
    assert stage_semantics('external_available_kg')['annual_sum_is_external_mass']
    assert not stage_semantics('mineralize_active_kg')['annual_sum_is_external_mass']
    checks['stock_days_recurrent_requests_and_actual_external_mass_distinguished']=True
    payload=dict(passed=all(checks.values()),checks=checks,full_history_run=False,
        no_training_or_evaluation_labels_read=True,physical_sources_modified=False,
        identities={n:sha(ROOT/n) for n in ['analysis/internal_flux_observer.py','analysis/test_internal_flux_observer.py']})
    write_json(ROOT/'evidence/internal_flux_observer_known_truth_v2.json',payload)
    print(json.dumps(payload))


if __name__=='__main__':main()
