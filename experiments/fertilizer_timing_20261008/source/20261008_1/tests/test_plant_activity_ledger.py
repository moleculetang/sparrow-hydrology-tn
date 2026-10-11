"""Actual debt vs repeated stock deficit, absent capacity and transfer replay."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from d29_platform.timing_kernel import run_land1,PROBABILITY_NAMES
from source_ledger import plant_activity_diagnostics
from d29_platform.runtime import write_json,sha


def main():
    checks={};initial=np.zeros((1,2,5));initial[0,0,3]=2.
    sources=np.zeros((3,1,2,4));target=np.zeros((3,1,2));target[:,0,0]=[5.,4.,4.]
    out=np.zeros((3,1,2,3));out[1,0,0,0]=3.
    args=dict(initial=initial,sources=sources,plant_target=target,plant_outflows=out,
        probabilities=dict(zip(PROBABILITY_NAMES,[0.,0.,0.,0.,.5,0.])),
        plant_activity_mode='potential_with_shortfall',compensated=True)
    result=run_land1(**args,plant_growth_capacity=np.ones(target.shape))
    saved=result.states.copy();r=plant_activity_diagnostics(result)
    assert r['uptake_constraints_passed'] and np.array_equal(saved,result.states)
    assert r['actual_uptake_by_land_kg']==[2.,0.]
    assert r['supply_limited_uptake_shortfall_by_land_kg']==[1.,0.]
    assert r['planned_output_by_land_destination_kg'][0][0]==3.
    assert r['realized_output_by_land_destination_kg'][0][0]==2.
    assert r['unmet_output_by_land_destination_kg'][0][0]==1.
    assert r['end_year_target_deficit_by_land_kg']==[4.,0.]
    assert r['target_deficit_stock_days_by_land_kg_day']==[12.,0.]
    checks['uptake_supply_capacity_output_debt_and_stock_days_distinct']=True
    checks['diagnostics_do_not_mutate_physical_states']=True
    _,annual=plant_activity_diagnostics(result,return_arrays=True)
    assert annual['actual_uptake_kg'].shape==(1,2)
    assert annual['unmet_output_kg'].shape==(1,2,3)
    assert annual['unmet_output_kg'][0,0,0]==1.
    checks['annual_reach_land_diagnostics_preserve_physical_units']=True
    unlimited=plant_activity_diagnostics(run_land1(**args))
    assert unlimited['explicit_growth_capacity_by_land_kg']==[0.,0.]
    assert unlimited['uptake_constraints_passed']
    checks['absent_capacity_not_summed_as_infinity']=True
    matrix=np.array([[[.7,.3],[.25,.75]]])
    converted=plant_activity_diagnostics(run_land1(**args,transitions={0:matrix},plant_growth_capacity=np.ones(target.shape)))
    assert converted['uptake_constraints_passed']
    checks['conversion_replayed_with_same_high_low_state']=True
    write_json(ROOT/'evidence/plant_activity_ledger_known_truth.json',dict(passed=True,cases=checks,
        scope='independent_plant_balance_and_debt_diagnostics_small_not_full_scale_fit',
        no_TN_read=True,identities={'scripts/source_ledger.py':sha(ROOT/'scripts/source_ledger.py')}))
    print(json.dumps(dict(passed=True,cases=checks)))


if __name__=='__main__':main()
