"""Known-value and refusal probes; no TN labels, physical fit or formal seal."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from bnf_calendar_partition import partition,LABELS

ROOT=Path(__file__).resolve().parents[1]


def main():
    columns=['year','reach_id','crop','bnf_plant_kg','bnf_soil_kg','harvest_n_kg','residue_n_kg']
    activity=pd.DataFrame([[2024,1,'known',8.,4.,10.,0.],
                           [2024,1,'unknown',4.,8.,10.,0.],
                           [2024,2,'zero_plan',4.,0.,0.,0.]],columns=columns)
    tags=np.zeros((4,2,1,5,4),dtype=np.float64)
    tags[:,0,0,3,0]=3.;tags[:,0,0,3,3]=3.;tags[:,1,0,3,0]=1.
    tags[...,0,3]=2.
    names=['deposition','fertilizer','field_manure','crop_BNF','initial_research_state']
    profiles={'known':{'1':{'seasons':'fixture'}},'zero_plan':{'2':{'seasons':'fixture'}}}
    grow=lambda year,seasons:np.array([0.,.5,.5,0.])
    results={}
    actual,rows,receipt=partition(2024,tags,names,activity,profiles,grow,.6,.4)
    assert np.array_equal(actual.sum(3),tags.sum(3)) and np.min(actual)==0.
    assert np.array_equal(actual[:,0,0,3,0],[2.,0.,0.,2.])
    assert np.array_equal(actual[:,0,0,2,0],[0.,2.,2.,0.])
    assert np.array_equal(actual[:,0,0,4,0],[1.,1.,1.,1.])
    assert np.array_equal(actual[:,0,0,4,3],[2.,2.,2.,2.])
    assert np.array_equal(actual[:,1,0,3,0],np.ones(4))
    assert rows[1]['nongrowing_days'] is None and rows[1]['plant_BNF_outside_growth_kg'] is None
    assert rows[2]['zero_growth_plan'] and rows[2]['nongrowing_days']==4
    results['known_growth_and_unknown_not_false_fallow']=True
    results['two_independent_BNF_entrances_and_zero_growth_plan']=True
    before,_,early=partition(2015,tags,names,activity,profiles,grow,.6,.4)
    assert np.array_equal(before[...,1,:],tags[...,3,:]) and np.all(before[...,2:,:]==0.)
    assert np.array_equal(before.sum(3),tags.sum(3)) and not early['temporal_calendar_inferred_before2016']
    results['no_calendar_intervention_during_warmup']=True
    known_zero=activity.copy();known_zero[['bnf_plant_kg','bnf_soil_kg']]=0.
    z=tags.copy();z[...,3,:]=0.
    empty,_,_=partition(2024,z,names,known_zero,profiles,grow,.6,.4)
    assert np.isfinite(empty).all() and np.array_equal(empty.sum(3),z.sum(3))
    results['known_zero_without_epsilon_or_nan']=True
    def refuses(label,**kw):
        args=dict(year=2024,tagged_sources=tags,labels=names,activity=activity,
                  profiles=profiles,growth_profile=grow,returned_fraction=.6,exported_fraction=.4)
        args.update(kw)
        try:partition(**args)
        except ValueError:results[label]=True
        else:raise AssertionError('WRONG_INPUT_ACCEPTED '+label)
    wrong=activity.copy();wrong.loc[0,'bnf_plant_kg']=9.
    refuses('annual_BNF_mass_change_rejected',activity=wrong)
    refuses('duplicate_crop_budget_rejected',activity=pd.concat([activity,activity.iloc[:1]],ignore_index=True))
    unknown=activity.copy();unknown.loc[0,'bnf_plant_kg']=np.nan
    refuses('unknown_BNF_mass_rejected',activity=unknown)
    refuses('missing_activity_not_known_zero',activity=activity.iloc[:0])
    missing=tags.copy();missing[:,1,:,3,:]=0.
    refuses('missing_reach_not_known_zero',activity=activity[activity.reach_id.eq(1)],tagged_sources=missing)
    refuses('double_residue_fate_rejected',returned_fraction=.8,exported_fraction=.4)
    wrong=tags.copy();wrong[0,0,0,3,1]=.1
    refuses('unregistered_BNF_entry_rejected',tagged_sources=wrong)
    refuses('invalid_growth_profile_rejected',growth_profile=lambda y,s:np.full(4,.5))
    wrong=tags.copy();wrong[0,0,0,0,3]=-1.
    refuses('negative_source_not_clipped',tagged_sources=wrong)
    result=dict(passed=all(results.values()),scope='synthetic_BNF_calendar_partition_not_actual44_model',
                cases=results,labels=list(LABELS),evaluation_labels_read=False,formal_gate_issued=False)
    result['implementation_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        for p in ('analysis/bnf_calendar_partition.py','analysis/test_bnf_calendar_partition.py')}
    folder=ROOT/'outputs/bnf_partition_software_controls';folder.mkdir(exist_ok=True)
    path=folder/'known_values_and_refusals.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
