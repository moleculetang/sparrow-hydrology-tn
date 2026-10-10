"""44-path registration and real training-only target isolation, no evaluation score."""
import sys,json,inspect,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from d29_training.objective import StrategyObjective
from d29_platform.runtime import write_json,sha
from timing.inputs import TimingInputs

def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text());assert len(jobs)==44
    assert len({j['id'] for j in jobs})==44
    assert all(j['fixed_log_source_multiplier']==0. and j['strategy']=='T2' for j in jobs)
    fixed=[j for j in jobs if j['space_block'] is None and j['calendar']!='K']
    constrained=[j for j in jobs if j['space_block'] is None and j['calendar']=='K']
    spatial=[j for j in jobs if j['space_block'] is not None]
    assert (len(fixed),len(constrained),len(spatial))==(12,8,24)
    assert all(j['entry']==0 for j in fixed+spatial)
    assert all(j['train_end']==(2022 if j['fold']=='F23' else 2023) for j in fixed+constrained)
    assert all(j['train_end']==2023 and j['space_block'] in (56,113,191) for j in spatial)
    assert all(j['priority']==3 for j in constrained if j['entry']==1)
    assert 'TN' not in str(inspect.signature(TimingInputs))
    m=pd.read_parquet(ROOT/'data/monthly_tn.parquet');h=pd.read_parquet(ROOT/'data/hf_daily.parquet')
    blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text())
    cases=[]
    for support,end,excluded in [('F23',2022,[]),('F24',2023,[])]+[(f'S{s}',2023,blocks[str(s)]['held_stations']+blocks[str(s)]['buffer_stations']) for s in (56,113,191)]:
        baseline=StrategyObjective(m[m.year.le(end)],h[h.date.dt.year.le(end)],'T2',end,excluded)
        # Overwrite future and held/buffer labels without examining their values.
        mm=m.copy();hh=h.copy();mb=(mm.year>end)|mm.station_key.isin(excluded)
        hb=(hh.date.dt.year>end)|hh.station_key.isin(excluded)
        mm.loc[mb,'tn_mg_l']=1e9;hh.loc[hb,'tn_mg_l']=-1e9
        changed=StrategyObjective(mm[mm.year.le(end)],hh[hh.date.dt.year.le(end)],'T2',end,excluded)
        assert baseline.long_sites==changed.long_sites and baseline.scales==changed.scales
        cols=[c for c in ['station_key','year','month','date','kind','tn_mg_l','read_count'] if c in baseline.rows]
        pd.testing.assert_frame_equal(baseline.rows[cols],changed.rows[cols])
        pred=np.sin(np.arange(len(baseline.rows),dtype=float)/17)+2
        v,g,t,_=baseline.evaluate(pred);v2,g2,t2,_=changed.evaluate(pred)
        assert v==v2 and np.array_equal(g,g2) and t==t2
        assert not baseline.rows.station_key.isin(excluded).any()
        assert baseline.rows.year.max()<=end
        cases.append(dict(support=support,train_end=end,excluded_station_count=len(set(excluded)),
            training_records=len(baseline.rows),evaluation_or_held_labels_changed=True,
            grouping_scales_objective_gradient_unchanged=True))
    report=dict(passed=True,scope='matrix_and_training_target_isolation_not_physical_prediction_gate',
       paths=44,fixed_time=12,K_time=8,space=24,cases=cases,no_evaluation_scores_computed=True,
       physical_initialization_and_prediction_isolation_pending=True,
       identities={p:sha(ROOT/p) for p in ['config/jobs.json','d29_training/objective.py']})
    write_json(ROOT/'evidence/matrix_training_contract.json',report);print(json.dumps(report))

if __name__=='__main__':main()
