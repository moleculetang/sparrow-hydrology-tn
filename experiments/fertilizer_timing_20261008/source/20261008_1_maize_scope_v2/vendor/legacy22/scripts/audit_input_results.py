"""Independent tabular re-aggregation, generated-input closure and frozen-file audit."""
import json,time
import native_runtime as rt
from pathlib import Path
import numpy as np,pandas as pd
R=rt.RUN

def main():
    checks=[];freeze=rt.read(R/'data/prediction_freeze.json')
    for p,h in freeze['files'].items():assert rt.sha(R/p)==h,('CHANGED_FROZEN_PREDICTION',p)
    selected=rt.read(R/'data/selected.json');assert len(selected)==12
    for key,tag in selected.items():
        a=rt.read(R/'reports/independent'/f'{tag}.json');b=rt.read(R/'outputs'/tag/'full_source_audit.json');assert a['status']=='PASS' and b['status']=='PASS'
    for table in (R/'reports').glob('*_*-*/1month/core_paired_table.csv'):
        paired=pd.read_csv(table.parent/'paired_station_changes.csv');core=pd.read_csv(table)
        for row in core.to_dict('records'):
            q=paired[(paired.fold==row['fold'])&(paired.scale==row['scale'])]
            if row['group']!='ALL':q=q[q.cohort==row['group']]
            assert len(q)==row['paired_stations']
            eligible=q[q.nse_eligible_R&q.nse_eligible_X&q.nse_R.notna()&q.nse_X.notna()]
            if len(eligible):
                a=float(np.median(eligible.nse_X)-np.median(eligible.nse_R));b=float(np.median(eligible.nse_X-eligible.nse_R));c=float(np.mean(eligible.nse_X>eligible.nse_R))
                assert abs(a-row['nse_difference_of_medians'])<1e-10 and abs(b-row['nse_median_paired_change'])<1e-10 and abs(c-row['nse_improved_fraction'])<1e-12
            for metric in ['rmse','bias','abs_bias']:
                assert np.isclose(np.mean(q[metric+'_X']-q[metric+'_R']),row[metric+'_change'],rtol=1e-11,atol=1e-11)
        events=pd.read_parquet(table.parent/'paired_events.parquet')
        for fold,n in [('F23',40),('F24',57)]:
            q=events[events.fold==fold];assert len(q)==n,(table,fold,len(q))
            for k in ['obs_base','obs_peak','n_base','n_peak']:assert np.array_equal(q[k+'_R'],q[k+'_X'])
        ledger=pd.read_parquet(table.parent/'bootstrap_ledger.parquet');assert not ledger.duplicated(['fold','replicate','copy_id']).any()
        assert ledger.groupby('fold').replicate.nunique().eq(1000).all()
        checks.append(dict(comparison=table.parent.parent.name,rows=len(core),event_counts=events.groupby('fold').size().to_dict(),bootstrap_replicates=1000))
    assert len(checks)==6,checks
    # Every synthetic noise/scalar objective has a declared original support.
    scenarios=0
    for p in (R/'outputs/synthetic').glob('*_*/status.json'):
        a=rt.read(p);assert a['status']=='COMPLETE' and a['scenarios']==8;scenarios+=a['scenarios']
        d=pd.read_csv(p.parent/'metrics.csv');assert d.scalar_calls.max()<=200
        for mode in ['true','uniform','month_first','shift14','previous_year','area','mass_067','combined']:
            pred=np.load(p.parent/(mode+'_predictions.npz'));truth=pred['oracle'];m=d[(d.input_mode==mode)&(d.target=='clean')&(d.fold=='F24')&(d.support=='observed_dates')]
            assert len(m)==3
            if mode=='true':assert np.array_equal(pred['uncorrected'],truth)
    assert scenarios==48
    rt.write(R/'reports/independent_completion_audit.json',dict(status='PASS',selected=len(selected),comparisons=checks,synthetic_scenarios=scenarios,frozen_files=len(freeze['files']),limits='Independent program and process, not external expert review; numerical sufficiency retained separately',finished=time.time()))
    print('PASS INDEPENDENT COMPLETION',flush=True)
if __name__=='__main__':main()
