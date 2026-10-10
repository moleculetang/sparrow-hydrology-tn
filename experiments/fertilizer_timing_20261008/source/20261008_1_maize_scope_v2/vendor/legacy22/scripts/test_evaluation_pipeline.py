"""End-to-end evaluation on fabricated labels only; no scientific predictions."""
import numpy as np,pandas as pd
import native_runtime as rt
import evaluation as ev
R=rt.RUN/'work/evaluation_pipeline_fixture';R.mkdir(exist_ok=True)
ev.R=R
obs=R/'data/heldout_labels';(obs/'events').mkdir(parents=True,exist_ok=True)
dates=pd.date_range('2024-01-01','2024-12-31');stations=[f's{i:02}' for i in range(15)]
rows=[];events=[]
for i,s in enumerate(stations):
    for d in dates:rows.append(dict(station_key=s,date=d,y=2+i*.01+np.sin(d.dayofyear*.08)*.5,n=6))
    for rank,month in enumerate([1,4,7,10] if i<12 else [1,4,7]):
        start=pd.Timestamp(2024,month,10)
        events.append(dict(station_key=s,event_rank=rank,start=start,end=start+pd.Timedelta(days=2),background_start=start-pd.Timedelta(days=7)))
    events.append(dict(station_key=s,event_rank=99,start=pd.Timestamp('2024-01-02'),end=pd.Timestamp('2024-01-03'),background_start=pd.Timestamp('2023-12-26')))
days=pd.DataFrame(rows);days.to_parquet(obs/'2024_days.parquet',index=False);days.to_parquet(obs/'events/observed_days.parquet',index=False)
pd.DataFrame(events).to_parquet(obs/'events/events_frozen.parquet',index=False)
pub=days.assign(year=2024,month=days.date.dt.month).groupby(['station_key','year','month']).y.mean().reset_index().rename(columns={'y':'tn_mg_l'});pub['model_eligible']=True;pub.to_parquet(obs/'monthly_original.parquet',index=False)
pd.DataFrame(dict(station_key=stations,cohort=['N']*15)).to_parquet(R/'data/station_registry.parquet',index=False)
for tag,delta in [('R',.5),('X',.2)]:
    out=R/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
    pred=days[['station_key','date']].copy();pred['concentration_mg_l']=days.y+delta;pred['water_m3_day']=100.
    pred.to_parquet(out/'daily_station_mass_water.parquet',index=False)
for block in [1,2]:
    ev.main({'F24_R':'R','F24_X':'X'},'synthetic',block)
    out=R/'reports/synthetic'/f'{block}month'
    core=pd.read_csv(out/'core_paired_table.csv');assert np.allclose(core.rmse_change,-.3)
    e=pd.read_csv(out/'event_centered_summary.csv');assert e.events.iloc[0]==57 and np.isclose(e.peak_error_change.iloc[0],-.3)
    assert e.model_ratio_undefined_R.iloc[0]==0 and e.model_ratio_undefined_X.iloc[0]==0 and np.isfinite(e.amplitude_error_change.iloc[0])
    ledger=pd.read_parquet(out/'bootstrap_ledger.parquet');assert not ledger.duplicated(['fold','replicate','copy_id']).any()
rt.write(rt.RUN/'reports/evaluation_pipeline_fixture.json',dict(status='PASS',synthetic_only=True,events=57,stations=15,blocks=[1,2],replicates=1000,expected_rmse_change=-.3))
print('PASS synthetic evaluation pipeline',flush=True)
