"""End-to-end synthetic evaluation, completely separate from scientific outputs."""
import numpy as np,pandas as pd
import native_runtime as rt
import evaluation,audit_evaluation
R=rt.RUN
def main():
    root=R/'work/reporting_fixture';root.mkdir(exist_ok=True)
    for folder in ['reports','data/heldout_labels/events','outputs/R','outputs/X']:(root/folder).mkdir(parents=True,exist_ok=True)
    dates=pd.date_range('2023-01-01','2023-12-31');obs=[];preds={'R':[],'X':[]};events=[];pub=[]
    for s,cohort in [('s1','N'),('s2','H'),('s3','X')]:
        y=2+np.sin(np.arange(len(dates))/30)
        for t,v in zip(dates,y):
            obs.append(dict(station_key=s,date=t,y=v,n=6,year=2023,month=t.month))
            for arm,offset in [('R',.4),('X',.2)]:preds[arm].append(dict(station_key=s,date=t,concentration_mg_l=v+offset,water_m3_day=1000.,mass_kg_day=v+offset))
        for month in range(1,13):pub.append(dict(station_key=s,year=2023,month=month,tn_mg_l=float(y[dates.month==month].mean()),model_eligible=True))
        events.append(dict(station_key=s,event_rank=0,start=pd.Timestamp('2023-03-10'),end=pd.Timestamp('2023-03-12'),background_start=pd.Timestamp('2023-03-03')))
    obs=pd.DataFrame(obs);obs.to_parquet(root/'data/heldout_labels/2023_days.parquet');obs.to_parquet(root/'data/heldout_labels/events/observed_days.parquet')
    pd.DataFrame(events).to_parquet(root/'data/heldout_labels/events/events_frozen.parquet');pd.DataFrame(pub).to_parquet(root/'data/heldout_labels/monthly_original.parquet')
    pd.DataFrame(dict(station_key=['s1','s2','s3'],cohort=['N','H','X'])).to_parquet(root/'data/station_registry.parquet')
    for arm,rows in preds.items():pd.DataFrame(rows).to_parquet(root/'outputs'/arm/'daily_station_mass_water.parquet')
    selected={'F23_R':'R','F23_X':'X'};rt.write(root/'data/selected.json',selected)
    evaluation.R=root;evaluation.main(selected);audit_evaluation.R=root;audit_evaluation.main()
    core=pd.read_csv(root/'reports/core_paired_table.csv');assert (core.rmse_change<0).all() and (core.nse_median_paired_change>0).all()
    rt.write(R/'reports/reporting_pipeline_fixture.json',dict(status='PASS',synthetic_only=True,independent_audit=rt.read(root/'reports/independent_evaluation.json')))
    print('PASS reporting pipeline')
if __name__=='__main__':main()
