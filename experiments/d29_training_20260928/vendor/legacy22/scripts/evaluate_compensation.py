"""Post-freeze descriptive four-corner and demand-calendar evaluation."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def scores(frame):
    rows=[]
    for station,g in frame.groupby('station_key'):
        e=g.p-g.y;rows.append(dict(station_key=station,n=len(g),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),mean_squared_error=float(np.mean(e*e))))
    return pd.DataFrame(rows)
def main():
    assert (R/'data/prediction_freeze.json').exists();selected=rt.read(R/'data/selected.json');rows=[]
    for key,tag in selected.items():
        fold,structure,mode=key.split('_')
        if mode=='P':continue
        year=2023 if fold=='F23' else 2024;obs=pd.read_parquet(R/f'data/heldout_labels/{year}_days.parquet')[['station_key','date','y']]
        paths={'baseline':R/'outputs'/selected[f'{fold}_{structure}_P']/'daily_station_mass_water.parquet','joint':R/'outputs'/tag/'daily_station_mass_water.parquet'}
        paths.update({name:R/'outputs'/tag/'counterfactuals'/(name+'.parquet') for name in ['input_and_c_only','process_only']})
        for corner,p in paths.items():
            pred=pd.read_parquet(p).rename(columns={'concentration_mg_l':'p'});g=obs.merge(pred[['station_key','date','p']],on=['station_key','date'],validate='one_to_one');assert len(g)==len(obs)
            s=scores(g);s['fold']=fold;s['structure']=structure;s['input']=mode;s['corner']=corner;rows.append(s)
    allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(R/'reports/four_corner_station_metrics.csv',index=False)
    summary=allrows.groupby(['fold','structure','input','corner'])[['rmse','bias','mean_squared_error']].mean().reset_index();summary.to_csv(R/'reports/four_corner_summary.csv',index=False)
    support=pd.read_parquet(R/'outputs/synthetic/synthetic_date_support.parquet')[['station_key','date']];obs=pd.read_parquet(R/'data/heldout_labels/2024_days.parquet')[['station_key','date','y']];demand=[]
    for structure in ['U','L3']:
        for mode in ['P','D']:
            for daily in [False,True]:
                p=R/f'outputs/synthetic/demand_calendar/{structure}_{mode}_demand_{daily}.npy';pred=support.assign(p=np.load(p));g=obs.merge(pred,on=['station_key','date'],validate='one_to_one');assert len(g)==len(obs)
                s=scores(g);s['structure']=structure;s['source_mode']=mode;s['daily_demand']=daily;demand.append(s)
    d=pd.concat(demand);d.to_csv(R/'reports/demand_calendar_station_metrics.csv',index=False);d.groupby(['structure','source_mode','daily_demand'])[['rmse','bias']].mean().reset_index().to_csv(R/'reports/demand_calendar_summary.csv',index=False)
    rt.write(R/'reports/compensation_evaluation_audit.json',dict(status='PASS',four_corner_station_rows=len(allrows),demand_calendar_station_rows=len(d),role='Post-freeze descriptive diagnostics; no fitting or candidate selection'))
    print('PASS COMPENSATION EVALUATION',flush=True)
if __name__=='__main__':main()
