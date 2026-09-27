"""Independent arithmetic over frozen prediction products; no evaluation helpers."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def main():
    selected=rt.read(R/'data/selected.json');published=pd.read_csv(R/'reports/station_metrics.csv');errors=[]
    events=pd.read_parquet(R/'data/heldout_labels/events/events_frozen.parquet');obs_e=pd.read_parquet(R/'data/heldout_labels/events/observed_days.parquet')
    pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet')
    count=0;ecount=0
    for key,tag in selected.items():
        fold,arm=key.split('_');year=2023 if fold=='F23' else 2024
        pred=pd.read_parquet(R/'outputs'/tag/'daily_station_mass_water.parquet');pred=pred[pred.date.dt.year.eq(year)].copy()
        assert (pred.water_m3_day>0).all();pred['v']=pred.mass_kg_day/pred.water_m3_day*1000
        assert np.allclose(pred.v,pred.concentration_mg_l,rtol=1e-14,atol=1e-14)
        obs=pd.read_parquet(R/'data/heldout_labels'/f'{year}_days.parquet');j=obs.merge(pred[['station_key','date','v']],on=['station_key','date'],validate='one_to_one');assert len(j)==len(obs)
        pm=pred.assign(year=year,month=pred.date.dt.month).groupby(['station_key','year','month']).v.mean().reset_index()
        monthly=pub[pub.year.eq(year)&pub.model_eligible&pub.tn_mg_l.notna()].merge(pm,on=['station_key','year','month'],validate='one_to_one').rename(columns={'tn_mg_l':'y'})
        hfm=[]
        for (s,month),g in j.groupby(['station_key','month']):
            hfm.append(dict(station_key=s,month=month,y=float(np.average(g.y,weights=g.n)),v=float(np.average(g.v,weights=g.n))))
        for scale,frame in [('HF_day',j),('HF_month',pd.DataFrame(hfm)),('PUB_month',monthly)]:
            for s,g in frame.groupby('station_key'):
                y=g.y.to_numpy();p=g.v.to_numpy();se=(p-y)**2;rmse=float(np.sqrt(se.sum()/len(y)));bias=float((p-y).sum()/len(y));den=float(((y-y.mean())**2).sum());nse=1-float(se.sum())/den if den>0 else np.nan
                row=published[(published.fold==fold)&(published.arm==arm)&(published.scale==scale)&(published.station_key==s)].iloc[0]
                for k,val in [('rmse',rmse),('bias',bias),('nse',nse)]:
                    if k=='nse' and not row.nse_eligible:
                        assert pd.isna(row.nse);continue
                    if np.isfinite(val):errors.append(abs(val-row[k]));assert abs(val-row[k])<1e-10*(1+abs(val)),(key,s,k)
                count+=1
        z=obs_e[obs_e.date.dt.year.eq(year)].merge(pred[['station_key','date','v']],on=['station_key','date'],validate='one_to_one')
        ep=pd.read_parquet(R/'outputs'/tag/'event_scores.parquet')
        for e in ep.itertuples():
            data=z[z.station_key.eq(e.station_key)];b=data[data.date.ge(e.background_start)&data.date.lt(e.start)];p=data[data.date.ge(e.start)&data.date.le(e.end)]
            assert len(b)>=4 and len(p)>=1 and e.background_start.year==year
            bp=float(np.median(b.v));pp=float(np.max(p.v));bo=float(np.median(b.y));po=float(np.max(p.y))
            for name,val in [('pred_base',bp),('pred_peak',pp),('obs_base',bo),('obs_peak',po)]:assert abs(val-getattr(e,name))<1e-11*(1+abs(val))
            ecount+=1
    paired=pd.read_csv(R/'reports/paired_station_changes.csv');core=pd.read_csv(R/'reports/core_paired_table.csv')
    for row in core.itertuples():
        g=paired[(paired.fold==row.fold)&(paired.scale==row.scale)]
        if row.group!='ALL':g=g[g.cohort==row.group]
        g=g[g.nse_eligible_R&g.nse_eligible_X&g.nse_R.notna()&g.nse_X.notna()]
        if len(g):
            assert abs((np.median(g.nse_X)-np.median(g.nse_R))-row.nse_difference_of_medians)<1e-10
            assert abs(np.median(g.nse_X-g.nse_R)-row.nse_median_paired_change)<1e-10
    rt.write(R/'reports/independent_evaluation.json',dict(status='PASS',station_scales_checked=count,events_checked=ecount,max_metric_error=max(errors,default=0),both_NSE_median_definitions=True))
    print('PASS independent metrics',count,ecount)
if __name__=='__main__':main()
