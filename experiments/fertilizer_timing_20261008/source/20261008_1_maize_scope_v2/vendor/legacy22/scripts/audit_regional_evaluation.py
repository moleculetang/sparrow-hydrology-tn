"""Independent arithmetic from saved station mass/water; does not import evaluator."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN

def main():
    selected=rt.read(R/'data/selected.json');freeze=rt.read(R/'data/prediction_freeze.json')
    for p,h in freeze['files'].items():assert rt.sha(R/p)==h,('FROZEN_PREDICTION_CHANGED',p)
    checked=0;events_checked=0;maxerror=0.;draw_identity={}
    pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');obs_e=pd.read_parquet(R/'data/heldout_labels/events/observed_days.parquet')
    for path in (R/'reports').glob('*/1month/station_metrics.csv'):
        comparison=path.parent.parent.name;parts=comparison.split('_');pair=parts[-1];high,low=pair.split('-');table=pd.read_csv(path);paired=pd.read_csv(path.parent/'paired_station_changes.csv');core=pd.read_csv(path.parent/'core_paired_table.csv');ep=pd.read_parquet(path.parent/'paired_events.parquet')
        for (fold,arm),sub in table.groupby(['fold','arm']):
            tag=selected[fold+'_'+(low if arm=='R' else high)];year=2023 if fold=='F23' else 2024
            pred=pd.read_parquet(R/'outputs'/tag/'daily_station_mass_water.parquet');pred=pred[pred.date.dt.year.eq(year)].copy();pred['v']=1000*pred.mass_kg_day/pred.water_m3_day
            assert (pred.water_m3_day>0).all() and np.allclose(pred.v,pred.concentration_mg_l,rtol=1e-14,atol=1e-14)
            obs=pd.read_parquet(R/'data/heldout_labels'/f'{year}_days.parquet');daily=obs.merge(pred[['station_key','date','v']],on=['station_key','date'],validate='one_to_one');daily['month']=daily.date.dt.month
            pm=pred.assign(month=pred.date.dt.month).groupby(['station_key','month']).v.mean().reset_index();monthly=pub[pub.year.eq(year)&pub.model_eligible&pub.tn_mg_l.notna()].merge(pm,on=['station_key','month'],validate='one_to_one').rename(columns={'tn_mg_l':'y'})
            hfm=[]
            for (s,month),g in daily.groupby(['station_key','month']):hfm.append(dict(station_key=s,month=month,y=np.average(g.y,weights=g.n),v=np.average(g.v,weights=g.n)))
            for scale,frame in [('HF_day',daily),('HF_month',pd.DataFrame(hfm)),('PUB_month',monthly)]:
                for row in sub[sub.scale.eq(scale)].itertuples():
                    g=frame[frame.station_key.eq(row.station_key)];y=g.y.to_numpy();p=g.v.to_numpy();e=p-y;den=np.sum((y-y.mean())**2)
                    calc=dict(rmse=np.sqrt(np.mean(e**2)),bias=e.mean(),abs_bias=abs(e.mean()),nse=1-np.sum(e**2)/den if den>0 else np.nan)
                    assert len(g)==row.n
                    for key,value in calc.items():
                        if key=='nse' and not row.nse_eligible:assert pd.isna(row.nse);continue
                        if np.isfinite(value):
                            err=abs(value-getattr(row,key));maxerror=max(maxerror,float(err));assert err<1e-10*(1+abs(value)),(comparison,fold,arm,row.station_key,key,err)
                    checked+=1
            z=obs_e[obs_e.date.dt.year.eq(year)].merge(pred[['station_key','date','v']],on=['station_key','date'],validate='one_to_one')
            for event in ep[ep.fold.eq(fold)].itertuples():
                start=getattr(event,'start_'+arm);end=getattr(event,'end_'+arm);bs=getattr(event,'background_start_'+arm);g=z[z.station_key.eq(event.station_key)];b=g[g.date.ge(bs)&g.date.lt(start)];p=g[g.date.ge(start)&g.date.le(end)];assert len(b)>=4 and len(p)>=1 and bs.year==year
                values=dict(pred_base=float(b.v.median()),pred_peak=float(p.v.max()),obs_base=float(b.y.median()),obs_peak=float(p.y.max()))
                for k,value in values.items():assert abs(value-getattr(event,k+'_'+arm))<1e-10*(1+abs(value))
                assert len(b)==getattr(event,'n_base_'+arm) and len(p)==getattr(event,'n_peak_'+arm);events_checked+=1
        for row in core.itertuples():
            g=paired[paired.fold.eq(row.fold)&paired.scale.eq(row.scale)]
            if row.group!='ALL':g=g[g.cohort.eq(row.group)]
            eligible=g.nse_eligible_R&g.nse_eligible_X&g.nse_R.notna()&g.nse_X.notna();n=g[eligible]
            if len(n):
                assert abs(n.nse_X.median()-n.nse_R.median()-row.nse_difference_of_medians)<1e-10
                assert abs((n.nse_X-n.nse_R).median()-row.nse_median_paired_change)<1e-10
                assert abs((n.nse_X>n.nse_R).mean()-row.nse_improved_fraction)<1e-12
        for block in [1,2]:
            bp=path.parent.parent/f'{block}month';draws=rt.read(bp/'bootstrap_draws.json')
            for fold,sequence in draws.items():
                a=np.asarray(sequence);assert a.shape==(1000,12) and a.min()>=1 and a.max()<=12
                if block==2:assert np.all(a[:,1::2]==a[:,::2]+1)
                key=(2023 if fold=='F23' else 2024,block)
                if key in draw_identity:assert np.array_equal(a,draw_identity[key])
                else:draw_identity[key]=a
            ledger=pd.read_parquet(bp/'bootstrap_ledger.parquet');assert not ledger.duplicated(['fold','replicate','copy_id']).any()
            boot=pd.read_csv(bp/'bootstrap_changes.csv');intervals=pd.read_csv(bp/'bootstrap_intervals.csv')
            for row in intervals.itertuples():
                values=boot[boot.fold.eq(row.fold)&boot.scale.eq(row.scale)][row.metric].dropna().to_numpy();assert len(values)==row.replicates
                lo,hi=np.quantile(values,[.025,.975]);assert abs(lo-row.lower)<1e-10*(1+abs(lo)) and abs(hi-row.upper)<1e-10*(1+abs(hi))
            for fold,seq in draws.items():
                original=ep[ep.fold.eq(fold)]
                if original.empty:continue
                for rep in [0,499,999]:
                    chunks=[original[original.event_month.eq(month)].assign(sample_copy=i) for i,month in enumerate(seq[rep])]
                    resample=pd.concat(chunks,ignore_index=True)
                    if resample.empty:continue
                    target=boot[boot.fold.eq(fold)&boot.scale.eq('events')&boot.replicate.eq(rep)].iloc[0]
                    for metric in ['amplitude_error','peak_error','base_error']:
                        if pd.isna(target[metric+'_change']):continue
                        station=resample.groupby('station_key')[[metric+'_R',metric+'_X']].median();change=float((station[metric+'_X']-station[metric+'_R']).mean())
                        assert abs(change-target[metric+'_change'])<1e-10*(1+abs(change))
    for p,h in rt.read(R/'evidence/inherited_manifest.json').items():assert rt.sha(p)==h,('OLD_FILE_CHANGED',p)
    rt.write(R/'reports/independent_regional_evaluation.json',dict(status='PASS',station_scales_checked=checked,events_checked=events_checked,max_metric_error=maxerror,both_NSE_definitions=True,shared_month_draws=True,old_files_unchanged=True,frozen_predictions_unchanged=True))
    print('PASS independent regional evaluation',checked,events_checked,flush=True)
if __name__=='__main__':main()
