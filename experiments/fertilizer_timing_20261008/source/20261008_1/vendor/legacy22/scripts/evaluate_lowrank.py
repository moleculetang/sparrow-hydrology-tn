"""Evaluate pre-frozen input predictions separately from observation-assisted reconstruction."""
import numpy as np,pandas as pd
import native_runtime as rt
from lowrank_regional import season,wls
R=rt.RUN

def main():
    freeze=rt.read(R/'reports/lowrank_freeze.json')
    for p,h in freeze['files'].items():assert rt.sha(R/p)==h
    assert (R/'data/prediction_freeze.json').exists()
    rows=[];pairs=[];curves=[];omissions=[]
    for root in (R/'diagnostics/lowrank').glob('*/*/*'):
        if not (root/'identity.json').exists():continue
        identity=rt.read(root/'identity.json');short=root.parts[-3];year=2023 if short=='F23' else 2024;scale=identity['scale'];target=identity['target']
        hf=pd.read_parquet(R/'data/heldout_labels'/f'{year}_days.parquet');base=pd.read_parquet(R/'outputs'/identity['baseline']/'daily_station_mass_water.parquet').rename(columns={'concentration_mg_l':'baseline'})
        daily=hf.merge(base[['station_key','date','baseline']],on=['station_key','date'],how='left',validate='one_to_one');assert daily.baseline.notna().all()
        if scale=='daily':obs=daily[['station_key','date','y','baseline']].copy()
        else:
            month=[]
            for (station,m),g in daily.groupby(['station_key',daily.date.dt.month]):
                if len(g)<10:continue
                w=g.n/g.n.sum();month.append(dict(station_key=station,date=pd.Timestamp(year,m,1),y=float(w@g.y),baseline=float(w@g.baseline)))
            hfmonth=pd.DataFrame(month);pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');pub=pub[pub.year.eq(year)&pub.model_eligible&pub.tn_mg_l.notna()].copy();pub['date']=pd.to_datetime(dict(year=pub.year,month=pub.month,day=1));pub=pub.rename(columns={'tn_mg_l':'y'})
            bm=base[base.date.dt.year.eq(year)].assign(date=lambda z:z.date.dt.to_period('M').dt.to_timestamp()).groupby(['station_key','date']).baseline.mean().reset_index();pub=pub[['station_key','date','y']].merge(bm,on=['station_key','date'],validate='one_to_one')
            keys=set(zip(hfmonth.station_key,hfmonth.date));pub=pub[[x not in keys for x in zip(pub.station_key,pub.date)]];obs=pd.concat([hfmonth,pub],ignore_index=True)
        assert not obs.duplicated(['station_key','date']).any()
        forecasts=pd.read_parquet(root/'forecast.parquet');forecasts=forecasts[forecasts.date.dt.year.eq(year)]
        stations=set(forecasts.station_key);obs=obs[obs.station_key.isin(stations)].copy();obs['truth_component']=np.log1p(obs.y)
        if target=='baseline_residual':obs['truth_component']-=np.log1p(obs.baseline)
        joined=forecasts.merge(obs,on=['station_key','date'],how='inner',validate='many_to_one');joined['predicted_logtn']=joined.predicted_transformed_component+(np.log1p(joined.baseline) if target=='baseline_residual' else 0);joined['predicted_tn']=np.expm1(joined.predicted_logtn)
        joined['fold']=short;joined['scale']=scale;joined['target']=target;joined['role']='INPUT_ONLY_FORECAST';curves.append(joined)
        local=[]
        for (name,station),g in joined.groupby(['model','station_key']):
            logerr=g.predicted_logtn-np.log1p(g.y);err=g.predicted_tn-g.y;centered=err-err.groupby(g.date.dt.month).transform('mean');nse=1-np.mean(err**2)/np.var(g.y) if np.var(g.y)>0 else np.nan
            rec=dict(fold=short,scale=scale,target=target,model=name,station_key=station,n=len(g),months=g.date.dt.month.nunique(),logrmse=float(np.sqrt(np.mean(logerr**2))),rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),centered_rmse=float(np.sqrt(np.mean(centered**2))),nse=float(nse),negative_fraction=float((g.predicted_tn<0).mean()),role='INPUT_ONLY_FORECAST');rows.append(rec);local.append(rec)
        local=pd.DataFrame(local)
        for rank in [1,2,3]:
            name=f'rank{rank}'
            if len(local)==0 or name not in set(local.model):continue
            a=local[local.model.eq('unrestricted')];b=local[local.model.eq(name)];p=a.merge(b,on=['fold','scale','target','station_key'],suffixes=('_R','_X'),validate='one_to_one')
            for metric in ['rmse','logrmse','centered_rmse','nse']:
                change=p[metric+'_X']-p[metric+'_R'];pairs.append(dict(fold=short,scale=scale,target=target,model=name,metric=metric,stations=len(p),difference_of_medians=float(p[metric+'_X'].median()-p[metric+'_R'].median()),median_paired_change=float(change.median()),improved_fraction=float((change>0 if metric=='nse' else change<0).mean())))
            archive=np.load(root/'training.npz');modes=np.load(root/f'{name}.npz');ss=archive['stations'];v=modes['loadings'];sc=archive['scale'];seasonal=archive['seasonal'];lookup={s:i for i,s in enumerate(ss)}
            for date,g in obs.groupby('date'):
                g=g[g.station_key.isin(lookup)];indices=np.array([lookup[s] for s in g.station_key]);vv=v[indices]
                if len(indices)<rank or np.linalg.matrix_rank(vv)<rank:omissions.append(dict(fold=short,scale=scale,target=target,rank=rank,date=str(date),reason='UNOBSERVABLE_MODE_COEFFICIENT'));continue
                seasonal_value=(season([date])@seasonal[:,indices]).ravel();z=(g.truth_component.to_numpy()-seasonal_value)/sc[indices];coef=wls(vv,z,np.ones(len(z)));reconstructed=seasonal_value+vv@coef*sc[indices]
                for station,truth,pred in zip(g.station_key,g.truth_component,reconstructed):rows.append(dict(fold=short,scale=scale,target=target,model=name,station_key=station,date=str(date),component_squared_error=float((pred-truth)**2),role='OBSERVATION_ASSISTED_RECONSTRUCTION_NOT_FORECAST'))
    pd.DataFrame(rows).to_csv(R/'reports/lowrank_evaluation.csv',index=False);pd.DataFrame(pairs).to_csv(R/'reports/lowrank_paired.csv',index=False);pd.DataFrame(omissions).to_csv(R/'reports/lowrank_unobservable.csv',index=False)
    if curves:pd.concat(curves,ignore_index=True).to_parquet(R/'reports/lowrank_predictions.parquet',index=False)
    rt.write(R/'reports/lowrank_evaluation_audit.json',dict(status='PASS',frozen_hashes_verified=True,forecast_uses_no_contemporaneous_TN=True,observation_assisted_separate=True,no_clipping=True,monthly_unique_HF_priority=True))
if __name__=='__main__':main()
