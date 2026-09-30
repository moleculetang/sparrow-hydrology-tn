"""Evaluation is called only after every path has frozen predictions or an audited omission."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
REPORT=R/'reports'

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('UNDEFINED_REQUIRED_PREDICTION')
    e=p-y;sy=float(np.std(y));sp=float(np.std(p));v=sy*sy
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),abs_bias=float(abs(e.mean())),nse=float(1-np.mean(e*e)/v) if v>0 else np.nan,r=float(np.corrcoef(y,p)[0,1]) if sy>0 and sp>0 else np.nan,logrmse=float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))),amplitude_ratio=sp/sy if sy>0 else np.nan)

def event_scores(pred,obs,events,year):
    joined=obs[obs.date.dt.year.eq(year)].merge(pred[['station_key','date','p']],on=['station_key','date'],how='left',validate='one_to_one')
    assert joined.p.notna().all()
    groups={s:g.set_index('date') for s,g in joined.groupby('station_key')};rows=[]
    ev=events[(events.start.dt.year==year)&(events.end.dt.year==year)]
    for e in ev.to_dict('records'):
        g=groups.get(e['station_key'])
        if g is None:continue
        b=g[(g.index>=e['background_start'])&(g.index<e['start'])];p=g[(g.index>=e['start'])&(g.index<=e['end'])]
        eligible=len(b)>=4 and len(p)>=1 and e['background_start'].year==year
        row=dict(**e,n_base=len(b),n_peak=len(p),eligible=eligible,event_month=e['start'].month)
        if eligible:
            for col,name in [('y','obs'),('p','pred')]:row[name+'_base']=float(b[col].median());row[name+'_peak']=float(p[col].max())
            defined=min(row[k] for k in ['obs_base','obs_peak','pred_base','pred_peak'])>0
            row['ratio_defined']=defined
            row['amplitude_error']=abs(np.log(row['pred_peak']/row['pred_base'])-np.log(row['obs_peak']/row['obs_base'])) if defined else np.nan
            row['peak_error']=abs(row['pred_peak']-row['obs_peak']);row['base_error']=abs(row['pred_base']-row['obs_base'])
        rows.append(row)
    return pd.DataFrame(rows)

def paired_summary(table):
    rows=[]
    for (fold,scale,group),g in table.groupby(['fold','scale','group']):
        eligible=g.nse_eligible_R&g.nse_eligible_X&g.nse_R.notna()&g.nse_X.notna();n=g[eligible]
        row=dict(fold=fold,scale=scale,group=group,paired_stations=len(g),nse_paired_stations=len(n),nse_median_R=float(n.nse_R.median()) if len(n) else np.nan,nse_median_X=float(n.nse_X.median()) if len(n) else np.nan,nse_difference_of_medians=float(n.nse_X.median()-n.nse_R.median()) if len(n) else np.nan,nse_median_paired_change=float((n.nse_X-n.nse_R).median()) if len(n) else np.nan,nse_improved_fraction=float((n.nse_X>n.nse_R).mean()) if len(n) else np.nan)
        for k in ['rmse','bias','abs_bias','logrmse','r','amplitude_ratio']:
            row[k+'_R']=float(g[k+'_R'].mean());row[k+'_X']=float(g[k+'_X'].mean());row[k+'_change']=float((g[k+'_X']-g[k+'_R']).mean())
            row[k+'_difference_of_medians']=float(g[k+'_X'].median()-g[k+'_R'].median());row[k+'_median_paired_change']=float((g[k+'_X']-g[k+'_R']).median())
            if k in ['rmse','abs_bias']:row[k+'_improved_fraction']=float((g[k+'_X']<g[k+'_R']).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def boot_metrics(joined,draws,qualified):
    """Sufficient-statistic month resampling; all stations share each month sequence."""
    ss=sorted(joined.station_key.unique());idx={s:i for i,s in enumerate(ss)};arr=np.zeros((len(ss),12,7))
    for (s,m),g in joined.groupby(['station_key','month']):
        y=g.y.to_numpy();r=g.p_R.to_numpy();x=g.p_X.to_numpy()
        arr[idx[s],m-1]=[len(g),y.sum(),(y*y).sum(),(r-y).sum(),((r-y)**2).sum(),(x-y).sum(),((x-y)**2).sum()]
    qual=np.array([s in qualified for s in ss]);rows=[]
    for rep,months in enumerate(draws):
        counts=np.bincount(months,minlength=13)[1:];v=np.einsum('smk,m->sk',arr,counts);use=v[:,0]>0;v=v[use];q=qual[use]
        if not len(v):continue
        n=v[:,0];variance=v[:,2]-v[:,1]**2/n;eligible=q&(variance>0)
        nr=np.full(len(n),np.nan);nx=nr.copy();nr[eligible]=1-v[eligible,4]/variance[eligible];nx[eligible]=1-v[eligible,6]/variance[eligible]
        delta_rmse=np.sqrt(v[:,6]/n)-np.sqrt(v[:,4]/n)
        rows.append(dict(replicate=rep,rmse_change=float(np.mean(delta_rmse)),rmse_median_paired_change=float(np.median(delta_rmse)),rmse_difference_of_medians=float(np.median(np.sqrt(v[:,6]/n))-np.median(np.sqrt(v[:,4]/n))),abs_bias_change=float(np.mean(abs(v[:,5]/n)-abs(v[:,3]/n))),nse_difference_of_medians=float(np.nanmedian(nx)-np.nanmedian(nr)) if eligible.any() else np.nan,nse_median_paired_change=float(np.nanmedian(nx-nr)) if eligible.any() else np.nan))
    return pd.DataFrame(rows)

def main(selected, comparison="main", block_months=1, scopes=None, held_stations=None):
    global REPORT
    REPORT=R/"reports"/comparison/(str(block_months)+"month");REPORT.mkdir(parents=True,exist_ok=True)
    obs_event=pd.read_parquet(R/'data/heldout_labels/events/observed_days.parquet');events=pd.read_parquet(R/'data/heldout_labels/events/events_frozen.parquet')
    pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');pub=pub[pub.model_eligible&pub.tn_mg_l.notna()].copy()
    registry=pd.read_parquet(R/'data/station_registry.parquet');cohort=registry.drop_duplicates('station_key').set_index('station_key').cohort.to_dict()
    scopes=scopes or [('F23',2023),('F24',2024)]
    if held_stations is not None:
        obs_event=obs_event[obs_event.station_key.isin(held_stations)].copy();events=events[events.station_key.isin(held_stations)].copy();pub=pub[pub.station_key.isin(held_stations)].copy()
    rng=np.random.default_rng(1729)
    if block_months==1:by_year={y:rng.integers(1,13,(1000,12)).tolist() for y in [2023,2024]}
    else:by_year={y:np.stack([np.array([m+j for m in rng.integers(1,12,6) for j in range(2)]) for _ in range(1000)]).tolist() for y in [2023,2024]}
    draws={f:by_year[y] for f,y in scopes}
    rt.write(REPORT/'bootstrap_draws.json',draws)
    pd.DataFrame([dict(fold=f,replicate=i,copy_id=j,month=m) for f,seqs in draws.items() for i,seq in enumerate(seqs) for j,m in enumerate(seq)]).to_parquet(REPORT/'bootstrap_ledger.parquet',index=False)
    stations=[];eventrows=[];centrows=[];joined_cache={};event_cache={};boot=[];curves=[]
    for fold,year in scopes:
        if not all(fold+'_'+arm in selected for arm in ['R','X']):continue
        for arm in ['R','X']:
            tag=selected[fold+'_'+arm];path=R/'outputs'/tag
            pred=pd.read_parquet(path/'daily_station_mass_water.parquet').rename(columns={'concentration_mg_l':'p'});pred=pred[pred.date.dt.year.eq(year)].copy()
            assert np.isfinite(pred.p).all() and (pred.water_m3_day>0).all()
            obs=pd.read_parquet(R/'data/heldout_labels'/f'{year}_days.parquet')
            if held_stations is not None:obs=obs[obs.station_key.isin(held_stations)].copy();pred=pred[pred.station_key.isin(held_stations)].copy()
            daily=obs.merge(pred[['station_key','date','p']],on=['station_key','date'],how='left',validate='one_to_one');assert daily.p.notna().all()
            daily['month']=daily.date.dt.month;daily['year']=year
            daily.to_parquet(path/'HF_evaluation_days.parquet',index=False)
            hf=[]
            for (s,month),g in daily.groupby(['station_key','month']):
                w=g.n/g.n.sum();e=g.p-g.y;mean=float(w@e)
                centrows.append(dict(fold=fold,arm=arm,station_key=s,month=int(month),centered_mse=float(w@((e-mean)**2))))
                hf.append(dict(station_key=s,month=month,y=float(w@g.y),p=float(w@g.p)))
            hf=pd.DataFrame(hf);monthly=pred.assign(month=pred.date.dt.month).groupby(['station_key','month']).p.mean().reset_index()
            monthly=pub[pub.year.eq(year)].merge(monthly,on=['station_key','month'],how='left',validate='one_to_one').rename(columns={'tn_mg_l':'y'});assert monthly.p.notna().all()
            monthly.to_parquet(path/'PUB_evaluation_months.parquet',index=False);hf.to_parquet(path/'HF_evaluation_months.parquet',index=False)
            for scale,frame in [('HF_day',daily),('HF_month',hf),('PUB_month',monthly)]:
                joined_cache[(fold,arm,scale)]=frame.copy()
                for s,g in frame.groupby('station_key'):
                    qual=(len(g)>=30 and g.month.nunique()>=3) if scale=='HF_day' else len(g)>=8
                    eligible=bool(qual and np.var(g.y)>0);stat=metrics(g.y,g.p)
                    if not eligible:stat['nse']=np.nan
                    stations.append(dict(fold=fold,arm=arm,scale=scale,station_key=s,cohort=cohort[s],n_months=g.month.nunique(),nse_eligible=eligible,**stat))
            ev=event_scores(pred,obs_event,events,year);ev.to_parquet(path/'event_coverage.parquet',index=False);good=ev[ev.eligible].copy()
            if held_stations is None and year==2024:assert len(good)==57 and good.station_key.nunique()==15
            if held_stations is None and year==2023:assert len(good)==40 and good.station_key.nunique()==11
            good.to_parquet(path/'event_scores.parquet',index=False);event_cache[(fold,arm)]=good
            for s,g in good.groupby('station_key'):
                absent=int(((g.obs_base>0)&(g.obs_peak>0)&~g.ratio_defined.astype(bool)).sum())
                stat={k:float(g[k].median()) for k in ['amplitude_error','peak_error','base_error']}
                if absent:stat['amplitude_error']=np.nan
                eventrows.append(dict(fold=fold,arm=arm,station_key=s,events=len(g),single_day_events=int(g.n_peak.eq(1).sum()),ratio_defined_events=int(g.ratio_defined.sum()),model_ratio_undefined_events=absent,**stat))
            for e in good.itertuples():
                curve=pred[pred.station_key.eq(e.station_key)&pred.date.between(e.background_start,e.end+pd.Timedelta(days=7))][['station_key','date','p']].copy()
                curve=curve.merge(obs_event[['station_key','date','y']],on=['station_key','date'],how='left');curve['event_rank']=e.event_rank;curve['fold']=fold;curve['arm']=arm;curves.append(curve)
    st=pd.DataFrame(stations);st.to_csv(REPORT/'station_metrics.csv',index=False)
    keys=['fold','scale','station_key','cohort'];paired=st[st.arm.eq('R')].drop(columns='arm').merge(st[st.arm.eq('X')].drop(columns='arm'),on=keys,suffixes=('_R','_X'),validate='one_to_one')
    for k in ['nse','rmse','bias','abs_bias','r','logrmse','amplitude_ratio']:paired['delta_'+k]=paired[k+'_X']-paired[k+'_R']
    paired.to_csv(REPORT/'paired_station_changes.csv',index=False)
    groups=pd.concat([paired.assign(group='ALL'),paired.assign(group=paired.cohort)],ignore_index=True);core=paired_summary(groups);core.to_csv(REPORT/'core_paired_table.csv',index=False)
    cent=pd.DataFrame(centrows);cp=cent[cent.arm.eq('R')].drop(columns='arm').merge(cent[cent.arm.eq('X')].drop(columns='arm'),on=['fold','station_key','month'],suffixes=('_R','_X'),validate='one_to_one');cp['change']=cp.centered_mse_X-cp.centered_mse_R;cp.to_csv(REPORT/'paired_centered_errors.csv',index=False)
    ep=pd.DataFrame(eventrows);ep=ep[ep.arm.eq('R')].drop(columns='arm').merge(ep[ep.arm.eq('X')].drop(columns='arm'),on=['fold','station_key'],suffixes=('_R','_X'),validate='one_to_one');ep.to_csv(REPORT/'paired_event_station_errors.csv',index=False)
    es=[];pairs=[]
    for fold in sorted(ep.fold.unique()):
        e=ep[ep.fold.eq(fold)];row=dict(fold=fold,stations=len(e),events=int(e.events_R.sum()),model_ratio_undefined_R=int(e.model_ratio_undefined_events_R.sum()),model_ratio_undefined_X=int(e.model_ratio_undefined_events_X.sum()))
        for k in ['amplitude_error','peak_error','base_error']:
            row[k+'_R']=float(e[k+'_R'].mean());row[k+'_X']=float(e[k+'_X'].mean());row[k+'_change']=float((e[k+'_X']-e[k+'_R']).mean());row[k+'_improved_fraction']=float((e[k+'_X']<e[k+'_R']).mean())
        if row['model_ratio_undefined_R'] or row['model_ratio_undefined_X']:row['amplitude_error_change']=np.nan;row['amplitude_error_improved_fraction']=np.nan
        cc=cp[cp.fold.eq(fold)].groupby('station_key')[['centered_mse_R','centered_mse_X']].mean().mean();row.update(cc.to_dict());es.append(row)
        a=event_cache[(fold,'R')];b=event_cache[(fold,'X')];p=a.merge(b,on=['station_key','event_rank','event_month'],suffixes=('_R','_X'),validate='one_to_one');p['fold']=fold
        p['baseline_peak_stratum']=np.where(p.pred_peak_R>p.obs_peak_R,'OVER','UNDER_OR_EQUAL');pairs.append(p)
        for scale in ['HF_day','HF_month','PUB_month']:
            a=joined_cache[(fold,'R',scale)];b=joined_cache[(fold,'X',scale)];keys=['station_key','date'] if scale=='HF_day' else ['station_key','month']
            z=a.merge(b[keys+['p']],on=keys,suffixes=('_R','_X'),validate='one_to_one')
            q=paired[(paired.fold==fold)&(paired.scale==scale)&paired.nse_eligible_R&paired.nse_eligible_X].station_key
            boot.append(boot_metrics(z,draws[fold],set(q)).assign(fold=fold,scale=scale))
        # Event statistics are computed on original events; repeated blocks get
        # distinct copy IDs. No new event adjacency is inferred after sampling.
        vals=[]
        bymonth={m:g for m,g in p.groupby('event_month')}
        for rep,seq in enumerate(draws[fold]):
            chunks=[bymonth[m].assign(copy_id=i) for i,m in enumerate(seq) if m in bymonth]
            if not chunks:continue
            sample=pd.concat(chunks,ignore_index=True);row=dict(fold=fold,scale='events',replicate=rep)
            for k in ['amplitude_error','peak_error','base_error']:
                agg=sample.groupby('station_key')[[k+'_R',k+'_X']].median().mean();row[k+'_change']=float(agg.iloc[1]-agg.iloc[0])
            # A bootstrap cannot recover support missing in the original model.
            # Keep all absolute errors, but do not report amplitude intervals on
            # a selected subset of events that happens to avoid undefined ratios.
            if e.model_ratio_undefined_events_R.sum() or e.model_ratio_undefined_events_X.sum():
                row['amplitude_error_change']=np.nan
            vals.append(row)
        boot.append(pd.DataFrame(vals))
        centered=cp[cp.fold.eq(fold)]
        centered_months={int(m):g for m,g in centered.groupby('month')}
        centered_rows=[]
        for rep,seq in enumerate(draws[fold]):
            pieces=[centered_months[m].assign(copy_id=i) for i,m in enumerate(seq) if m in centered_months]
            if not pieces:continue
            sampled=pd.concat(pieces,ignore_index=True).groupby('station_key')[['centered_mse_R','centered_mse_X']].mean()
            centered_rows.append(dict(fold=fold,scale='HF_centered',replicate=rep,centered_mse_change=float((sampled.centered_mse_X-sampled.centered_mse_R).mean()),centered_rmse_median_paired_change=float((np.sqrt(sampled.centered_mse_X)-np.sqrt(sampled.centered_mse_R)).median())))
        boot.append(pd.DataFrame(centered_rows))
    pd.DataFrame(es).to_csv(REPORT/'event_centered_summary.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_parquet(REPORT/'paired_events.parquet',index=False);pd.concat(curves,ignore_index=True).to_parquet(REPORT/'event_curves.parquet',index=False)
    bootframe=pd.concat(boot,ignore_index=True);bootframe.to_csv(REPORT/'bootstrap_changes.csv',index=False)
    intervals=[]
    for (fold,scale),g in bootframe.groupby(['fold','scale']):
        for k in [x for x in g.columns if x not in ['fold','scale','replicate']]:
            v=g[k].dropna()
            if len(v):
                lo,hi=np.quantile(v,[.025,.975]);intervals.append(dict(fold=fold,scale=scale,metric=k,replicates=len(v),median=float(np.median(v)),lower=float(lo),upper=float(hi),crosses_zero=bool(lo<=0<=hi)))
    pd.DataFrame(intervals).to_csv(REPORT/'bootstrap_intervals.csv',index=False)
    return dict(core=core.to_dict('records'),event_summary=es)
