"""Post-freeze evaluation helpers. Never imported by any training worker."""
import numpy as np
import pandas as pd
from .metrics import station_table,paired_summary,basic


def join_pair(observed,baseline,candidate,monthly=False):
    keys=['station_key','year','month'] if monthly else ['station_key','date']
    o=observed.copy()
    if 'eligible' in o:o=o[o.eligible]
    o=o.rename(columns={'tn_mg_l':'observed'})
    for name,p in [('baseline',baseline),('candidate',candidate)]:
        if p.duplicated(keys).any():raise ValueError('DUPLICATE_PREDICTION_SUPPORT')
        o=o.merge(p[keys+['prediction_mg_l']].rename(columns={'prediction_mg_l':name}),on=keys,how='inner',validate='one_to_one')
    o=o[np.isfinite(o[['observed','baseline','candidate']]).all(axis=1)].copy()
    return o


def hf_monthly(frame):
    f=frame.copy();f['year']=f.date.dt.year;f['month']=f.date.dt.month
    records=[]
    for (s,y,m),g in f.groupby(['station_key','year','month']):
        w=g.read_count.to_numpy(float);w=w/w.sum()
        r={'station_key':s,'year':y,'month':m,'date':pd.Timestamp(y,m,1),'eligible':True,'daily_support':len(g)}
        for c in ['observed','baseline','candidate']:r[c]=float(w@g[c].to_numpy(float))
        records.append(r)
    return pd.DataFrame(records)


def evaluate_events(frame,events,year):
    """Common measured days only; background may not cross the heldout year."""
    records=[]
    for e in events.to_dict('records'):
        start=pd.Timestamp(e['start']);end=pd.Timestamp(e['end'])
        if start.year!=year:continue
        bg=max(pd.Timestamp(e['background_start']),pd.Timestamp(year,1,1))
        g=frame[frame.station_key.eq(e['station_key'])]
        active=g[g.date.between(start,end)].sort_values('date')
        background=g[g.date.ge(bg)&g.date.lt(start)]
        if len(active)<1 or len(background)<4:
            records.append({'station_key':e['station_key'],'event_rank':e['event_rank'],'year':year,'status':'insufficient_common_event_or_background_days','NSE':np.nan});continue
        for name in ['baseline','candidate']:
            y=active.observed.to_numpy(float);p=active[name].to_numpy(float)
            oy=int(np.argmax(y));op=int(np.argmax(p))
            ybg=float(background.observed.median());pbg=float(background[name].median())
            records.append({'station_key':e['station_key'],'event_rank':e['event_rank'],'year':year,'configuration':name,'status':'common_observed_dates','event_start':start,'event_end':end,'background_start_used':bg,'background_days':len(background),'observed_peak_day':active.date.iloc[oy],'predicted_peak_day':active.date.iloc[op],'peak_day_offset':int((active.date.iloc[op]-active.date.iloc[oy]).days),'observed_amplitude':float(y.max()-ybg),'predicted_amplitude':float(p.max()-pbg),'amplitude_error':float(p.max()-pbg-y.max()+ybg),'peak_error':float(p.max()-y.max()),'background_error':pbg-ybg,**basic(y,p)})
            defined=min(ybg,pbg,float(y.max()),float(p.max()))>0
            records[-1]['log_amplitude_ratio_defined']=defined
            records[-1]['log_amplitude_error']=float(abs(np.log(p.max()/pbg)-np.log(y.max()/ybg))) if defined else np.nan
    return pd.DataFrame(records)


def pair_tables(frame,daily):
    if frame.empty:return pd.DataFrame(),{'status':'no_common_support','common_stations':0}
    t=station_table(frame,daily=daily,minimum_coverage=True)
    s={'NSE':paired_summary(t),'common_rows':len(frame),'common_available_stations':frame.station_key.nunique()}
    if daily:s['month_centered_NSE']=paired_summary(t,'month_centered_NSE')
    return t,s
