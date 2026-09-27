"""Post-freeze real TN descriptions, not calibrated/validated new forecasts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd,torch
from d29_platform.legacy import build_legacy,SNAP
from d29_platform.metrics import CoveragePolicy,evaluate_pair,monthly_views,evaluate_events,paired_calendar_bootstrap
OUT=ROOT/'outputs/supply_limited_reference/evaluation';OUT.mkdir(parents=True,exist_ok=True)
RUN=OUT.parent
def main():
    receipt=json.loads((RUN/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='completed_conditional_forward':raise ValueError('PREDICTIONS_NOT_FROZEN')
    candidate=pd.read_parquet(RUN/'station_days.parquet');fingerprint=sha(RUN/'station_days.parquet')
    if candidate.duplicated(['station_key','date']).any():raise ValueError('DUPLICATE_MODEL_STATION_DAY')
    m,x,_=build_legacy('F23','U',inference_only=True)
    meta=pd.read_parquet(SNAP/'data/prediction_calendar.parquet');meta=meta[meta.year.between(2021,2024)].reset_index(drop=True)
    m.registry=json.loads((SNAP/'data/prediction_registry.json').read_text(encoding='utf-8'))['records'];m._daily_meta_cache={}
    with torch.no_grad():b=m.daily_boundary(torch.tensor(x),meta)
    baseline=pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[b['record'].numpy()],date=m.data.dates[b['day_index'].numpy()],baseline=1000*b['mass'].numpy()/b['water'].numpy()))
    baseline.to_parquet(OUT/'U_F23_reference_station_days.parquet',index=False)
    model=candidate.merge(baseline,on=['station_key','date'],validate='one_to_one')
    files=[SNAP/'data/cohorts/F23_G/hf_days.parquet',SNAP/'data/heldout_labels/2023_days.parquet',SNAP/'data/heldout_labels/2024_days.parquet']
    labels=[]
    for f in files:
        t=pd.read_parquet(f)
        if 'eligible_day' in t:t=t[t.eligible_day]
        labels.append(t[['station_key','date','y','n']])
    labels=pd.concat(labels,ignore_index=True)
    if labels.duplicated(['station_key','date']).any():raise ValueError('LABEL_SUPPORT_OVERLAP')
    pair=labels.merge(model.drop(columns=['read_count']),on=['station_key','date'],validate='one_to_one').rename(columns={'y':'truth','n':'read_count'})
    pair.to_parquet(OUT/'paired_HF_days.parquet',index=False)
    summaries=[];pol=CoveragePolicy(120,6)
    for year in (2021,2022,2023,2024,'2021_2024'):
        sub=pair if isinstance(year,str) else pair[pair.date.dt.year.eq(year)]
        station,summary=evaluate_pair(sub,policy=pol,scope='conditional_HF_daily')
        station.to_csv(OUT/f'{year}_daily_station.csv',index=False);summary['period']=year;summaries.append(summary)
        summary.to_csv(OUT/f'{year}_daily_summary.csv',index=False)
        print(year,summary.to_json(orient='records'),flush=True)
    pd.concat(summaries).to_csv(OUT/'daily_all_periods_summary.csv',index=False)
    # Original monthly labels are separately identified and retain their uncertain sampling support.
    old=pd.read_parquet(SNAP/'data/heldout_labels/monthly_original.parquet');old=old[old.model_eligible&~old.confirmed_invalid].copy()
    if old.duplicated(['station_key','year','month']).any():raise ValueError('NONUNIQUE_ORIGINAL_MONTH')
    model['year']=model.date.dt.year;model['month']=model.date.dt.month
    monthly=model.groupby(['station_key','year','month']).apply(lambda g:pd.Series(dict(baseline=np.average(g.baseline,weights=g.read_count),candidate=np.average(g.candidate,weights=g.read_count))),include_groups=False).reset_index()
    om=old[['station_key','year','month','tn_mg_l']].merge(monthly,on=['station_key','year','month'],validate='one_to_one').rename(columns={'tn_mg_l':'truth'})
    om['date']=pd.to_datetime(dict(year=om.year,month=om.month,day=1));om['read_count']=1.
    for name,table in monthly_views(pair,om).items():
        table.to_parquet(OUT/f'{name}_paired.parquet',index=False)
        for year in (2023,2024):
            s,summary=evaluate_pair(table[table.date.dt.year.eq(year)],policy=CoveragePolicy(6,6),scope=name)
            s.to_csv(OUT/f'{year}_{name}_station.csv',index=False);summary.to_csv(OUT/f'{year}_{name}_summary.csv',index=False)
    events=pd.read_parquet('E:/SPARROW/5_Test/20260920_4/data/evaluation/events_frozen.parquet').rename(columns={'event_rank':'event_id'})
    events['background_end']=pd.to_datetime(events.start)-pd.Timedelta(days=1)
    eventrows=[]
    for year in (2023,2024):
        ee=events[(pd.to_datetime(events.start).dt.year==year)&(pd.to_datetime(events.end).dt.year==year)&(pd.to_datetime(events.background_start)>=pd.Timestamp(year,1,1))]
        ev=evaluate_events(pair[pair.date.dt.year==year],ee,evaluation_start=f'{year}-01-01');ev['year']=year;eventrows.append(ev)
    pd.concat(eventrows).to_csv(OUT/'frozen_events.csv',index=False)
    for year in (2023,2024):
        t=pair[pair.date.dt.year==year]
        for length in (1,2):
            print('bootstrap',year,length,flush=True)
            boot=paired_calendar_bootstrap(t,policy=pol,n_bootstrap=1000,block_months=length,seed=1729)
            for name in ('replicates','intervals','ledger'):boot[name].to_csv(OUT/f'{year}_bootstrap_{length}month_{name}.csv',index=False)
    write_json(OUT/'receipt.json',dict(status='completed_conditional_description',candidate_sha256=fingerprint,
        prediction_unchanged=sha(RUN/'station_days.parquet')==fingerprint,label_files={str(p):sha(p) for p in files},
        baseline='Frozen F23 U at all dates; 2024 table is NOT F24 refit',candidate='uncalibrated S1 LAND1 supply-limited research run',
        compared_input_history_and_equations_differ=True,single_factor_causal_gain_allowed=False,strict_mass_pass=receipt['strict_mass_pass'],
        fits=0,bootstrap_interpretation='frozen-output support stability only; no predictive certification',
        daily_weight='read counts',centered_weight='demean by read counts within station-month; equal months',
        original_monthly_operator='frozen model calendar day weighting; actual original sampling time unverified'))
if __name__=='__main__':main()
