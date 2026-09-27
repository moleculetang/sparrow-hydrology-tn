"""Frozen-result diagnostics only: NH4, DO and unresolved subdaily TN scale.

No NH4/DO enters model inputs, gradients, grouping or checkpoint selection.
Strata thresholds are estimated from each path's admitted training months.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure
configure()
import numpy as np
import pandas as pd
from d29_training.evaluation import pair_tables
from scripts.evaluate_frozen_campaign import write_json


def main():
    out=ROOT/'outputs/evaluation'
    if not (out/'evaluation_completed.json').exists():raise RuntimeError('WAIT_FOR_FROZEN_CAMPAIGN_EVALUATION')
    raw=pd.read_parquet(ROOT/'data/monthly_auxiliary_archive.parquet')
    daily=pd.read_parquet(ROOT/'data/hf_daily.parquet')
    readings=pd.read_parquet(ROOT/'data/hf_readings.parquet')
    ammonia=readings[readings.eligible & readings.indicator.eq('NH3_N')].groupby(['station_key','date']).adopted_value.mean().rename('ammonia_N_mg_l').reset_index()
    jobs={j['id']:j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))}
    keys=['station_key','year','month'];aux=[];conflicts=[]
    for name,column in [('NH4','氨氮'),('DO','溶解氧')]:
        f=raw[keys+[column,'valid_date']].copy();f['value']=pd.to_numeric(f[column],errors='coerce')
        f=f[f.valid_date & np.isfinite(f.value) & f.value.ge(0)]
        for key,g in f.groupby(keys):
            if g.value.nunique()>1:conflicts.append(dict(zip(keys,key),indicator=name,reason='conflicting_monthly_auxiliary_values'));continue
            aux.append(dict(zip(keys,key),indicator=name,value=float(g.value.iloc[0])))
    aux=pd.DataFrame(aux);pd.DataFrame(conflicts).to_csv(out/'auxiliary_conflicts_excluded.csv',index=False)
    for folder in out.iterdir():
        if not folder.is_dir() or folder.name not in jobs:continue
        j=jobs[folder.name]
        f=pd.read_parquet(folder/'monthly_report_common_support.parquet')
        ident=json.loads((ROOT/'data/training_contracts'/j['id']/'identity.json').read_text(encoding='utf-8'))
        excluded=set(ident['excluded_stations']);thresholds={};records=[];paired=[]
        for indicator,a in aux.groupby('indicator'):
            train=a[a.year.le(j['train_end'])&a.year.ge(2021 if j['strategy']=='T0' else 2016)&~a.station_key.isin(excluded)]
            if train.empty:thresholds[indicator]={'status':'no_training_support'};continue
            q=train.value.quantile([1/3,2/3]).to_numpy();thresholds[indicator]={'lower_tertile':float(q[0]),'upper_tertile':float(q[1]),'training_months':len(train)}
            merged=f.merge(a[keys+['value']],on=keys,validate='one_to_one')
            merged['auxiliary_stratum']=np.where(merged.value<q[0],'low',np.where(merged.value>q[1],'high','middle'))
            for (year,stratum),g in merged.groupby(['year','auxiliary_stratum']):
                table,summary=pair_tables(g,False)
                table.to_csv(folder/f'{indicator}_{year}_{stratum}_TN_metrics.csv',index=False)
                paired.append({'indicator':indicator,'year':int(year),'stratum':stratum,'scope':'monthly_TN',**summary})
                # Metrics remain TN scores conditional on observed chemistry,
                # never NH4/DO prediction scores or independent validation.
                records.append({'indicator':indicator,'year':int(year),'stratum':stratum,'stations':int(g.station_key.nunique()),'months':len(g),'role':'descriptive TN residual stratification; training and heldout roles retained in main results'})
        write_json(folder/'auxiliary_diagnostic_identity.json',{'thresholds':thresholds,'strata':records,'not_a_training_covariate':True})
        h=pd.read_parquet(folder/'daily_common_support.parquet')
        merged=h.merge(ammonia,on=['station_key','date'],validate='one_to_one')
        atrain=ammonia[ammonia.date.dt.year.le(j['train_end'])&~ammonia.station_key.isin(excluded)]
        if len(atrain):
            q=atrain.ammonia_N_mg_l.quantile([1/3,2/3]).to_numpy()
            merged['stratum']=np.where(merged.ammonia_N_mg_l<q[0],'low',np.where(merged.ammonia_N_mg_l>q[1],'high','middle'))
            for (year,stratum),g in merged.groupby([merged.date.dt.year,'stratum']):
                table,summary=pair_tables(g,True)
                paired.append({'indicator':'HF_ammonia_N','year':int(year),'stratum':stratum,'scope':'daily_TN',**summary})
                table.to_csv(folder/f'HF_ammonia_N_{year}_{stratum}_TN_metrics.csv',index=False)
            write_json(folder/'HF_ammonia_diagnostic_identity.json',{'raw_indicator':'NH3_N','meaning':'ammonia nitrogen; not a separately modeled NH4 species','training_thresholds':q.tolist(),'not_used_as_predictor':True,'concurrent_chemistry_stratification_only':True,'HF_DO':'not available in canonical archive'})
        h['season']=((h.date.dt.month%12)//3).map({0:'DJF',1:'MAM',2:'JJA',3:'SON'})
        for (year,season),g in h.groupby([h.date.dt.year,'season']):
            table,summary=pair_tables(g,True)
            paired.append({'indicator':'season','year':int(year),'stratum':season,'scope':'daily_TN',**summary})
            table.to_csv(folder/f'daily_{year}_{season}_TN_metrics.csv',index=False)
        h=h.merge(daily[['station_key','date','within_day_std','within_day_min','within_day_max']],on=['station_key','date'],suffixes=('','_raw'),validate='one_to_one')
        rows=[]
        for (s,y),g in h.groupby(['station_key',h.date.dt.year]):
            n=g.read_count.to_numpy(float);sd=g.within_day_std.to_numpy(float)
            valid=(n>=2)&np.isfinite(sd)
            rows.append({'station_key':s,'year':int(y),'days_with_multiple_reads':int(valid.sum()),'mean_intraday_range':float((g.within_day_max-g.within_day_min).mean()),'within_day_variance_sum':float(np.sum((n[valid]-1)*sd[valid]**2)),'NSE':'not applicable: unresolved observation variability, no subdaily model prediction','definition':'sum (n_day-1)*sample_variance_day; not a claim of achievable TN accuracy'})
        pd.DataFrame(rows).to_csv(folder/'unresolved_four_hour_variability.csv',index=False)
        write_json(folder/'auxiliary_paired_summaries.json',{'comparisons':paired,'interpretation':'descriptive common supports only; formal spatial and temporal scores are in the campaign main tables'})

if __name__=='__main__':main()
