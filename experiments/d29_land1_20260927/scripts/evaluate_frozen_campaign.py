"""Evaluate an explicit completed campaign manifest after every prediction freezes.

The manifest lists `selected_jobs`, `excluded_jobs` with reasons, and all 32
registered path ids in `accounted_paths`. Selection is generated from training
objectives only. This script refuses partial/live campaign evaluation.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json as _write_json,sha
configure()
import pandas as pd
import numpy as np
from d29_training.evaluation import join_pair,hf_monthly,evaluate_events,pair_tables
from d29_training.bootstrap import paired_resamples
from d29_training.metrics import station_table,eligible_nse_stations
from d29_training.campaign import select_campaign,year_role


def write_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):v=v.item()
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    _write_json(path,clean(value))


def run(manifest_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    registered=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
    jobs={j['id']:j for j in registered}
    if set(manifest['accounted_paths'])!=set(jobs):raise RuntimeError('INCOMPLETE_CAMPAIGN_ACCOUNTING')
    if not manifest.get('selection_training_only'):raise RuntimeError('SELECTION_RULE_NOT_CERTIFIED')
    selection=select_campaign(registered,manifest['training_records'],manifest['excluded_jobs'])
    if selection['selected_jobs']!=manifest['selected_jobs']:raise RuntimeError('TRAINING_SELECTION_MISMATCH')
    event_contract_path=ROOT/'data/event_support_contract.json'
    if sha(event_contract_path)!=manifest['event_support_contract_sha256']:raise RuntimeError('EVENT_SUPPORT_CONTRACT_CHANGED')
    event_contract=json.loads(event_contract_path.read_text(encoding='utf-8'))
    if any(sha(p)!=h for p,h in event_contract['files'].items()):raise RuntimeError('FROZEN_EVENT_SUPPORT_CHANGED')
    predictions={};freezes={}
    for job_id in manifest['selected_jobs']:
        folder=ROOT/'outputs/jobs'/job_id
        freeze=json.loads((folder/'prediction_freeze.json').read_text(encoding='utf-8'))
        for file,key in [('frozen_station_days.parquet','days_sha256'),('frozen_station_months.parquet','months_sha256'),('frozen_parameters.npy','parameter_sha256')]:
            if sha(folder/file)!=freeze[key]:raise RuntimeError('FROZEN_PREDICTION_CHANGED')
        predictions[job_id]=(pd.read_parquet(folder/'frozen_station_days.parquet'),pd.read_parquet(folder/'frozen_station_months.parquet'))
        freezes[job_id]=sha(folder/'prediction_freeze.json')
    # Evaluation labels are first loaded only after the full manifest gate.
    monthly=pd.read_parquet(ROOT/'data/monthly_tn.parquet')
    daily=pd.read_parquet(ROOT/'data/hf_daily.parquet')
    events_path=ROOT/'data/frozen_evaluation_events.parquet'
    events=pd.read_parquet(events_path)
    blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text(encoding='utf-8'))
    out=ROOT/'outputs/evaluation';out.mkdir(exist_ok=True)
    if (out/'evaluation_started.json').exists():raise RuntimeError('EVALUATION_ALREADY_STARTED_USE_VERSIONED_RERUN')
    write_json(out/'evaluation_started.json',{'manifest_sha256':sha(manifest_path),'prediction_freezes':freezes,'design_frozen_before_readout':True,'retrospective_not_new_blind_test':True})
    summaries=[]
    absolute=out/'absolute_results';absolute.mkdir(exist_ok=True)
    for job_id,(d,m) in predictions.items():
        j=jobs[job_id]
        held=set(blocks[str(j['space_block'])]['held_stations']) if j['space_block'] is not None else None
        for scope,observed,pred,monthly_flag in [('daily',daily,d,False),('monthly_report',monthly,m,True)]:
            frame=join_pair(observed,pred,pred,monthly=monthly_flag)
            for year,g in frame.groupby(frame.date.dt.year):
                for support,sub in [('all_descriptive',g)]+([('heldout',g[g.station_key.isin(held)])] if held is not None else []):
                    if sub.empty:continue
                    t=station_table(sub,predictions=('baseline',),daily=not monthly_flag,minimum_coverage=True)
                    t['configuration']=job_id;t['year']=int(year);t['support']=support
                    t['role']=year_role(j,int(year),support=='heldout')
                    t.to_csv(absolute/f'{job_id}_{scope}_{year}_{support}.csv',index=False)
    for job_id in manifest['selected_jobs']:
        j=jobs[job_id]
        if j['strategy']=='T0':continue
        base=[k for k in manifest['selected_jobs'] if jobs[k]['model']==j['model'] and jobs[k]['fold']==j['fold'] and jobs[k]['space_block']==j['space_block'] and jobs[k]['strategy']=='T0']
        if len(base)!=1:raise RuntimeError('UNIQUE_BASELINE_REQUIRED')
        base=base[0];folder=out/job_id;folder.mkdir(exist_ok=True)
        bd,bm=predictions[base];cd,cm=predictions[job_id]
        df=join_pair(daily,bd,cd);mf=join_pair(monthly,bm,cm,monthly=True)
        ident=json.loads((ROOT/'data/training_contracts'/job_id/'identity.json').read_text(encoding='utf-8'))
        long=set(ident['long_stations'])
        spatial=j['space_block'] is not None
        held=set(blocks[str(j['space_block'])]['held_stations']) if spatial else set()
        buffer=set(blocks[str(j['space_block'])]['buffer_stations']) if spatial else set()
        for scope,frame in [('daily',df),('monthly_report',mf),('HF_monthly',hf_monthly(df))]:
            if frame.empty:continue
            frame=frame.copy();frame['year']=frame.date.dt.year
            frame['record_group']=frame.station_key.map(lambda s:'long' if s in long else 'other')
            frame['spatial_role']=frame.station_key.map(lambda s:'held' if s in held else 'buffer' if s in buffer else 'training_region')
            frame.to_parquet(folder/f'{scope}_common_support.parquet',index=False)
            for year,g in frame.groupby('year'):
                for label,sub in [('all',g),('long',g[g.record_group.eq('long')]),('other',g[g.record_group.eq('other')])]:
                    if sub.empty:continue
                    table,summary=pair_tables(sub,scope=='daily')
                    table.to_csv(folder/f'{scope}_{year}_{label}_stations.csv',index=False)
                    summaries.append(dict(job=job_id,baseline=base,scope=scope,year=int(year),group=label,role=year_role(j,int(year)),**summary))
                if int(year) not in j['evaluate']:continue
                sub=g[g.spatial_role.eq('held')] if spatial else g
                if sub.empty:
                    summaries.append(dict(job=job_id,scope=scope,year=int(year),role='heldout',status='no_common_support'));continue
                table,summary=pair_tables(sub,scope=='daily')
                table.to_csv(folder/f'{scope}_{year}_heldout_stations.csv',index=False)
                role='same_year_spatial' if spatial and year<=j['train_end'] else 'space_time' if spatial else 'time'
                summaries.append(dict(job=job_id,baseline=base,scope=scope,year=int(year),role=role,**summary))
                for block in [1,2]:
                    for center in ([False,True] if scope=='daily' else [False]):
                        draws,receipt=paired_resamples(sub,1000,block,1729,center,eligible_sites=eligible_nse_stations(sub,scope=='daily'))
                        receipt['coverage_policy']='fixed original common support: daily >=30 days across >=3 months; monthly >=8 values; not reselected per bootstrap draw'
                        name=f'{scope}_{year}_{"centered" if center else "raw"}_block{block}'
                        draws.to_csv(folder/f'{name}_bootstrap.csv',index=False)
                        receipt['intervals']={c:draws[c].quantile([.025,.5,.975]).to_dict() for c in ['median_difference','median_paired_difference','improved_fraction']}
                        receipt['undefined_replicates']=int(draws.common_stations.eq(0).sum())
                        write_json(folder/f'{name}_bootstrap_receipt.json',receipt)
        for year in j['evaluate']:
            sub=df[df.station_key.isin(held)] if spatial else df
            eligible_events=events[events.station_key.isin(held)] if spatial else events
            evaluate_events(sub,eligible_events,year).to_csv(folder/f'events_{year}.csv',index=False)
            registered_year=pd.to_datetime(events['start']).dt.year.eq(year)
            eligible_year=pd.to_datetime(eligible_events['start']).dt.year.eq(year)
            write_json(folder/f'events_{year}_support.json',dict(year=year,
                original_candidate_events=event_contract.get('candidate_counts',{}).get(str(year)),
                registered_events=int(registered_year.sum()),events_in_requested_spatial_support=int(eligible_year.sum()),
                frozen_event_file_sha256=sha(events_path),
                support='heldout_stations_only' if spatial else 'all_certified_event_stations',
                background_clipped_to_evaluation_year=True))
    write_json(out/'paired_summaries.json',summaries)
    write_json(out/'evaluation_completed.json',{'manifest_sha256':sha(manifest_path),'selected_jobs':manifest['selected_jobs'],'paired_summary_sha256':sha(out/'paired_summaries.json'),'NSE_is_not_alone_evidence_of_correct_dynamics':True})

if __name__=='__main__':run(Path(sys.argv[1]))
