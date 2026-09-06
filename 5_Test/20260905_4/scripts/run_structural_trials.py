"""Registered single additions; combinations require independent evidence."""
from experiment_io import ROOT,save_spec,fit_starts,temporal_spec
import sys
sys.path.insert(0,str(ROOT/'5_Test/20260905_3/scripts'))
from summarize_development import station_metrics,summary,paired_bootstrap
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
from pathlib import Path
import json
import pandas as pd
import os


BASE='H7_CONTACT_LIFETIME__STATION_NORMALIZED_MSE'
# Chosen from the physical hypothesis and user objective before inspection of
# the complete factorial ranking; no data-dependent base capacity selection.
BASE_CONFIG=dict(model='H7_CONTACT_LIFETIME',loss='STATION_NORMALIZED_MSE',prior_scale=1.,calendar='CENTRAL',timing='monthly_pulse',dynamic=False)


def candidates():
    variations=[('uniform_daily',dict(timing='uniform_daily')),('calendar_early',dict(calendar='EARLY')),
                ('calendar_late',dict(calendar='LATE')),('contact_dynamic',dict(dynamic=True))]
    return [dict(BASE_CONFIG,**{'candidate_id':name,**change}) for name,change in variations]


def main():
    stage='20260905_4';run=ROOT/'5_Test'/stage
    development=json.loads((ROOT/'5_Test/20260905_3/reports/development_summary.json').read_text(encoding='utf-8'))
    if development['status']!='DEVELOPMENT_OOF_COMPLETE_REQUIRES_SPATIAL_AND_CONFIRMATION':raise RuntimeError('Development matrix incomplete')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    statuspath=run/'reports/structural_queue.json';status=dict(status='RUNNING',pid=os.getpid(),runtime=RUNTIME,started_utc=utc_now(),completed=[])
    config=candidates()
    registry=dict(base=BASE,base_capacity_chosen_without_full_development_ranking=True,candidates=config,
        single_addition_gates='positive paired station and tree median-NSE CI; no q25/log-RMSE regression',
        combinations='at most one timing/calendar alteration plus two-coefficient contact dynamics; both must pass alone',
        temperature='requires separate conditional residual evidence')
    registry_path=run/'reports/structural_registry.json'
    if registry_path.exists() and json.loads(registry_path.read_text(encoding='utf-8'))!=registry:raise RuntimeError('Structural registry changed')
    atomic_json(registry,registry_path)
    baseframe=pd.read_parquet(ROOT/'5_Test/20260905_3/outputs'/f'{BASE.lower()}_oof_predictions.parquet')
    bm=station_metrics(baseframe);bs=summary(bm,baseframe);outcomes={}
    def execute(candidate):
        frames=[]
        for year in range(2020,2024):
            spec=temporal_spec(candidate,year,obs,stage)
            frame,_=fit_starts(save_spec(spec,stage),statuspath,status)
            frame=frame.loc[frame.primary_gate].copy();frame['fold']=f'T{year}';frames.append(frame)
        frame=pd.concat(frames,ignore_index=True).sort_values('observation_id')
        if set(frame.observation_id)!=set(baseframe.observation_id):raise ValueError('Different structural comparison cohort')
        sm=station_metrics(frame);s=summary(sm,frame);paired=paired_bootstrap(sm,bm)
        gates=dict(station_ci_positive=paired['station_ci95'][0]>0,tree_ci_positive=paired['tree_cluster_ci95'][0]>0,
            q25_noninferior=s['q25_nse']>=bs['q25_nse']-.05,
            log_rmse_noninferior=s['mean_station_log_rmse']<=1.05*bs['mean_station_log_rmse'])
        outcome=dict(candidate=candidate,summary=s,paired_vs_base=paired,gates=gates,addition_supported=all(gates.values()))
        key=candidate['candidate_id'];outcomes[key]=outcome
        atomic_parquet(frame,run/'outputs'/f'{key}_oof_predictions.parquet')
        atomic_parquet(sm,run/'outputs'/f'{key}_oof_station_metrics.parquet')
        atomic_json(outcome,run/'reports'/f'{key}_oof_summary.json')
        status['completed'].append(key);atomic_json(status,statuspath)
        print('STRUCTURE_OOF_FINISHED',key,s['median_nse'],outcome['addition_supported'],flush=True)
    for candidate in config:execute(candidate)
    timing=[o for k,o in outcomes.items() if k!='contact_dynamic' and o['addition_supported']]
    if timing and outcomes['contact_dynamic']['addition_supported']:
        best=max(timing,key=lambda o:(o['summary']['median_nse'],o['candidate']['candidate_id']))
        combo=dict(best['candidate']);combo.update(candidate_id='timing_plus_dynamic',dynamic=True)
        execute(combo)
    status.update(status='STRUCTURAL_OOF_COMPLETE',active=None,updated_utc=utc_now());atomic_json(status,statuspath)
    atomic_json(dict(status=status['status'],runtime=RUNTIME,base=BASE,base_summary=bs,outcomes=outcomes,
        promotion=False,remaining=['temperature_residual_decision','nested_spatial','temporal_confirmation','prior_identifiability','final_exports']),run/'reports/structural_summary.json')


if __name__=='__main__':main()
