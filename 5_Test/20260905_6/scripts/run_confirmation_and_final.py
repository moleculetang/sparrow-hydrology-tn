"""Freeze a development choice before confirmation, sensitivity and final fits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_4/scripts'))
sys.path.insert(0,str(ROOT/'5_Test/20260905_3/scripts'))
from experiment_io import save_spec,fit_starts,temporal_spec
from summarize_development import station_metrics,summary,paired_bootstrap
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
import pandas as pd
import json
import subprocess
import os


def choose_development():
    run3=ROOT/'5_Test/20260905_3';run4=ROOT/'5_Test/20260905_4'
    dev=json.loads((run3/'reports/development_summary.json').read_text(encoding='utf-8'))
    structural=json.loads((run4/'reports/structural_summary.json').read_text(encoding='utf-8'))
    if not dev['status'].startswith('DEVELOPMENT_OOF_COMPLETE') or structural['status']!='STRUCTURAL_OOF_COMPLETE':raise RuntimeError('Incomplete development')
    options=[]
    for key,s in dev['summaries'].items():
        model,loss=key.split('__')
        if model=='CONTROL_H7':continue
        config=dict(candidate_id=key.lower(),model=model,loss=loss,calendar='CENTRAL',timing='monthly_pulse',dynamic=False,prior_scale=1.)
        options.append(dict(candidate=config,summary=s,prediction_path=str(run3/'outputs'/f'{key.lower()}_oof_predictions.parquet')))
    for key,result in structural['outcomes'].items():
        options.append(dict(candidate=result['candidate'],summary=result['summary'],
            prediction_path=str(run4/'outputs'/f'{key}_oof_predictions.parquet'),addition_supported=result['addition_supported']))
    best=sorted(options,key=lambda o:(-o['summary']['median_nse'],o['candidate']['candidate_id']))[0]
    return dict(status='DEVELOPMENT_CHOICE_FROZEN_BEFORE_CONFIRMATION',chosen=best,options=options,
        rule='maximum development OOF median station NSE; physical starts selected only by training objective',
        confirmation_used_for_selection=False,sensitivity_2025_used_for_selection=False,
        inputs={str(p):sha256(p) for p in [run3/'reports/development_summary.json',run4/'reports/structural_summary.json']})


def main():
    stage='20260905_6';run=ROOT/'5_Test'/stage
    choice=choose_development();choicepath=run/'reports/development_choice.json'
    if choicepath.exists() and json.loads(choicepath.read_text(encoding='utf-8'))!=choice:raise RuntimeError('Frozen development choice changed')
    atomic_json(choice,choicepath);candidate=choice['chosen']['candidate']
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    statuspath=run/'reports/final_queue.json';status=dict(status='RUNNING',pid=os.getpid(),runtime=RUNTIME,started_utc=utc_now(),completed=[])
    control=dict(candidate_id='refit_control_h7',model='CONTROL_H7',loss='STUDENT_T4_LOG1P',calendar='CENTRAL',timing='monthly_pulse',dynamic=False,prior_scale=1.)
    temporal={}
    for year in [2024,2025]:
        frames={};full_frames={}
        for config,label in [(candidate,'new'),(control,'control')]:
            fit=temporal_spec(config,year,obs,stage)
            pred,selected=fit_starts(save_spec(fit,stage),statuspath,status)
            full_frames[label]=pred.copy()
            frames[label]=pred.loc[pred.primary_gate].copy()
        nm,om=station_metrics(frames['new']),station_metrics(frames['control'])
        temporal[str(year)]=dict(new=summary(nm,frames['new']),control=summary(om,frames['control']),paired=paired_bootstrap(nm,om),
            use='retrospective confirmation' if year==2024 else 'sensitivity only; incomplete TN December and forcing/source extension')
        domains={}
        for name,column,value in [('reach17_position_unresolved','reach_id',17),('tree163_open_lake','terminal_tree_id',163)]:
            domain={}
            expected_ids=set(obs.loc[obs.year.eq(year)&obs[column].eq(value),'observation_id'])
            for label,full in full_frames.items():
                subset=full.loc[full[column].eq(value)]
                if set(subset.observation_id)!=expected_ids or subset.primary_gate.any():
                    raise AssertionError('Excluded-domain cohort mismatch')
                domain[label]=summary(station_metrics(subset),subset) if len(subset) else None
            domain.update(observation_ids=sorted(expected_ids),rows=len(expected_ids),
                role='Prespecified excluded-domain diagnostic; not used in fitting, candidate selection or primary promotion gates')
            domains[name]=domain
        temporal[str(year)]['excluded_domains']=domains
        atomic_json(temporal,run/'reports/temporal_confirmation.json')
    # Prior sensitivity is diagnostic, never a second selection using 2024.
    sensitivities=[]
    for scale in [.5,2.]:
        for year in [2023,2024]:
            config=dict(candidate);config.update(candidate_id=candidate['candidate_id']+f'_prior{scale:g}',prior_scale=scale)
            fit=temporal_spec(config,year,obs,'20260905_5')
            pred,selected=fit_starts(save_spec(fit,'20260905_5'),statuspath,status)
            frame=pred.loc[pred.primary_gate]
            sensitivities.append(dict(prior_scale=scale,year=year,selected=selected,summary=summary(station_metrics(frame),frame)))
    atomic_json(dict(status='PRIOR_SENSITIVITY_FITS_COMPLETE',results=sensitivities,not_used_to_retune_choice=True),ROOT/'5_Test/20260905_5/reports/prior_sensitivity.json')
    for end,label in [(2024,'F24'),(2025,'F25')]:
        train=obs.loc[obs.primary_gate&obs.year.between(2016,end)]
        fit=dict(candidate,id=candidate['candidate_id']+'_'+label.lower(),train_ids=train.observation_id.tolist(),
            eval_ids=obs.loc[obs.year.between(2016,end),'observation_id'].tolist(),product='formal' if end==2024 else 'sensitivity',
            output_dir=f'5_Test/{stage}',final_fit=True,evaluation_role='fitted_not_validation')
        _,selected=fit_starts(save_spec(fit,stage),statuspath,status)
        selectedpath=run/'reports'/f"{fit['id']}_selected.json"
        command=[sys.executable,'-B',str(Path(__file__).with_name('audit_and_export.py')),'--selected',str(selectedpath),'--label',label]
        with (run/'logs'/f'{label.lower()}_audit_export.log').open('a',encoding='utf-8') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError(f'{label} final replay/export audit failed')
        if label=='F24':
            command=[sys.executable,'-B',str(ROOT/'5_Test/20260905_5/scripts/profile_selected.py'),'--selected',str(selectedpath)]
            with (run/'logs'/'f24_identifiability.log').open('a',encoding='utf-8') as stream:
                process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError('F24 identifiability/profile diagnosis requires resolution')
        status['completed'].append(label);atomic_json(status,statuspath)
    status.update(status='FINAL_FITS_AND_EXPORTS_COMPLETE_REQUIRES_REQUIREMENT_AUDIT',active=None,updated_utc=utc_now());atomic_json(status,statuspath)
    print('CONFIRMATION_FINAL_FINISHED',candidate['candidate_id'],flush=True)


if __name__=='__main__':main()
