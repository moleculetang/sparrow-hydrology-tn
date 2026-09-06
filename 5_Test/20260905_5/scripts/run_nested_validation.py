"""Nested reach/tree validation with no outer TN data in model selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_4/scripts'))
sys.path.insert(0,str(ROOT/'5_Test/20260905_3/scripts'))
from experiment_io import save_spec,fit_starts
from run_structural_trials import candidates as structural_candidates
from summarize_development import station_metrics,summary,paired_bootstrap
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
import pandas as pd
import json
import os
import subprocess


def candidate_set():
    result=[]
    for model in ['GLOBAL','H7_CONTACT','H7_CONTACT_LIFETIME']:
        for loss in ['STATION_NORMALIZED_MSE','STUDENT_T4_LOG1P']:
            result.append(dict(candidate_id=model.lower()+'_'+loss.lower(),model=model,loss=loss,
                calendar='CENTRAL',timing='monthly_pulse',dynamic=False,prior_scale=1.))
    result.extend(structural_candidates())
    return result


def main():
    stage='20260905_5';run=ROOT/'5_Test'/stage
    inference_check=ROOT/'5_Test/20260905_4/scripts/validate_inference_only.py'
    completed=subprocess.run([sys.executable,'-B',str(inference_check)],cwd=ROOT)
    if completed.returncode:raise RuntimeError('Independent history-free inference validation failed')
    registry_path=run/'reports/nested_split_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    if any(sha256(p)!=h for p,h in registry['inputs'].items()):raise RuntimeError('Split inputs changed')
    configs=candidate_set()
    # All six base models and all four prespecified single additions are tested
    # in every outer split, independent of all-domain temporal performance.
    candidate_registry=dict(candidates=configs,selection='inner spatial OOF median station NSE',
        ordering_tiebreak='candidate_id lexical',outer_TN_used_for_candidate_pruning=False,
        conditional_combination='best independently supported timing plus independently supported dynamic contact, using inner OOF only')
    cp=run/'reports/nested_candidate_registry.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=candidate_registry:raise RuntimeError('Candidate registry changed')
    atomic_json(candidate_registry,cp)
    statuspath=run/'reports/nested_queue.json';status=dict(status='RUNNING',pid=os.getpid(),runtime=RUNTIME,started_utc=utc_now(),completed=[])
    outer_predictions=[];control_predictions=[];selections=[]
    control=dict(candidate_id='control_h7',model='CONTROL_H7',loss='STUDENT_T4_LOG1P',calendar='CENTRAL',timing='monthly_pulse',dynamic=False,prior_scale=1.)
    def spec(candidate,split):
        return dict(candidate,id=candidate['candidate_id']+'_'+split['id'].lower(),
            train_ids=split['train_ids'],eval_ids=split['eval_ids'],heldout_reaches=split['heldout_reaches'],
            require_eval_heldout=True,product='formal',output_dir=f'5_Test/{stage}',evaluation_role='nested_spatial',split_id=split['id'])
    for kind,outers in [('reach',registry['outer_reach']),('tree',registry['outer_terminal_tree'])]:
        for outer in outers:
            scores=[];local_configs=list(configs);inner_frames={}
            for candidate in configs:
                parts=[]
                for inner in outer['inner']:
                    fit=spec(candidate,inner);pred,_=fit_starts(save_spec(fit,stage),statuspath,status);parts.append(pred)
                pred=pd.concat(parts,ignore_index=True)
                if set(pred.reach_id)&set(outer['eval_reaches']):raise ValueError('Outer reach in inner validation')
                sm=station_metrics(pred);s=summary(sm,pred)
                inner_frames[candidate['candidate_id']]=pred
                scores.append(dict(candidate_id=candidate['candidate_id'],summary=s))
                atomic_parquet(pred,run/'outputs'/f"{candidate['candidate_id']}_{outer['id'].lower()}_inner_oof.parquet")
                atomic_json(scores,run/'reports'/f"{outer['id'].lower()}_inner_scores_partial.json")
                print('NESTED_INNER_CANDIDATE',outer['id'],candidate['candidate_id'],s['median_nse'],flush=True)
            base_key='h7_contact_lifetime_station_normalized_mse'
            base_metrics=station_metrics(inner_frames[base_key]);base_summary=summary(base_metrics,inner_frames[base_key])
            def supported(key):
                m=station_metrics(inner_frames[key]);s=summary(m,inner_frames[key]);p=paired_bootstrap(m,base_metrics)
                return (p['station_ci95'][0]>0 and p['tree_cluster_ci95'][0]>0
                    and s['q25_nse']>=base_summary['q25_nse']-.05
                    and s['mean_station_log_rmse']<=base_summary['mean_station_log_rmse']*1.05)
            timing=[key for key in ['uniform_daily','calendar_early','calendar_late'] if supported(key)]
            if timing and supported('contact_dynamic'):
                score_map={r['candidate_id']:r['summary']['median_nse'] for r in scores}
                best_timing=sorted(timing,key=lambda k:(-score_map[k],k))[0]
                combo=dict(next(c for c in configs if c['candidate_id']==best_timing))
                combo.update(candidate_id='timing_plus_dynamic',dynamic=True);local_configs.append(combo)
                parts=[]
                for inner in outer['inner']:
                    fit=spec(combo,inner);pred,_=fit_starts(save_spec(fit,stage),statuspath,status);parts.append(pred)
                pred=pd.concat(parts,ignore_index=True);s=summary(station_metrics(pred),pred)
                scores.append(dict(candidate_id=combo['candidate_id'],summary=s,inner_independent_support=True))
                atomic_parquet(pred,run/'outputs'/f"timing_plus_dynamic_{outer['id'].lower()}_inner_oof.parquet")
            valid=[s for s in scores if s['summary']['median_nse'] is not None]
            if not valid:raise RuntimeError('No defined inner station NSE')
            winner=sorted(valid,key=lambda s:(-s['summary']['median_nse'],s['candidate_id']))[0]
            chosen=next(c for c in local_configs if c['candidate_id']==winner['candidate_id'])
            for candidate,label in [(chosen,'new'),(control,'control')]:
                fit=spec(candidate,outer);pred,selected=fit_starts(save_spec(fit,stage),statuspath,status)
                if pred.station_seen_in_training.any():raise ValueError('Station history in outer prediction')
                pred['outer_id']=outer['id'];pred['outer_kind']=kind
                (outer_predictions if label=='new' else control_predictions).append(pred)
                atomic_parquet(pred,run/'outputs'/f"{outer['id'].lower()}_{label}_predictions.parquet")
            selection=dict(outer_id=outer['id'],outer_kind=kind,chosen=chosen,inner_scores=scores,
                outer_train_rows=len(outer['train_ids']),outer_eval_rows=len(outer['eval_ids']))
            selections.append(selection);atomic_json(selection,run/'reports'/f"{outer['id'].lower()}_selection.json")
            status['completed'].append(outer['id']);atomic_json(status,statuspath)
            print('NESTED_OUTER_FINISHED',outer['id'],chosen['candidate_id'],flush=True)
    result={}
    for kind in ['reach','tree']:
        new=pd.concat([p for p in outer_predictions if p.outer_kind.iloc[0]==kind],ignore_index=True)
        old=pd.concat([p for p in control_predictions if p.outer_kind.iloc[0]==kind],ignore_index=True)
        if set(new.observation_id)!=set(old.observation_id):raise ValueError('Spatial control cohort mismatch')
        nm,om=station_metrics(new),station_metrics(old);ns,osummary=summary(nm,new),summary(om,old)
        paired=paired_bootstrap(nm,om)
        gates=dict(median_not_materially_worse=ns['median_nse']>=osummary['median_nse']-.05,
            q25_noninferior=ns['q25_nse']>=osummary['q25_nse']-.05,
            station_log_rmse_noninferior=ns['mean_station_log_rmse']<=osummary['mean_station_log_rmse']*1.05)
        result[kind]=dict(new=ns,control=osummary,paired=paired,nonregression_gates=gates,passes_nonregression=all(gates.values()))
        atomic_parquet(new,run/'outputs'/f'nested_{kind}_new_oof.parquet');atomic_parquet(old,run/'outputs'/f'nested_{kind}_control_oof.parquet')
    status.update(status='NESTED_VALIDATION_COMPLETE',active=None,updated_utc=utc_now());atomic_json(status,statuspath)
    atomic_json(dict(status=status['status'],runtime=RUNTIME,comparisons=result,selections=selections,
        material_spatial_regression_definition='median or q25 NSE decline >0.05, or mean station log-RMSE increase >5%',
        interpretation='nested zero-TN-history transfer conditional on previously calibrated hydrology'),run/'reports/nested_validation_summary.json')


if __name__=='__main__':main()
