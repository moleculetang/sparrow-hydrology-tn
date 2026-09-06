"""Select starts on training objective; evaluate common-cohort temporal OOF.

Incomplete matrices produce inventory only. Never select a model from partial
folds, failed stationarity, or evaluation objective.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME, atomic_json, atomic_parquet, sha256, utc_now
RUN=ROOT/'5_Test/20260905_3'


def station_metrics(frame):
    if frame.observation_id.duplicated().any():
        raise ValueError('Repeated OOF observation IDs')
    rows=[]
    for key,g in frame.groupby('station_key',sort=True):
        if g.terminal_tree_id.nunique()!=1: raise ValueError('Changing station tree')
        y=g.tn_mg_l.to_numpy(float);p=g.prediction_mg_l.to_numpy(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(p<0):
            raise ValueError('Invalid prediction/observation; never silently drop')
        sst=np.sum((y-y.mean())**2)
        rows.append(dict(station_key=key,reach_id=int(g.reach_id.iloc[0]),
            terminal_tree_id=int(g.terminal_tree_id.iloc[0]),rows=len(g),
            nse=1-np.sum((y-p)**2)/sst if len(g)>=8 and sst>0 else np.nan,
            nse_status='defined' if len(g)>=8 and sst>0 else ('fewer_than_8' if len(g)<8 else 'constant_observed'),
            time_r=np.corrcoef(y,p)[0,1] if len(g)>=8 and np.std(y)>0 and np.std(p)>0 else np.nan,
            log_rmse=np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2)),
            bias_mg_l=np.mean(p-y),rmse_mg_l=np.sqrt(np.mean((p-y)**2))))
    return pd.DataFrame(rows)


def summary(metrics,frame):
    n=metrics.nse.dropna();r=metrics.time_r.dropna()
    y=frame.tn_mg_l.to_numpy(float);p=frame.prediction_mg_l.to_numpy(float)
    sst=np.sum((y-y.mean())**2)
    return dict(rows=len(frame),stations=len(metrics),defined_nse=len(n),
        nse_status=metrics.nse_status.value_counts().to_dict(),
        median_nse=float(n.median()) if len(n) else None,
        q25_nse=float(n.quantile(.25)) if len(n) else None,
        fraction_nse_positive=float((n>0).mean()) if len(n) else None,
        median_time_r=float(r.median()) if len(r) else None,
        mean_station_log_rmse=float(metrics.log_rmse.mean()),
        mean_station_bias_mg_l=float(metrics.bias_mg_l.mean()),
        descriptive_pooled_nse=float(1-np.sum((p-y)**2)/sst) if sst>0 else None)


def paired_bootstrap(candidate,control,reps=10000,seed=260905):
    merged=candidate.merge(control,on='station_key',suffixes=('_new','_old'),validate='one_to_one')
    valid=merged.nse_new.notna()&merged.nse_old.notna()
    m=merged.loc[valid].reset_index(drop=True)
    if not (m.terminal_tree_id_new==m.terminal_tree_id_old).all(): raise ValueError('Tree mismatch')
    if len(m)<2:return {'status':'insufficient_defined_stations'}
    a,b=m.nse_new.to_numpy(),m.nse_old.to_numpy()
    rng=np.random.default_rng(seed);station=[];tree=[]
    groups=[np.flatnonzero(m.terminal_tree_id_new.to_numpy()==t) for t in sorted(m.terminal_tree_id_new.unique())]
    for _ in range(reps):
        i=rng.integers(0,len(m),len(m));station.append(np.median(a[i])-np.median(b[i]))
        i=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        tree.append(np.median(a[i])-np.median(b[i]))
    return dict(status='computed',paired_stations=len(m),trees=len(groups),replicates=reps,seed=seed,
        estimand='difference of station medians on paired identical observations',
        delta_median_nse=float(np.median(a)-np.median(b)),
        median_paired_station_delta=float(np.median(a-b)),
        station_ci95=np.quantile(station,[.025,.975]).tolist(),
        tree_cluster_ci95=np.quantile(tree,[.025,.975]).tolist(),
        tree_ci_interpretation='few terminal trees limit cluster-bootstrap precision')


def main():
    queue=json.loads((RUN/'reports/development_queue.json').read_text(encoding='utf-8'))
    expected=queue['jobs']; rows=[]; reports={};missing=[]
    for job in expected:
        path=Path(job['report'])
        if not path.exists():missing.append(job['tag']);continue
        report=json.loads(path.read_text(encoding='utf-8'));identity=report['identity']
        for p,h in identity['code_sha256'].items():
            if sha256(p)!=h:raise RuntimeError(f'Stale fitted code {p}')
        if identity['input_manifest_sha256']!=sha256(ROOT/'5_Test/20260905_1/reports/input_manifest.json'):
            raise RuntimeError('Stale fitted input manifest')
        if any(identity[k]!=job[k] for k in ['model','loss','fold','start']):raise ValueError('Job identity mismatch')
        passed=bool(report['converged'] and report['projected_gradient_max']<=1e-5)
        rows.append(dict(tag=job['tag'],model=identity['model'],loss=identity['loss'],fold=identity['fold'],
            start=identity['start'],converged=passed,objective=report['objective'],
            projected_gradient_max=report['projected_gradient_max'],seconds=report['seconds'],
            evaluation_primary_median_nse=report['evaluation_primary_median_nse']))
        reports[job['tag']]=report
    inventory=pd.DataFrame(rows)
    atomic_parquet(inventory,RUN/'outputs/development_fit_inventory.parquet')
    result=dict(created_utc=utc_now(),runtime=RUNTIME,expected_fits=len(expected),available_fits=len(rows),
        converged_fits=int(inventory.converged.sum()) if len(rows) else 0,missing=missing,
        status='INCOMPLETE_INVENTORY_ONLY',promotion=False)
    if missing or not inventory.converged.all():
        atomic_json(result,RUN/'reports/development_summary.json')
        print('SUMMARY_INCOMPLETE',len(rows),len(expected),result['converged_fits'],flush=True);return
    selected=inventory.sort_values(['objective','start']).groupby(['model','loss','fold'],sort=True).head(1)
    atomic_parquet(selected,RUN/'outputs/development_selected_starts.parquet')
    observations=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    expected_obs=observations.loc[observations.year.between(2020,2023)&observations.primary_gate]
    frames={};metrics={};summaries={}
    for (model,loss),group in selected.groupby(['model','loss'],sort=True):
        key=model+'__'+loss;parts=[]
        for row in group.itertuples():
            stage='20260905_2' if model=='CONTROL_H7' else '20260905_3'
            p=pd.read_parquet(ROOT/'5_Test'/stage/'outputs'/f'{row.tag}_predictions.parquet')
            p=p.loc[p.primary_gate].copy();p['fold']=row.fold
            train_ids=reports[row.tag]['identity']['train_observation_ids']
            seen=set(observations.loc[observations.observation_id.isin(train_ids),'station_key'])
            p['station_seen_in_training']=p.station_key.isin(seen)
            parts.append(p)
        frame=pd.concat(parts,ignore_index=True).sort_values('observation_id')
        if set(frame.observation_id)!=set(expected_obs.observation_id):raise ValueError('OOF coverage mismatch')
        frames[key]=frame;metrics[key]=station_metrics(frame);summaries[key]=summary(metrics[key],frame)
        summaries[key]['seen_status']={}
        for flag,g in frame.groupby('station_seen_in_training'):
            summaries[key]['seen_status'][str(flag)]=summary(station_metrics(g),g)
        atomic_parquet(frame,RUN/'outputs'/f'{key.lower()}_oof_predictions.parquet')
        atomic_parquet(metrics[key],RUN/'outputs'/f'{key.lower()}_oof_station_metrics.parquet')
    control='CONTROL_H7__STUDENT_T4_LOG1P';comparisons={}
    baseline_parts=[]
    for row in selected.loc[selected.model.eq('CONTROL_H7')].itertuples():
        train_ids=reports[row.tag]['identity']['train_observation_ids']
        train=observations.loc[observations.observation_id.isin(train_ids)]
        means=train.groupby('station_key').tn_mg_l.mean()
        baseline=expected_obs.loc[expected_obs.year.eq(int(row.fold[1:]))].copy()
        baseline['prediction_mg_l']=baseline.station_key.map(means).fillna(float(means.mean()))
        baseline['baseline_source']=np.where(baseline.station_key.isin(means.index),'training_station_mean','training_equal_station_global_mean_for_unseen')
        baseline_parts.append(baseline)
    baseline=pd.concat(baseline_parts,ignore_index=True)
    baseline_summary=summary(station_metrics(baseline),baseline)
    atomic_parquet(baseline,RUN/'outputs/training_mean_diagnostic_baseline_oof.parquet')
    for key in frames:
        if key==control:continue
        new,old=summaries[key],summaries[control]
        paired=paired_bootstrap(metrics[key],metrics[control])
        gates=dict(median_positive=new['median_nse']>0,
            median_delta_ge_010=paired['delta_median_nse']>=.1,
            paired_station_ci_positive=paired['station_ci95'][0]>0,
            paired_tree_ci_positive=paired['tree_cluster_ci95'][0]>0,
            q25_noninferior=new['q25_nse']>=old['q25_nse']-.05,
            log_rmse_noninferior=new['mean_station_log_rmse']<=old['mean_station_log_rmse']*1.05)
        comparisons[key]={'paired':paired,'temporal_gates':gates,'all_temporal_gates':all(gates.values())}
    rank=sorted((k for k in summaries if k!=control),key=lambda k:(-summaries[k]['median_nse'],k))
    result.update(status='DEVELOPMENT_OOF_COMPLETE_REQUIRES_SPATIAL_AND_CONFIRMATION',summaries=summaries,
        comparisons=comparisons,development_rank=rank,selected_start_rule='minimum training objective among five stationary starts',
        diagnostic_training_mean_baseline=baseline_summary,
        baseline_interpretation='diagnostic only; seen stations use training mean, unseen stations use equal-station global training mean; not a TN process candidate',
        remaining=['structural_timing','nested_spatial','2024_confirmation','prior_sensitivity','final_fits','exports'])
    atomic_json(result,RUN/'reports/development_summary.json')
    print('SUMMARY_COMPLETE',json.dumps({k:v['median_nse'] for k,v in summaries.items()}),flush=True)


if __name__=='__main__':main()
