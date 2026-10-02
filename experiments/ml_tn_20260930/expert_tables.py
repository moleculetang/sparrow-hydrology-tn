"""Descriptive tables from frozen outputs only; no fitting or model selection."""
from pathlib import Path
import numpy as np,pandas as pd
from mltn.common import ROOT,read,write,sha

def main():
    source=ROOT/'outputs/evaluation';out=ROOT/'outputs/expert_review';out.mkdir(exist_ok=True)
    metrics=pd.read_csv(source/'all_station_metrics.csv');rows=[]
    for (configuration,context,task),g in metrics.groupby(['configuration','context','task']):
        valid=g[g.nse_eligible.eq(True)];r=dict(configuration=configuration,context=context,task=task,total_station_rows=len(g),eligible_stations=len(valid),undefined_stations=len(g)-len(valid))
        for column in ['NSE','RMSE','bias','correlation','amplitude_ratio','month_centered_NSE','month_centered_RMSE','month_centered_correlation','month_centered_amplitude_ratio']:
            if column in valid:
                r[column+'_median']=valid[column].median();r[column+'_defined_stations']=int(valid[column].notna().sum())
        r['nonnegative_NSE_fraction']=float(valid.NSE.ge(0).mean()) if len(valid) else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/'configuration_summary.csv',index=False,encoding='utf-8-sig')
    pairs=read(source/'paired_summary.json');uncertainty=[];events=[]
    for pair in pairs:
        tag=f"{pair['candidate']}__versus__{pair['baseline']}_{pair['task']}"
        for block in [1,2]:
            file=source/f'{tag}_bootstrap_{block}month.parquet'
            if not file.exists():continue
            b=pd.read_parquet(file);assert len(b)==1000
            r={k:v for k,v in pair.items() if k!='centered'};r['block_months']=block;r['replicates']=len(b)
            for c in ['median_difference','median_paired_difference','improved_fraction','centered_median_paired_difference']:
                r[c+'_p025']=b[c].quantile(.025);r[c+'_p975']=b[c].quantile(.975);r[c+'_defined_replicates']=int(b[c].notna().sum())
            r['bootstrap_common_stations_min']=int(b.common_stations.min());r['bootstrap_common_stations_max']=int(b.common_stations.max());uncertainty.append(r)
        file=source/f'{tag}_paired_events.csv'
        if not file.exists() or file.stat().st_size<5:continue
        e=pd.read_csv(file)
        if e.empty:continue
        r={k:pair[k] for k in ['context','task','baseline','candidate']};r['events']=len(e);r['common_stations']=e.station_key.nunique()
        for c in ['amplitude_absolute_error_change','background_absolute_error_change','peak_value_absolute_error_change','NSE_change']:
            v=e[c].dropna();r[c+'_median']=v.median();r[c+'_defined_events']=len(v);r[c+'_improved_fraction']=float((v>0 if c=='NSE_change' else v<0).mean()) if len(v) else np.nan
        phase=e[e.unique_peak_phase_supported.eq(True)];r['unique_peak_phase_supported_events']=len(phase)
        r['peak_day_absolute_error_change_median']=phase.peak_day_absolute_error_change.median()
        r['peak_day_improved_fraction']=float(phase.peak_day_absolute_error_change.lt(0).mean()) if len(phase) else np.nan
        events.append(r)
    pd.DataFrame(uncertainty).to_csv(out/'paired_uncertainty_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(events).to_csv(out/'event_paired_summary.csv',index=False,encoding='utf-8-sig')
    attempts=[]
    roles=read(ROOT/'outputs/result_roles.json')
    for folder in sorted((ROOT/'jobs').iterdir()):
        if not folder.is_dir() or not (folder/'result.json').exists():continue
        result=read(folder/'result.json');start=read(folder/'start.json') if (folder/'start.json').exists() else {}
        attempts.append(dict(job_id=folder.name,included=folder.name in roles['include'],status=result.get('status'),stop_reason=result.get('stop_reason'),numerical_convergence=result.get('numerical_convergence'),epochs_executed=result.get('epochs_executed'),selected_epochs=result.get('selected_epochs'),optimizer_steps=result.get('optimizer_steps'),elapsed_s=result.get('elapsed_s'),owner=start.get('owner'),device=start.get('device'),threads=start.get('threads'),result_sha256=sha(folder/'result.json')))
    pd.DataFrame(attempts).to_csv(out/'training_attempts_summary.csv',index=False,encoding='utf-8-sig')
    training=[]
    for kind,file in [('individual','training_station_metrics.csv'),('three_seed_prediction_mean','training_seedmean_station_metrics.csv')]:
        if not (source/file).exists():continue
        t=pd.read_csv(source/file)
        for (job,task,period),g in t.groupby(['job','task','period']):
            v=g[g.nse_eligible.eq(True)];r=dict(job=job,task=task,period=period,kind=kind,eligible_stations=len(v),total_station_rows=len(g))
            for c in ['NSE','RMSE','bias','correlation','amplitude_ratio','month_centered_NSE']:
                if c in v:r[c+'_median']=v[c].median()
            training.append(r)
    pd.DataFrame(training).to_csv(out/'training_summary.csv',index=False,encoding='utf-8-sig')
    decomposition=source/'error_decomposition_station.csv'
    if decomposition.exists():
        d=pd.read_csv(decomposition);fields=[c for c in ['bias_fraction','amplitude_fraction','decorrelation_fraction','NSE','correlation','amplitude_ratio'] if c in d]
        d.groupby(['model_readout','scope'])[fields].median().reset_index().to_csv(out/'error_component_summary.csv',index=False,encoding='utf-8-sig')
    write(out/'receipt.json',dict(configuration_rows=len(rows),registered_pairs=len(pairs),uncertainty_rows=len(uncertainty),event_pairs=len(events),attempts=len(attempts),source_metrics_sha256=sha(source/'all_station_metrics.csv'),source_pairs_sha256=sha(source/'paired_summary.json'),selection='none; summaries do not modify configurations, predictions or labels',uncertainty='percentile intervals of existing 1000 synchronous month/block draws; descriptive retrospective uncertainty',phase='only events with common frozen peak support and unique observed/both predicted peaks contribute to phase summary'))

if __name__=='__main__':main()
