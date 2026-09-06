"""Exact station error decomposition; no corrected predictions are produced."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
import numpy as np
import pandas as pd
import json


def main():
    run=ROOT/'5_Test/20260905_3'
    dev=json.loads((run/'reports/development_summary.json').read_text(encoding='utf-8'))
    if not dev['status'].startswith('DEVELOPMENT_OOF_COMPLETE'):raise RuntimeError('Incomplete development OOF')
    rows=[];summaries={}
    paths={model:run/'outputs'/f'{model.lower()}_oof_predictions.parquet' for model in dev['summaries']}
    structural=ROOT/'5_Test/20260905_4/reports/structural_summary.json'
    if structural.exists():
        complete=json.loads(structural.read_text(encoding='utf-8'))
        if complete['status']=='STRUCTURAL_OOF_COMPLETE':
            paths.update({model:ROOT/'5_Test/20260905_4/outputs'/f'{model}_oof_predictions.parquet' for model in complete['outcomes']})
    for model,path in paths.items():
        frame=pd.read_parquet(path)
        block=[]
        for station,g in frame.groupby('station_key'):
            y=g.tn_mg_l.to_numpy(float);p=g.prediction_mg_l.to_numpy(float)
            vy=float(np.var(y));vp=float(np.var(p));cov=float(np.mean((y-y.mean())*(p-p.mean())))
            bias2=float((p.mean()-y.mean())**2);mse=float(np.mean((p-y)**2))
            dynamic=vy+vp-2*cov
            if not np.isclose(mse,bias2+dynamic,rtol=1e-12,atol=1e-12):raise AssertionError('MSE decomposition identity')
            if len(g)<8 or vy<=0:continue
            block.append(dict(model=model,station_key=station,reach_id=int(g.reach_id.iloc[0]),rows=len(g),
                observed_mean=float(y.mean()),predicted_mean=float(p.mean()),observed_sd=float(np.sqrt(vy)),predicted_sd=float(np.sqrt(vp)),
                signed_bias=float(p.mean()-y.mean()),mse=mse,bias_squared=bias2,dynamic_error=dynamic,
                bias_fraction_of_mse=bias2/mse if mse>0 else 0.,
                nse=1-mse/vy,mean_centered_nse_diagnostic=1-dynamic/vy,
                time_correlation=cov/np.sqrt(vy*vp) if vp>0 else np.nan,
                sd_ratio=float(np.sqrt(vp/vy))))
        table=pd.DataFrame(block);rows.extend(block)
        summaries[model]=dict(stations=len(table),median_nse=float(table.nse.median()),
            median_bias_fraction=float(table.bias_fraction_of_mse.median()),median_abs_bias_mg_l=float(table.signed_bias.abs().median()),
            median_mean_centered_nse_diagnostic=float(table.mean_centered_nse_diagnostic.median()),
            median_time_correlation=float(table.time_correlation.median()),median_predicted_observed_sd_ratio=float(table.sd_ratio.median()),
            prediction_sha256=sha256(path))
    atomic_parquet(pd.DataFrame(rows),run/'outputs/oof_error_decomposition.parquet')
    result=dict(status='EXACT_ERROR_DECOMPOSITION_COMPLETE',runtime=RUNTIME,created_utc=utc_now(),summaries=summaries,
        identity='MSE = squared mean bias + variance(observed) + variance(predicted) - 2 covariance(observed,predicted)',
        interpretation='The mean-centered quantity is an oracle diagnostic using evaluation means. It is not an allowable model prediction, achievable score, or fitted correction head.')
    atomic_json(result,run/'reports/oof_error_decomposition.json')
    print('ERROR_DECOMPOSITION',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
