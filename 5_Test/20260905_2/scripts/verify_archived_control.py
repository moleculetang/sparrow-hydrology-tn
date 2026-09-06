"""Reproduce archived F25 heads at their own 2021-2025 normalization window.

This diagnostic uses archived parameters only for numerical parity. It does
not produce initialization files and is never called by the training queue.
"""
from pathlib import Path
from fit_models import ROOT, FitObjective, load_data
from common import atomic_json, utc_now, sha256, memory_guard
import numpy as np
import pandas as pd
import torch


def main():
    data=load_data('sensitivity')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    train=obs.loc[obs.year.between(2021,2025)].copy()
    objective=FitObjective(data,train,'CONTROL_H7','STUDENT_T4_LOG1P')
    directory=ROOT/'5_Test/20260904_7/outputs'
    params=pd.read_parquet(directory/'f25_parameters.parquet').iloc[0]
    gamma=pd.read_parquet(directory/'f25_gamma_coefficients.parquet').set_index('feature').gamma
    sites=pd.read_parquet(directory/'f25_station_residuals.parquet').set_index('station_key').station_residual_log_unit
    theta=objective.initial(0)
    for i,name in enumerate(objective.names):
        if name in params.index:theta[i]=float(params[name])
        elif name.startswith('gamma_head_'):theta[i]=float(gamma.loc[objective.fields[int(name.rsplit('_',1)[1])]])
        elif name.startswith('site_'):theta[i]=float(sites.loc[objective.station_names[int(name.split('_')[1])]])
    # Main experiment fits start in 2016. Reproduction explicitly restores
    # the archived 2021-2025 Q normalization, without changing main fit code.
    for frame in [obs,objective.train]:
        c=objective.observations_cache(frame)
        ri=frame.reach_id.to_numpy(int)-1;f=frame.downstream_fraction_on_reach.to_numpy(float)
        selection=(data.months.year>=2021)&(data.months.year<=2025)
        seconds=(data.stops-data.starts)*86400.
        base=objective.water_inlet_month[selection][:,ri]+objective.water_local_month[selection][:,ri]*f
        logq=np.log(base/seconds[selection,None]);k=(len(logq)-1)//2
        median=np.partition(logq,k,axis=0)[k]
        ti=((frame.year.to_numpy()-1961)*12+frame.month.to_numpy()-1).astype(int)
        anomaly=np.log(c['water'].numpy()/seconds[ti])-median
        c['qlow']=torch.tensor(np.minimum(anomaly,0));c['qhigh']=torch.tensor(np.maximum(anomaly,0))
    with torch.no_grad():
        _,population=objective.predict(torch.tensor(theta),obs)
        loss=float(objective.loss(torch.tensor(theta)))
        rawtheta=theta.copy()
        for name in ['delta_path','beta_low','beta_high',*[f'gamma_head_{k}' for k in range(7)]]:
            rawtheta[objective.indices[name]]=0.
        _,raw=objective.predict(torch.tensor(rawtheta),obs)
    old=pd.read_parquet(directory/'f25_station_predictions_2016_2025.parquet')
    rows=[]
    for layer,lp in [('population_transferable',population),('raw_mass_process',raw)]:
        got=obs[['station_key','year','month']].copy();got['new_prediction']=np.maximum(np.expm1(lp.numpy()),0)
        comparison=got.merge(old.loc[old.layer.eq(layer),['station_key','year','month','pred_tn_mg_l']],
                             on=['station_key','year','month'],validate='one_to_one')
        assert len(comparison)==len(obs)
        delta=comparison.new_prediction-comparison.pred_tn_mg_l
        rows.append({'layer':layer,'rows':len(comparison),'max_abs_difference_mg_l':float(np.max(np.abs(delta))),
                     'p95_abs_difference_mg_l':float(np.quantile(np.abs(delta),.95))})
        np.testing.assert_allclose(comparison.new_prediction,comparison.pred_tn_mg_l,rtol=1e-9,atol=1e-9)
    archived=json.loads((ROOT/'5_Test/20260904_7/reports/f25_training_report.json').read_text(encoding='utf-8'))
    oldloss=archived['optimization']['final_objective']
    assert abs(loss-oldloss)<1e-9
    atomic_json({'status':'PASS_ARCHIVED_CONTROL_PARITY','created_utc':utc_now(),'comparisons':rows,
                 'loss':loss,'archived_loss':oldloss,'loss_abs_difference':abs(loss-oldloss),
                 'purpose':'numerical parity only; archived parameters are prohibited as new fit starts',
                 'q_normalization_override_years':[2021,2025],'memory':memory_guard(),
                 'code_sha256':sha256(Path(__file__))},ROOT/'5_Test/20260905_2/reports/archived_control_parity.json')
    print('ARCHIVED_CONTROL_PARITY_PASS',rows,loss,flush=True)


if __name__=='__main__':
    import json
    main()
