"""Executed full-history checks; reference results do not imply fit completion."""
from __future__ import annotations

from tn_reference import ROOT, RUNTIME, load_data, local_daily, local_monthly_coefficients, redistribute_by_water, route, station_predictions
from common import atomic_json, atomic_parquet, sha256, utc_now, memory_guard
import gc
import json
import time
import numpy as np
import pandas as pd

RUN=ROOT/'5_Test/20260905_2'
REPORT=RUN/'reports'
OUT=RUN/'outputs'


def max_relative(a,b):
    return float(np.max(np.abs(a-b)/np.maximum(np.maximum(np.abs(a),np.abs(b)),1)))


def local_ledger(data,result):
    fast,slow=data.monthly_sum(result['fast']),data.monthly_sum(result['slow'])
    stocks=result['mineral']+result['lower']
    previous=np.vstack([np.zeros((1,230)),stocks[:-1]])
    residual=data.source+previous-result['crop']-result['other']-fast-slow-stocks
    scale=data.source+previous
    passed=bool(np.all(np.abs(residual)<=1e-6+1e-10*np.maximum(scale,1)))
    return {'pass':passed,'max_abs_month_reach_kg_n':float(np.max(np.abs(residual))),
            'max_relative_month_reach':float(np.max(np.abs(residual)/np.maximum(scale,1))),
            'global_input_kg_n':float(data.source.sum()),'global_crop_kg_n':float(result['crop'].sum()),
            'global_other_kg_n':float(result['other'].sum()),'global_to_water_kg_n':float(fast.sum()+slow.sum()),
            'final_mineral_kg_n':float(result['mineral'][-1].sum()),'final_lower_kg_n':float(result['lower'][-1].sum()),
            'minimum_stock_kg_n':float(min(result['mineral'].min(),result['lower'].min()))}


def main():
    started=time.perf_counter()
    data=load_data('formal')
    print('REFERENCE_DATA_LOADED',len(data.dates),flush=True)
    p=pd.read_parquet(ROOT/'5_Test/20260904_7/outputs/f25_parameters.parquet').iloc[0]
    fitted={'log_alpha':float(p.log_alpha_contact),'beta':float(p.beta_contact),'tau':float(np.exp(p.log_tau_mineral_days))}
    vf=float(p.v_f)
    tests=[]
    for label,parameters in [('frozen_F25',fitted),('prior',{'log_alpha':-1.0,'beta':1.,'tau':365.25}),
                             ('long_lifetime',{'log_alpha':-4.,'beta':2.,'tau':3652.5})]:
        daily=local_daily(data,**parameters,corrected_lower=False)
        monthly=local_monthly_coefficients(data,**parameters,corrected_lower=False)
        errors={}
        for field in monthly:
            got=data.monthly_sum(daily[field]) if field in ['fast','slow'] else daily[field]
            errors[field]=max_relative(got,monthly[field])
            np.testing.assert_allclose(got,monthly[field],rtol=1e-10,atol=1e-6)
        ledger=local_ledger(data,daily)
        assert ledger['pass'] and ledger['minimum_stock_kg_n']>=0
        tests.append({'parameter_case':label,'parameters':parameters,'monthly_equation_relative_errors':errors,'ledger':ledger})
        print('LOCAL_REFERENCE_PASS',label,json.dumps(errors),flush=True)
        del daily,monthly;gc.collect()

    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    water_error=max_relative(water['official'],data.official_water)
    water_abs=float(np.max(np.abs(water['official']-data.official_water)))
    np.testing.assert_allclose(water['official'],data.official_water,rtol=1e-10,atol=1e-5)
    atomic_json({'status':'PASS','official_reach_water_max_relative_error':water_error,
                 'official_reach_water_max_abs_m3_day':water_abs,
                 'boundaries':'single-control post-dam; multiple-control pre-dam arms; released once at common downstream',
                 'preout_vs_official_max_m3_s':float(np.max(np.abs(water['preout']-data.official_water)))/86400},REPORT/'water_boundary_replay.json')
    print('WATER_BOUNDARY_PASS',water_error,flush=True)

    exact=local_daily(data,**fitted,corrected_lower=True)
    ledger=local_ledger(data,exact)
    local=exact['fast']+exact['slow']
    routed=route(data,local,vf=vf)
    ledger['channel_removed_kg_n']=float(routed['channel_removed'].sum())
    ledger['terminal_output_kg_n']=float(routed['terminal'].sum())
    ledger['final_reservoir_kg_n']=float(routed['stocks'][-1].sum())
    total=ledger['global_input_kg_n']
    residual=total-sum(ledger[k] for k in ['global_crop_kg_n','global_other_kg_n','final_mineral_kg_n',
                                          'final_lower_kg_n','channel_removed_kg_n','terminal_output_kg_n','final_reservoir_kg_n'])
    ledger['full_system_residual_kg_n']=residual
    ledger['full_system_relative_error']=abs(residual)/total
    assert abs(residual)<=1e-6+1e-10*total
    assert min(routed['stocks'].min(),routed['inlet'].min(),routed['official'].min())>=-1e-7
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet').loc[lambda x:x.year.le(2024)]
    pred=station_predictions(data,local,routed,water,obs,vf)
    assert np.isfinite(pred.prediction_mg_l).all()
    pred['variant']='exact_daily_corrected_lower'
    rows=[pred]
    monthly_exact={k:data.monthly_sum(exact[k]) for k in ['fast','slow']}
    month_table=pd.DataFrame({'year':np.repeat(data.months.year,230),'month':np.repeat(data.months.month,230),
                             'reach_id':np.tile(np.arange(1,231),len(data.months))})
    for field in ['other','crop','mineral','lower']:
        month_table[field+'_kg_n']=exact[field].ravel()
    for field in ['fast','slow']:
        month_table[field+'_kg_n']=monthly_exact[field].ravel()
    atomic_parquet(month_table,OUT/'reference_full_land_ledger.parquet')
    station_exact=pred.prediction_mg_l.to_numpy()
    exact_month_out=data.monthly_sum(routed['official'])
    del exact,routed,local,pred;gc.collect()

    old=local_monthly_coefficients(data,**fitted,corrected_lower=False)
    f,s=redistribute_by_water(data,old['fast'],old['slow'])
    legacy_local=f+s
    legacy_route=route(data,legacy_local,vf=vf)
    pred_old=station_predictions(data,legacy_local,legacy_route,water,obs,vf)
    pred_old['variant']='legacy_monthly_water_redistribution'
    rows.append(pred_old)
    delta=station_exact-pred_old.prediction_mg_l.to_numpy()
    out_delta=exact_month_out-data.monthly_sum(legacy_route['official'])
    timing={'parameters':fitted,'vf':vf,'same_parameters_refit':False,
            'station_abs_difference_mg_l_p50':float(np.quantile(np.abs(delta),.5)),
            'station_abs_difference_mg_l_p95':float(np.quantile(np.abs(delta),.95)),
            'station_abs_difference_mg_l_max':float(np.max(np.abs(delta))),
            'station_month_abs_difference_gt_0p01':int((np.abs(delta)>.01).sum()),
            'station_months':len(delta),'monthly_outlet_load_max_abs_difference_kg_n':float(np.max(np.abs(out_delta))),
            'interpretation':'fixed-parameter numerical/timing effect, not a newly trained performance result'}
    atomic_json(timing,REPORT/'daily_timing_effect.json')
    atomic_parquet(pd.concat(rows,ignore_index=True),OUT/'reference_station_comparison.parquet')
    del old,f,s,legacy_local,legacy_route;gc.collect()

    # Quantify the first-day repair separately from daily redistribution.
    uncorrected=local_daily(data,**fitted,corrected_lower=False)
    unc_route=route(data,uncorrected['fast']+uncorrected['slow'],vf=vf)
    unc_pred=station_predictions(data,uncorrected['fast']+uncorrected['slow'],unc_route,water,obs,vf)
    first_day_delta=station_exact-unc_pred.prediction_mg_l.to_numpy()
    first_day={'station_max_abs_effect_2016_2024_mg_l':float(np.max(np.abs(first_day_delta))),
               'station_p95_abs_effect_mg_l':float(np.quantile(np.abs(first_day_delta),.95))}
    atomic_json(first_day,REPORT/'first_day_repair_effect.json')
    tests_pass=all(t['ledger']['pass'] for t in tests)
    summary={'stage':'20260905_2','status':'PASS_REFERENCE_NUMERICS_FITTING_PENDING','created_utc':utc_now(),
             'runtime':RUNTIME,'runtime_seconds':time.perf_counter()-started,'local_tests':tests,
             'water_boundary_relative_error':water_error,'full_system_ledger':ledger,
             'daily_timing_effect':timing,'first_day_repair_effect':first_day,
             'memory':memory_guard(),'checks':{'three_parameter_cases_monthly_equivalent':tests_pass,
                'water_boundary_replay':True,'full_system_mass_closure':True,'finite_station_predictions':True},
             'pending':['differentiable_implementation_and_gradient_tests','old_structure_refits','complete_program_stages_3_to_6'],
             'code_sha256':{str(p):sha256(p) for p in [Path(__file__),RUN/'scripts/tn_reference.py']}}
    atomic_json(summary,REPORT/'reference_validation.json')
    print('REFERENCE_VALIDATION_COMPLETE',json.dumps({'timing':timing,'first_day':first_day,'seconds':summary['runtime_seconds']}),flush=True)


if __name__=='__main__':
    from pathlib import Path
    try:main()
    except Exception as exc:
        atomic_json({'status':'FAILED','at_utc':utc_now(),'type':type(exc).__name__,'message':str(exc)},REPORT/'last_error.json')
        raise
