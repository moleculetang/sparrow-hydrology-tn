"""Independent physical replay, source accounts and optional scenario export.

Prediction here never accepts TN labels. Trained scalar/vector parameters and
saved fold scalers are sufficient; labels are read only for final comparison.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse
import numpy as np
import pandas as pd
import torch
from fc_io import *
from fc_data import load_data,Scaler,daily_drivers,prediction_metadata,observations
from fc_independent import replay
from fc_legacy import route


def decode_artifact(artifact):
    state=artifact['model'];values={k.removeprefix('values.'):v.detach().cpu().numpy().copy() for k,v in state.items() if k.startswith('values.')}
    network=[]
    if artifact['identity']['spec']['family']=='P_MCN3':
        network=[(state[f'network.{i}.weight'].numpy(),state[f'network.{i}.bias'].numpy()) for i in [0,2]]
    return {'family':artifact['identity']['spec']['family'],'values':values,'network':network}


def observe(data,local,routed,water,metadata,vf):
    if 'tn_mg_l' in metadata:raise ValueError('Independent inference must not accept TN labels')
    ti=((metadata.year.to_numpy()-1961)*12+metadata.month.to_numpy()-1).astype(int);ri=metadata.reach_id.to_numpy(int)-1
    types=metadata.station_type.to_numpy();fraction=np.where(types=='predam',1.,metadata.downstream_fraction_on_reach.to_numpy(float))
    reservoir=metadata.reservoir_index.to_numpy(int)
    if np.any((types=='dam_outlet')&(reservoir<0)):raise ValueError('Unregistered reservoir boundary')
    reservoir=np.maximum(reservoir,0);h=data.h_month[ti,ri]
    mass=data.monthly_sum(routed['inlet'])[ti,ri]*np.exp(-vf*h*fraction)+fraction*data.monthly_sum(local)[ti,ri]*np.exp(-.5*vf*h*fraction)
    volume=data.monthly_sum(water['inlet'])[ti,ri]+fraction*data.monthly_sum(data.fast_water+data.slow_water)[ti,ri]
    mass=np.where(types=='dam_outlet',data.monthly_sum(routed['releases'])[ti,reservoir],np.where(types=='postdam_mixed',data.monthly_sum(routed['official'])[ti,ri],mass))
    volume=np.where(types=='dam_outlet',data.monthly_sum(water['releases'])[ti,reservoir],np.where(types=='postdam_mixed',data.monthly_sum(water['official'])[ti,ri],volume))
    if np.any(volume<=0):raise ValueError('Nonpositive water at observation boundary')
    return 1000*mass/volume


def replay_model(data,artifact,scenario='baseline',history_tags=False,keep_source_routing=False):
    description=decode_artifact(artifact);design=artifact['design'];nr=len(data.area_ha);nd=len(data.dates);nm=len(data.months)
    static=Scaler.from_dict(design['static']['scaler']).transform(data.static_raw)
    dynamic=eta=None
    if description['family']=='P_MCN3':
        raw,_=daily_drivers(data);dynamic=Scaler.from_dict(design['dynamic']['scaler']).transform(raw)
        raw_eta=np.stack([np.log1p(data.upper_water),np.log1p(data.percolation)],axis=-1)
        eta=Scaler.from_dict(design['dynamic']['eta_scaler']).transform(raw_eta)
        del raw,raw_eta
    source=data.source_tags.copy();future=data.months.year>=2020
    if scenario=='fertilizer_minus20':source[future,:,0]*=.8
    elif scenario=='manure_minus20':source[future,:,1]*=.8
    elif scenario=='fertilizer_manure_minus20':source[future,:,:2]*=.8
    elif scenario=='cessation_2020':source[future]=0.
    elif scenario!='baseline':raise ValueError('Unregistered scenario')
    # Changing inputs changes both mass and the source drivers in the neural
    # hazard network. Saved scalers remain fixed; do not freeze baseline drivers.
    if description['family']=='P_MCN3' and scenario!='baseline':
        import copy
        changed=copy.copy(data);changed.source_tags=source
        raw,_=daily_drivers(changed);dynamic=Scaler.from_dict(design['dynamic']['scaler']).transform(raw);del raw
    if history_tags:
        tagged=np.zeros((nm,nr,8));tagged[~future,:,:4]=source[~future];tagged[future,:,4:]=source[future]
        source=tagged;parent=[0,1,2,3,0,1,2,3]
    else:parent=None
    ns=source.shape[-1];state=None;local=np.zeros((nd,nr));local_tags=np.zeros((nd,nr,ns)) if keep_source_routing else None
    monthly_flux=np.zeros((nm,nr,6,ns));monthly_stocks=np.zeros((nm,nr,3,ns));daily_max_rel=0.;minimum=0.;uptake_excess=0.
    first=np.r_[True,np.diff(data.mid)!=0]
    for m,(a,b) in enumerate(zip(data.starts,data.stops)):
        guard()
        src=source[data.mid[a:b]]*first[a:b,None,None];demand=data.crop[data.mid[a:b]]*first[a:b,None]
        prior=np.zeros((nr,3,ns)) if state is None else state
        result=replay(description,src,demand,data.contact[a:b],data.fast_fraction[a:b],data.lower_release[a:b],static,
                      np.broadcast_to(np.zeros(15),(b-a,nr,15)) if dynamic is None else dynamic[a:b],
                      np.broadcast_to(np.zeros(2),(b-a,nr,2)) if eta is None else eta[a:b],data.area_ha,state=state,source_parent_index=parent)
        states=result['stocks'];flux=result['fluxes'];previous=np.concatenate([prior[None,:,:,:],states[:-1]],axis=0)
        balance=previous.sum(axis=2)+src-states.sum(axis=2)-flux[:,:,:4,:].sum(axis=2)
        scale=np.maximum(1.,previous.sum(axis=2)+src)
        daily_max_rel=max(daily_max_rel,float(np.max(np.abs(balance)/scale)))
        minimum=min(minimum,float(states.min()),float(flux.min()))
        uptake_excess=max(uptake_excess,float(np.max(flux[:,:,2,:].sum(axis=-1)-demand)))
        state=result['state'];monthly_flux[m]=flux.sum(axis=0);monthly_stocks[m]=state
        tags=flux[:,:,:2,:].sum(axis=2);local[a:b]=tags.sum(axis=-1)
        if local_tags is not None:local_tags[a:b]=tags
    initial_and_source=source.sum(axis=(0,1));out=monthly_flux[:,:,:4,:].sum(axis=(0,1,2));ending=state.sum(axis=(0,1))
    residual=initial_and_source-out-ending;relative=float(np.max(np.abs(residual)/np.maximum(1.,initial_and_source)))
    vf=float(description['values']['v_f']);routed=route(data,local,vf=vf,exposure='monthly')
    river_residual=float(local.sum()-routed['channel_removed'].sum()-routed['terminal'].sum()-routed['stocks'][-1].sum())
    river_relative=abs(river_residual)/max(1.,float(local.sum()))
    passed=daily_max_rel<1e-10 and relative<1e-10 and river_relative<1e-10 and minimum>=-1e-8 and uptake_excess<=1e-6
    annual=[]
    for year in sorted(set(data.months.year)):
        mask=data.months.year==year;entry={'year':int(year),'source':float(source[mask].sum()),'demand':float(data.crop[mask].sum())}
        for index,name in enumerate(['fast','slow','uptake','loss','F_to_M','slow_injection']):entry[name]=float(monthly_flux[mask,:,index,:].sum())
        entry['uptake_demand_ratio']=entry['uptake']/entry['demand'] if entry['demand'] else 0.
        annual.append(entry)
    result={'physical_pass':passed,'annual':annual,'maximum_relative_daily_balance':daily_max_rel,'full_history_relative_balance':relative,
            'source_residual_kg_n':residual.tolist(),'river_relative_balance':river_relative,'river_residual_kg_n':river_residual,
            'minimum_stock_or_flux':minimum,'maximum_uptake_excess_kg_n':uptake_excess,'scenario':scenario,'history_tags':history_tags}
    return result,{'local':local,'local_tags':local_tags,'routed':routed,'source':source,'flux':monthly_flux,'stocks':monthly_stocks}


def audit_fit(tag,scenario='baseline',history_tags=False,export=False):
    require_environment();guard(5)
    fit=read(RUN/'reports/fits'/f'{tag}.json')
    model_path=Path(fit['model_path'])
    if sha(model_path)!=fit['model_sha256']:raise RuntimeError('Model artifact identity changed')
    artifact=torch.load(model_path,map_location='cpu',weights_only=False)
    if artifact['identity']!=fit['identity']:raise RuntimeError('Model and fit identities differ')
    spec=fit['identity']['spec']
    if spec['family'] not in ['P_KERNEL','P_MCN3']:raise ValueError('Empirical family has no physical audit')
    data=load_data(spec['product'],spec['forcing'])
    audit,values=replay_model(data,artifact,scenario,history_tags,keep_source_routing=export)
    label=tag+('_history' if history_tags else '')+('' if scenario=='baseline' else '_'+scenario)
    artifacts={}
    if scenario=='baseline':
        expected=pd.read_parquet(fit['prediction_path'])
        water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
        prediction=observe(data,values['local'],values['routed'],water,prediction_metadata(expected),float(decode_artifact(artifact)['values']['v_f']))
        np.testing.assert_allclose(prediction,expected.prediction_mg_l.to_numpy(),rtol=3e-8,atol=3e-8)
        audit['maximum_prediction_error_mg_l']=float(np.max(np.abs(prediction-expected.prediction_mg_l.to_numpy())))
    ns=values['source'].shape[-1]
    frame=pd.DataFrame({'month':np.repeat(data.months,len(data.area_ha)*ns),'reach_id':np.tile(np.repeat(np.arange(1,len(data.area_ha)+1),ns),len(data.months)),
                        'source_tag':np.tile(np.arange(ns),len(data.months)*len(data.area_ha)),'input_kg_n':values['source'].ravel()})
    for i,name in enumerate(['fast','slow','uptake','loss','F_to_M','slow_injection']):frame[name+'_kg_n']=values['flux'][:,:,i,:].ravel()
    for i,name in enumerate(['F_or_short','M_or_long','slow_water']):frame[name+'_stock_kg_n']=values['stocks'][:,:,i,:].ravel()
    output=local(RUN/'outputs/physical'/f'{label}_monthly_ledger.parquet');temp=output.with_suffix('.tmp');frame.to_parquet(temp,index=False);replace(temp,output)
    artifacts[str(output)]=sha(output)
    if export:
        vf=float(decode_artifact(artifact)['values']['v_f'])
        source_exports=[]
        for k in range(ns):
            routed=route(data,values['local_tags'][:,:,k],vf=vf,exposure='monthly')
            source_exports.append(data.monthly_sum(routed['official']))
        tagged=np.stack(source_exports,axis=-1)
        np.testing.assert_allclose(tagged.sum(axis=-1),data.monthly_sum(values['routed']['official']),rtol=3e-8,atol=1e-6)
        frame['outlet_kg_n']=tagged.ravel()
        output=local(RUN/'outputs/physical'/f'{label}_source_outlets.parquet');temp=output.with_suffix('.tmp');frame.to_parquet(temp,index=False);replace(temp,output)
        artifacts[str(output)]=sha(output)
    audit.update(status='PASS_PHYSICAL_REPLAY' if audit['physical_pass'] else 'FAILED_PHYSICAL_REPLAY',utc=now(),
                 identity={'fit_sha256':sha(RUN/'reports/fits'/f'{tag}.json'),'model_sha256':fit['model_sha256']},artifacts=artifacts)
    write(RUN/'reports/physical'/f'{label}.json',audit)
    if not audit['physical_pass']:raise RuntimeError('PHYSICAL_BALANCE_FAILED '+label)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--scenario',default='baseline');p.add_argument('--history',action='store_true');p.add_argument('--export',action='store_true');args=p.parse_args()
    try:audit_fit(args.tag,args.scenario,args.history,args.export)
    except MemoryError:raise SystemExit(75)
    print('PASS_PHYSICAL_REPLAY',args.tag,args.scenario,flush=True)


if __name__=='__main__':main()
