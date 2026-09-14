"""Implementation and operational tests; no fit or heldout score is computed."""
import os
from pathlib import Path
os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'work/numba_cache')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc
import time
import numpy as np
import pandas as pd
import torch
from fc_io import *
from fc_data import prediction_metadata
from sc_model import SharedBox,data_cache,wetting,SIGMA
from sc_physical import replay_artifact,prediction
from sc_kernel import forward
from sc_runtime import exclusive,checkpoint,restore,continuation,admission
from fc_resources import process_snapshot

def main():
    require_environment();torch.set_num_threads(1);started=time.time();checks={};details={}
    data=data_cache('sensitivity');fold='F23';cfg=read(RUN/f'configs/{fold}.json')
    train=pd.read_parquet(RUN/f'outputs/data/{fold}_train.parquet');meta=prediction_metadata(train)
    base=SharedBox(data,train);model=SharedBox(data,train,'SC',cfg['scale'])
    x=base.initial(1);xx=np.r_[x,np.zeros(6)]
    rb=base.residual(x);rr=model.residual(xx)
    np.testing.assert_allclose(rb,rr[:len(rb)],rtol=2e-12,atol=2e-12);assert np.all(rr[len(rb):]==0)
    fb,gb=base.value_gradient(x);ff,gg=model.value_gradient(xx)
    np.testing.assert_allclose(gb,gg[:21],rtol=1e-9,atol=1e-9)
    with torch.no_grad():old=float(base.context.loss(torch.tensor(x)))
    np.testing.assert_allclose(fb,old,rtol=1e-12,atol=1e-12)
    checks['zero_extension_and_original_RAW_MAP']=True
    xx[-6:]=SIGMA*np.array([1,-1,1,-1,1,-1]);f,g=model.value_gradient(xx)
    rows=[]
    for i,name in enumerate(model.names):
        step=1e-5*model.variable_scale()[i];xp=xx.copy();xm=xx.copy();xp[i]+=step;xm[i]-=step
        rp=model.residual(xp);rm=model.residual(xm);fd=(.5*rp@rp-.5*rm@rm)/(2*step)
        err=abs(fd-g[i]);tol=2e-6+3e-5*max(abs(fd),abs(g[i]))
        rows.append({'parameter':name,'analytic':float(g[i]),'central':float(fd),'absolute_error':float(err),'tolerance':float(tol),'pass':bool(err<=tol)})
        print('GRADIENT',name,err<=tol,flush=True)
    details['full_history_gradient']=rows;checks['all27_full_history_derivatives']=all(r['pass'] for r in rows)
    pred=model.predict(xx,meta);artifact=model.snapshot(xx)
    physical,values=replay_artifact(data,artifact);independent=prediction(data,values,meta)
    np.testing.assert_allclose(pred,independent,rtol=3e-8,atol=3e-8)
    details['physical']=physical;checks['independent_full_history_forward']=True;checks.update(physical['checks'])
    del values;gc.collect()
    # Causality with S and other train preprocessing frozen. Small exact recurrence
    # spans a month/year boundary; future source/water/wetness changes only later.
    nd=70;nr=2;h=np.full((nd,nr),.08);s=np.array([.998,.996]);f0=np.full_like(h,.4);l=np.full_like(h,.02)
    w=np.linspace(.1,.9,nd)[:,None]*np.ones((1,nr));delta=wetting(w);scale=np.array([5.,12.]);c=xx[-6:]
    mid=np.r_[np.zeros(31,int),np.ones(31,int),np.full(8,2,int)];source=np.full((3,nr),20.);crop=np.full((3,nr),3.)
    original=forward(h,s,f0,l,w,delta,scale,c,source,crop,mid)
    hw=h.copy();hw[40:]*=2;ww=w.copy();ww[40:]*=.4;ss=source.copy();ss[2]*=3
    changed=forward(hw,s,f0,l,ww,wetting(ww),scale,c,ss,crop,mid)
    for a,b in zip(original,changed):np.testing.assert_array_equal(a[:40],b[:40])
    for hz,src in [(np.zeros_like(h),source),(h,np.zeros_like(source))]:
        fast,slow,_,_=forward(hz,s,f0,l,w,delta,scale,c,src,crop,mid);assert np.max(abs(fast))+np.max(abs(slow))==0
    checks['future_drivers_no_past_change_fixed_preprocessing']=True;checks['zero_water_zero_source_empty_inventory']=True
    try:model.predict(xx,train)
    except ValueError:checks['prediction_rejects_TN']=True
    else:checks['prediction_rejects_TN']=False
    # Label perturbation changes no worker input bytes: labels are split from
    # metadata, weights, S, initialization and equations before training.
    for fn in ('F23','F24'):
        validation=pd.read_parquet(RUN/f'outputs/data/{fn}_heldout_labels.parquet')
        changed=validation.copy();changed['tn_mg_l']=changed.tn_mg_l*999+7
        pd.testing.assert_frame_equal(prediction_metadata(validation),prediction_metadata(changed))
        assert not set(pd.read_parquet(RUN/f'outputs/data/{fn}_train.parquet').observation_id)&set(validation.observation_id)
    checks['heldout_perturbation_identical_predictor_inputs']=True
    # Hash/counter restoration and live duplication faults require no model runs.
    cp=RUN/'work/tests/checkpoint.pt';ident={'test':1};checkpoint(cp,{'identity':ident,'calls':3999,'active_seconds':123.,'stage':1})
    assert restore(cp,ident)['calls']==3999
    with cp.open('ab') as fcp:fcp.write(b'corruption')
    try:restore(cp,ident)
    except RuntimeError:checks['corrupt_checkpoint_rejected']=True
    else:checks['corrupt_checkpoint_rejected']=False
    with exclusive('validation_lock'):
        try:
            with exclusive('validation_lock'):pass
        except RuntimeError:checks['duplicate_live_process_rejected']=True
        else:checks['duplicate_live_process_rejected']=False
    tr=[{'objective':1.,'projected_gradient':1.} for _ in range(20)]
    assert not continuation(tr)['continue'];tr[-10:]=[{'objective':.99,'projected_gradient':.7}]*10
    assert continuation(tr)['continue'] and not continuation([],True)['continue']
    assert not admission({'ram':{'used_percent':10,'available_gib':90,'total_gib':100},'cpu':{'used_percent':None}},1,[])
    checks['training_trend_and_unknown_CPU_rules']=True
    result={'status':'PASS_MODEL_IMPLEMENTATION' if all(checks.values()) else 'FAILED_IMPLEMENTATION','checks':checks,'details':details,
      'peak':process_snapshot(os.getpid()),'elapsed_seconds':time.time()-started,'heldout_scores_computed':False,
      'remaining_operational_checks':'controller child events and extended budget reviewed before launch'}
    write(RUN/'reports/model_validation.json',result);print(result['status'],flush=True)
    if not all(checks.values()):raise RuntimeError('IMPLEMENTATION_CHECK_FAILED')
if __name__=='__main__':main()
