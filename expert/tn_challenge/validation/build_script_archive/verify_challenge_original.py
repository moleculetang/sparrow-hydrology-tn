import os,sys,json,hashlib,copy,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('E:/SPARROW');PKG=BASE/'repository/expert/tn_challenge'
os.environ['NUMBA_CACHE_DIR']=str(BASE/'work/numba')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT/'5_Test/20260911_1/scripts'))
import torch,numpy as np,pandas as pd
from sc_model import SharedBox,data_cache
from fc_io import require_environment
require_environment();torch.set_num_threads(1)
sys.path.insert(0,str(PKG));from model import load_data,Objective,Predictor,projected_gradient
d=load_data();train=pd.read_csv(PKG/'data/train.csv');out=PKG/'validation';out.mkdir(exist_ok=True)
report={'tests':{},'reference':'Original full-domain F23 saved models; portable subset with identical full-fold preprocessing and parameters.'}
full=data_cache('sensitivity');fulltrain=pd.read_parquet(ROOT/'5_Test/20260911_1/outputs/data/F23_train.parquet')
rr=np.array(d.global_reach_ids)-1;reference_dir=PKG/'reference_only';reference_dir.mkdir(exist_ok=True)
for variant in ['M0','SC']:
    saved=torch.load(ROOT/f'5_Test/20260911_1/outputs/models/F23_{variant}_0.pt',weights_only=False,map_location='cpu')
    print(variant,'saved keys',list(saved),flush=True)
    design=saved['design'];theta=np.asarray(saved['theta'],float)
    pdsg={k:design[k] for k in ['low','high','mean','sd','dynamic_scales']}
    if variant=='SC':pdsg['inventory_scale_kg']=np.asarray(design['inventory_scale_kg'])[rr].tolist()
    m=Predictor(d,pdsg,variant)
    original=SharedBox(full,fulltrain,variant,scale=np.asarray(design['inventory_scale_kg']) if variant=='SC' else None)
    sample=pd.read_csv(PKG/'data/observations_all.csv');sample=sample.loc[sample.primary_gate].reset_index(drop=True)
    meta=sample.drop(columns=['tn_mg_l']);originalmeta=meta.copy();originalmeta['reach_id']=originalmeta.global_reach_id
    originalmeta['reservoir_index']=originalmeta.global_reservoir_index
    p=m.predict(theta,meta);q=original.predict(theta,originalmeta)
    err=float(np.max(np.abs(p-q)));np.testing.assert_allclose(p,q,rtol=2e-10,atol=2e-10)
    report['tests'][variant+'_full_domain_prediction_parity_max_mg_l']=err
    sample['prediction_mg_l']=p;sample.to_csv(reference_dir/f'F23_{variant}_0_predictions.csv',index=False)
    (reference_dir/f'F23_{variant}_0.json').write_text(json.dumps({'variant':variant,'theta':theta.tolist(),'design':pdsg,
        'training_ids':saved.get('training_ids',fulltrain.observation_id.tolist()),'purpose':'Diagnostic full-domain reference only; never default sample training initialization.'},ensure_ascii=False,indent=2),encoding='utf-8')
    del original
# Fresh, sample-only training transforms, weights, priors and starts.
m0=Objective(d,train,'M0');sc=Objective(d,train,'SC');x=sc.initial(0);x[0]=-4.;x[3]=np.log(700);x[-6:]=np.array([1,-1,1,-1,1,-1])*.01
x[4:18]=np.linspace(-.01,.01,14);x[18:21]=[.01,-.02,.03]
J,g=sc.value_gradient(x);errors=[]
for i in range(len(x)):
    h=2e-5*max(1.,abs(x[i]));u=x.copy();v=x.copy();u[i]+=h;v[i]-=h
    a=sc.value_gradient(u)[0];b=sc.value_gradient(v)[0];fd=(a-b)/(2*h);err=abs(fd-g[i])/max(1,abs(fd),abs(g[i]));errors.append(err)
assert max(errors)<2e-5,errors
report['tests']['all_27_parameter_full_history_gradient_max_relative_error']=max(errors)
x0=m0.initial(0);p=m0.predict(x0,m0.meta);q=sc.predict(np.r_[x0,np.zeros(6)],sc.meta);np.testing.assert_allclose(p,q,rtol=1e-13,atol=1e-13)
report['tests']['zero_extension_max_mg_l']=float(np.max(np.abs(p-q)))
ledger=sc.ledger(x);report['tests']['local_mass_balance_max_kg']=ledger['local_balance_max_kg'];report['tests']['network_mass_balance_kg']=ledger['network_balance_kg']
assert ledger['local_balance_max_kg']<1e-5 and abs(ledger['network_balance_kg'])<1e-3
assert ledger['M'].min()>=0 and ledger['L'].min()>-1e-5 and np.max(ledger['uptake']-ledger['demand'])<=0
report['tests']['stock_and_uptake_boundaries']='PASS'
# Original RAW loss agrees at the same sample geometry / training preprocessing.
otrain=train.copy();otrain['reach_id']=otrain.global_reach_id;otrain['reservoir_index']=otrain.global_reservoir_index
original=SharedBox(full,otrain,'M0');oldJ,oldg=original.value_gradient(x0);newJ,newg=m0.value_gradient(x0)
np.testing.assert_allclose(newJ,oldJ,rtol=1e-11);np.testing.assert_allclose(newg,oldg,rtol=1e-9,atol=1e-8)
report['tests']['original_sample_RAW_MAP_value_abs_error']=abs(newJ-oldJ);report['tests']['original_sample_gradient_max_error']=float(np.max(abs(newg-oldg)))
splits={name:pd.read_csv(PKG/f'data/{name}.csv') for name in ['train','development','evaluation','hindcast']}
assert all(not set(a.observation_id)&set(b.observation_id) for i,a in enumerate(splits.values()) for b in list(splits.values())[i+1:])
report['tests']['four_split_ID_disjoint']='PASS'
# Predictor state contains frozen preprocessing, never reconstruct from evaluation years.
dc=copy.deepcopy(d);dc.source[d.months.year==2024]*=1.37;dc.percolation[d.dates.year==2024]*=1.2;dc.soil_wetness[d.dates.year==2024]*=.9
future=Predictor(dc,sc.design,'SC');np.testing.assert_allclose(future.predict(x,sc.meta),sc.predict(x,sc.meta),rtol=0,atol=0)
report['tests']['future_driver_past_prediction_invariance']='PASS'
report['status']='PASS';report['versions']={k:__import__(k).__version__ for k in ['numpy','pandas','scipy','numba','torch']}
(out/'original_parity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
