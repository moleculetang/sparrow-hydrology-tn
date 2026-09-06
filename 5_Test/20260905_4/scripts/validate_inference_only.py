"""Verify station-history-free process decoding against the fitted generator."""
from extended_objective import ROOT,ExtendedObjective
from inference_only import reconstruct_process
from fit_models import load_data
from common import RUNTIME,atomic_json,sha256,utc_now,memory_guard
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path


def main():
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet');checks=[]
    paths=[ROOT/'5_Test/20260905_4/reports'/f'{name}_t2020_s0.json' for name in ['uniform_daily','calendar_early','calendar_late','contact_dynamic']]
    for path in paths:
        report=json.loads(path.read_text(encoding='utf-8'));spec=report['identity']['spec']
        data=load_data('formal',spec['calendar']);train=obs.loc[obs.observation_id.isin(spec['train_ids'])].copy()
        o=ExtendedObjective(data,train,spec['model'],spec['loss'],spec.get('prior_scale',1),spec['timing'],spec['dynamic'])
        theta=torch.tensor([report['parameters'][n] for n in o.names])
        with torch.no_grad():p,s,a,tau=o.process(theta)
        reader=pd.read_parquet;read_paths=[]
        def forbid_observations(source,*args,**kwargs):
            name=str(source);read_paths.append(name)
            if Path(source).name!='h7_raw_features.parquet':raise AssertionError(f'Unnecessary inference read: {name}')
            return reader(source,*args,**kwargs)
        # Discard all training IDs and other observation provenance before
        # decoding. Only parameter values and saved environmental scalers remain.
        stripped={'identity':{'spec':{k:spec[k] for k in ['model','timing','dynamic']},
            'prepared_features_sha256':report['identity']['prepared_features_sha256']},
            'parameters':report['parameters'],'design':{k:v for k,v in report['design'].items() if k in ['fields','low','high','mean','sd','dynamic_scales']}}
        try:
            pd.read_parquet=forbid_observations
            got=reconstruct_process(data,stripped)
        finally:pd.read_parquet=reader
        errors={}
        for key,expected in [('probability',p.numpy()),('survival',s.numpy()),('log_alpha',a.numpy()),('log_tau',tau.numpy())]:
            np.testing.assert_allclose(got[key],expected,rtol=1e-11,atol=1e-13)
            errors[key]=float(np.max(np.abs(got[key]-expected)))
        checks.append(dict(report_path=str(path),report_sha256=sha256(path),max_abs_errors=errors,inference_reads=read_paths,all_TN_history_removed=True))
        print('INFERENCE_ONLY_VALIDATED',spec['candidate_id'],errors,flush=True)
        del o,data,p,s,a,tau,got
    script=Path(__file__).with_name('inference_only.py')
    atomic_json(dict(status='PASS_INFERENCE_WITHOUT_TN_HISTORY',runtime=RUNTIME,created_utc=utc_now(),checks=checks,
        code_sha256={str(p):sha256(p) for p in [Path(__file__),script]},memory=memory_guard()),ROOT/'5_Test/20260905_4/reports/inference_only_validation.json')


if __name__=='__main__':main()
