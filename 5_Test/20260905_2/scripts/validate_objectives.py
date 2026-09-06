"""Check assembled objectives, regionalization and no future N-source influence."""
from fit_models import ROOT, FitObjective, MODELS, LOSSES, load_data
from common import atomic_json, sha256, utc_now, memory_guard
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def main():
    data=load_data('formal')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    train=obs.loc[obs.year.between(2016,2019)&obs.primary_gate].copy()
    results=[]
    for model in MODELS:
        for loss in LOSSES:
            if model=='CONTROL_H7' and loss!='STUDENT_T4_LOG1P':continue
            objective=FitObjective(data,train,model,loss)
            initial=objective.initial(2)
            # Nonzero coefficients check the bounded generator off its global limit.
            for i,name in enumerate(objective.names):
                if name.startswith('gamma_'):initial[i]=.01*np.sin(i)
            theta=torch.tensor(initial,requires_grad=True)
            value=objective.loss(theta);value.backward()
            gradient=theta.grad.detach().numpy()
            names=['log_alpha_contact','beta_contact','v_f','log_tau_mineral_days']
            names += [n for n in objective.names if n in ['log_sigma','gamma_contact_0','gamma_lifetime_0','gamma_head_0','site_0']]
            checks=[]
            for name in names:
                k=objective.indices[name];step=1e-6 if name=='v_f' else 1e-5
                plus=initial.copy();minus=initial.copy();plus[k]+=step;minus[k]-=step
                with torch.no_grad():fd=float((objective.loss(torch.tensor(plus))-objective.loss(torch.tensor(minus)))/(2*step))
                passed=abs(fd-gradient[k])<=1e-6+1e-3*abs(fd)
                checks.append({'parameter':name,'analytic':float(gradient[k]),'finite_difference':fd,'pass':bool(passed)})
            assert all(c['pass'] for c in checks),checks
            # Strongly perturb all source and demand AFTER training; earlier
            # predictions and the complete fitted-objective gradient must agree.
            boundary=(2020-1961)*12
            saved_source=data.source[boundary:].copy();saved_crop=data.crop[boundary:].copy()
            data.source[boundary:]*=17.;data.crop[boundary:]*=.02
            altered=torch.tensor(initial,requires_grad=True)
            altered_value=objective.loss(altered);altered_value.backward()
            data.source[boundary:]=saved_source;data.crop[boundary:]=saved_crop
            np.testing.assert_allclose(float(altered_value.detach()),float(value.detach()),rtol=0,atol=1e-10)
            np.testing.assert_allclose(altered.grad.detach().numpy(),gradient,rtol=0,atol=1e-10)
            results.append({'model':model,'loss':loss,'gradient_checks':checks,'future_source_and_crop_causality':True,
                            'objective':float(value.detach()),'parameters':len(initial)})
            print('OBJECTIVE_VALIDATED',model,loss,flush=True)
            del objective,theta,value,altered,altered_value
    atomic_json({'status':'PASS_ASSEMBLED_OBJECTIVES','created_utc':utc_now(),'results':results,
                 'train_rows':len(train),'memory':memory_guard(),
                 'code_sha256':{str(p):sha256(p) for p in [Path(__file__),ROOT/'5_Test/20260905_2/scripts/fit_models.py']}},
                ROOT/'5_Test/20260905_2/reports/objective_validation.json')


if __name__=='__main__':main()
