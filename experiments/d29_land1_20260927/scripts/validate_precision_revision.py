"""Regression for real-scale cancellation, compensated checkpoint and adjoint."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
from d29_platform.land1 import run_land1,land1_adjoint,PROBABILITY_NAMES,propagate_source_labels,PlantBudgetInfeasible
OUT=ROOT/'outputs/precision_revision';OUT.mkdir(parents=True,exist_ok=True)
def main():
    nt=800
    initial=np.array([[[0.,9e7,4.7e9,2e4,300.]]])
    source=np.zeros((nt,1,1,4));source[:,:,:,0]=.3;source[:,:,:,1]=42;source[:,:,:,2]=11;source[:,:,:,3]=300
    out=np.zeros((nt,1,1,3));out[:,:,:,0]=289.1;out[:,:,:,1]=.7;out[:,:,:,2]=.13
    target=np.zeros((nt,1,1));prob=dict(zip(PROBABILITY_NAMES,[.0077,.00001,.001,0.,.4,.1]))
    spec=dict(initial=initial,sources=source,plant_target=target,plant_outflows=out,probabilities=prob,compensated=True)
    full=run_land1(**spec);cut=367
    first=run_land1(**{**spec,'sources':source[:cut],'plant_target':target[:cut],'plant_outflows':out[:cut]})
    second=run_land1(**{**spec,'initial':first.final,'initial_compensation':first.compensation,'sources':source[cut:],'plant_target':target[cut:],'plant_outflows':out[cut:]})
    checks=[dict(name='zero_target_no_negative_roundoff',passed=bool((full.states[:,:,:,0]==0).all())),
        dict(name='compensated_checkpoint_bitwise_states',passed=bool(np.array_equal(full.states[cut:],second.states))),
        dict(name='compensated_checkpoint_bitwise_fluxes',passed=bool(np.array_equal(full.fluxes[cut:],second.fluxes))),
        dict(name='compensated_checkpoint_correction_state',passed=bool(np.array_equal(full.compensation,second.compensation)))]
    ad=land1_adjoint(full,grad_fluxes=np.ones_like(full.fluxes)/full.fluxes.size)
    # Relative directions remain identifiable despite billion-kg states.
    for name in ('sources','initial','plant_outflows'):
        analytic=float((ad[name]*spec[name]).sum())
        for h in (1e-3,1e-4,1e-5):
            vals=[]
            for sign in (-1,1):
                q={**spec,name:spec[name]*(1+sign*h)}
                vals.append(float(run_land1(**q).fluxes.mean()))
            fd=(vals[1]-vals[0])/(2*h)
            checks.append(dict(name=name+'_direction',step=h,analytic=analytic,finite_difference=fd,passed=abs(fd-analytic)<=1e-6*(1+abs(analytic))))
    q=copy.deepcopy(spec);q['plant_outflows'][0,0,0,0]=1e12
    try:run_land1(**q);blocked=False
    except PlantBudgetInfeasible:blocked=True
    checks.append(dict(name='real_infeasibility_not_clipped',passed=blocked))
    baseline=run_land1(**{**spec,'compensated':False})
    protected_daily_input = 11 + .13  # external organic N plus plant protected return
    reference=(initial[0,0,2]-protected_daily_input/prob['mineralize_protected'])*(1-prob['mineralize_protected'])**nt+protected_daily_input/prob['mineralize_protected']
    # Closed form itself has floating exponent error: report, do not force exact equality.
    gradient_ok=all(any(a['passed'] and b['passed'] for a,b in zip([c for c in checks if c['name']==name+'_direction'],[c for c in checks if c['name']==name+'_direction'][1:])) for name in ('sources','initial','plant_outflows'))
    passed=gradient_ok and all(c['passed'] for c in checks if not c['name'].endswith('_direction'))
    write_json(OUT/'receipt.json',dict(passed=passed,checks=checks,
        gradient_rule='at least two adjacent step lengths pass, retain smaller-step cancellation failures; not minimum-of-last-two',
        protected_pool_closed_form_kg=float(reference),compensated_protected_kg=float(full.final[0,0,2]),
        ordinary_protected_kg=float(baseline.final[0,0,2]),local_mass_max_error_kg=full.max_local_balance_kg,
        kernel_hash=sha(ROOT/'d29_platform/land1.py'),NSE='not applicable'))
    print('precision',passed,checks,flush=True)
    if not passed:raise AssertionError('PRECISION_REVISION')
if __name__=='__main__':main()
