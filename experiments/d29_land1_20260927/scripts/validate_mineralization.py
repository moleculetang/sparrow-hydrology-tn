from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json
configure()
import numpy as np,pandas as pd
from d29_platform.mineralization import first_order_probability,land1_mineralization_fields,mineralization_pullback
from d29_platform.land1 import run_land1,land1_adjoint
OUT=ROOT/'outputs/soil_temperature';OUT.mkdir(parents=True,exist_ok=True)
checks=[]
def check(name,ok,**evidence):checks.append(dict(name=name,passed=bool(ok),**evidence))
def reject(name,fn):
    try:fn();check(name,False)
    except ValueError:check(name,True)

def main():
    dates=pd.date_range('1961-01-01','2024-12-31');T=len(dates)
    config=json.loads((ROOT/'config/mineralization_reference.json').read_text(encoding='utf-8'))
    ka=config['k_active_per_day'];kp=1/(config['protected_turnover_years']*config['days_per_reference_year'])
    fixed,fd=land1_mineralization_fields(ka,kp)
    check('small_rate_expm1',first_order_probability(1e-16)[0]>0)
    check('zero_rate_known_zero',first_order_probability(0)[0]==0)
    check('high_rate_no_negative_stock',0<=first_order_probability(1000)[0]<=1)
    reject('missing_temperature_rejected',lambda:first_order_probability(ka,mode='soil_temperature_q10',soil_temperature_c=np.nan,q10=1.5,reference_temperature_c=25))
    reject('fixed_does_not_silently_ignore_temperature',lambda:first_order_probability(ka,soil_temperature_c=20))
    source=np.zeros((T,1,1,4));source[0,0,0,1]=10
    init=np.array([[[0.,100.,200.,0.,0.]]])
    shared=dict(mobilize=.01,available_loss=.002,fast_fraction=.3,lower_release=.02)
    def run(a=ka,p=kp,temp=None,q10=1.5):
        kw={} if temp is None else dict(mode='soil_temperature_q10',soil_temperature_c=temp,q10=q10,reference_temperature_c=25.)
        fields,der=land1_mineralization_fields(a,p,**kw)
        r=run_land1(initial=init,sources=source,plant_target=0.,plant_outflows=0.,probabilities={**shared,**fields})
        return r,der
    b,der=run();expected=(100*np.exp(-ka*T)+10*np.exp(-ka*(T-1)))
    check('full_1961_2024_exact_active_stock',abs(b.final[0,0,1]-expected)<1e-10,error=abs(b.final[0,0,1]-expected))
    check('new_organic_input_not_mineralized_same_day',abs(b.states[1,0,0,1]-(100*np.exp(-ka)+10))<1e-12)
    check('full_history_mass_closed',b.max_local_balance_kg<=1e-6,error_kg=b.max_local_balance_kg)
    weight=np.sin(np.arange(T)*.013)[:,None,None,None]*np.array([1.,.7,0.,0.])[None,None,None,:]/T
    temp=(20+8*np.sin(np.arange(T)*2*np.pi/365.25))[:,None,None]
    gradchecks=[]
    for mode in ('fixed','soil_temperature_q10'):
        tv=None if mode=='fixed' else temp
        r,der=run(temp=tv);ad=land1_adjoint(r,grad_fluxes=weight)
        g=mineralization_pullback(ad['probabilities'],der)
        for name in (['active','protected'] if tv is None else ['active','protected','temperature_offset','q10']):
            if name=='active':analytic=g['k_active_per_day']*ka
            elif name=='protected':analytic=g['k_protected_per_day']*kp
            elif name=='temperature_offset':analytic=float(g['soil_temperature_c'].sum())
            else:analytic=g['q10']
            for h in (1e-3,1e-4):
                v=[]
                for sign in (-1,1):
                    a=ka*(1+sign*h) if name=='active' else ka
                    p=kp*(1+sign*h) if name=='protected' else kp
                    t=tv+sign*h if name=='temperature_offset' else tv
                    q=1.5+sign*h if name=='q10' else 1.5
                    rr,_=run(a,p,t,q);v.append(float(np.sum(rr.fluxes*weight)))
                numeric=(v[1]-v[0])/(2*h);err=abs(numeric-analytic)
                gradchecks.append(dict(mode=mode,direction=name,step=h,analytic=analytic,finite_difference=numeric,absolute_error=err,passed=err<=1e-6*(1+abs(analytic))))
    check('two_adjacent_steps_all_directions',len(gradchecks)==12 and all(x['passed'] for x in gradchecks))
    # Future explicit soil-T changes cannot affect earlier states.
    r,_=run(temp=temp);changed=temp.copy();changed[-30:]+=5.;rr,_=run(temp=changed)
    check('future_temperature_causality',np.array_equal(r.states[:-30],rr.states[:-30]))
    p25,_=first_order_probability(ka,mode='soil_temperature_q10',soil_temperature_c=25.,q10=1.5,reference_temperature_c=25.)
    check('reference_temperature_embeds_fixed',p25==fixed['mineralize_active'])
    receipt=dict(passed=all(x['passed'] for x in checks),checks=checks,gradients=gradchecks,days=T,
        data_identity='synthetic kernel verification, not measured soil T or water-quality performance',NSE='not applicable')
    write_json(OUT/'mineralization_validation.json',receipt)
    print(json.dumps({'passed':receipt['passed'],'checks':len(checks),'gradient_checks':len(gradchecks),'days':T}),flush=True)
    if not receipt['passed']:raise AssertionError('MINERALIZATION_ACCEPTANCE')
if __name__=='__main__':main()
