"""Locate real-scale roundoff without changing equations or accepting failed gates."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json
configure()
import numpy as np
from d29_platform.legacy import build_legacy
from d29_platform.conditional_inputs import ConditionalInputs
from d29_platform.coupling import response_mapping
from d29_platform.land1 import run_land1,hazard_to_probability

m,x,_=build_legacy('F23','U',True);inputs=ConditionalInputs(m.data.dates)
retired={'log_tau_mineral_days','log_source_correction_0'}|{f'gamma_lifetime_{i}' for i in range(7)}
h,f=response_mapping(m,{n:float(v) for n,v in zip(m.names,x) if n not in retired});pm,_=hazard_to_probability(h)
mini=json.loads((ROOT/'config/mineralization_reference.json').read_text(encoding='utf-8'))
ka=-np.expm1(-mini['k_active_per_day']);kp=-np.expm1(-1/(270*365.25))
history=np.load(ROOT.parent/'20260927_1/outputs/supply_limited_reference/annual_end_stocks.npy',mmap_mode='r')
results=[]
for year in [1961,1962,2024]:
    initial=inputs.initial() if year==1961 else np.array(history[year-1962])
    ix=np.flatnonzero(m.data.dates.year==year);post=initial[:,:,0].copy()
    if int(ix[0]) in inputs.transitions:post[:,:12]=np.einsum('ri,rij->rj',post[:,:12],inputs.transitions[int(ix[0])])
    a=inputs.annual(year,post);active=np.ones((1,1,13));active[:,:,4]=0;active[:,:,12]=0
    probs=dict(mineralize_active=ka*active,mineralize_protected=kp*active,mobilize=pm[ix,:,None]*active,available_loss=inputs.config['available_loss_probability']*active,fast_fraction=f[ix,:,None],lower_release=m.data.lower_release[ix,:,None]*active)
    b=run_land1(initial=initial,probabilities=probs,compensated=True,plant_activity_mode='potential_with_shortfall',**{k:a[k] for k in ['sources','plant_target','plant_outflows','transitions']})
    before=b.states[:-1].copy();transition=[]
    for day,mat in a['transitions'].items():
        moved=np.einsum('rik,rij->rjk',before[day],mat)
        for r in range(230):
            transition.append(dict(day=int(day),reach_id=r+1,ordinary_sum_error=float(moved[r].sum()-before[day,r].sum()),accurate_sum_error=math.fsum([*moved[r].ravel(),*(-before[day,r]).ravel()])))
        before[day]=moved
    diff=b.states[1:]-before
    residual=diff.sum(-1)-a['sources'].sum(-1)+b.fluxes.sum(-1)
    flat=np.argsort(np.abs(residual).ravel())[-20:][::-1];witness=[]
    for k in flat:
        t,r,l=np.unravel_index(k,residual.shape)
        exact=math.fsum([*b.states[t+1,r,l],*(-before[t,r,l]),*(-a['sources'][t,r,l]),*b.fluxes[t,r,l]])
        witness.append(dict(date=str(m.data.dates[ix[t]].date()),reach_id=int(r+1),land=int(l),ordinary_residual=float(residual[t,r,l]),accurate_sum_residual=exact,largest_state=float(before[t,r,l].max()),state_spacing=float(np.spacing(before[t,r,l].max()))))
    results.append(dict(year=year,checkpoint_role='diagnostic frozen annual state; compensation reset explicitly, not a replay acceptance',kernel_max=b.max_local_balance_kg,witnesses=witness,transition_worst=sorted(transition,key=lambda d:abs(d['ordinary_sum_error']),reverse=True)[:10],formal_gate_pass=False))
    write_json(ROOT/'outputs/land1_precision_diagnosis.json',results);print(year,b.max_local_balance_kg,flush=True)
