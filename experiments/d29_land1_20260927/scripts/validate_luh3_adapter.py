"""Real LUH3 areas; fictional tracer. Event-compressed transport-only check."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd
from d29_platform.luh3_adapter import load_area_transfers,require_nitrogen_transfer_policy,LAND_STATES
from d29_platform.land1 import run_land1,land1_adjoint,PROBABILITY_NAMES

def main():
    src=Path('E:/SPARROW/5_Test/20260926_1/outputs/luh3_reach')
    paths=[src/'luh3_reach_state_1961_2024.csv',src/'luh3_land_area_transfer_operator_1961_2023.csv']
    dates=pd.DatetimeIndex([pd.Timestamp(y,1,1) for y in range(1961,2025)])
    area,tr=load_area_transfers(*paths,dates)
    yearly=[]
    for t,m in tr.items():
        pred=np.einsum('ri,rij->rj',area[t-1],m)
        err=np.abs(pred-area[t]).sum(1)/area[t].sum(1)
        yearly.append(dict(from_year=int(dates[t].year-1),to_year=int(dates[t].year),max_land_relative_area_error=float(err.max())))
    initial=area[0,:,:,None]*np.array([1.,2.,3.,4.,5.])[None,None,:]*1e-6
    spec=dict(initial=initial,sources=np.zeros((64,230,12,4)),plant_target=0.,plant_outflows=0.,
        probabilities={p:0. for p in PROBABILITY_NAMES},transitions=tr)
    b=run_land1(**spec)
    ref=initial.copy()
    for t in range(64):
        if t in tr:
            new=np.zeros_like(ref)
            for r in range(230):
                for s in range(5):new[r,:,s]=ref[r,:,s] @ tr[t][r]
            ref=new
        if not np.allclose(ref,b.states[t+1],rtol=0,atol=1e-8):raise AssertionError('INDEPENDENT_TRANSFER_DISAGREEMENT')
    rng=np.random.default_rng(1729);weight=rng.normal(size=initial.shape)
    ad=land1_adjoint(b,grad_final=weight)
    checks=[]
    for h in (1e-4,1e-5):
        a=float(np.sum(ad['initial']*initial));vals=[]
        for sign in (-1,1):vals.append(float(np.sum(run_land1(**{**spec,'initial':initial*(1+sign*h)},keep_history=False).final*weight)))
        fd=(vals[1]-vals[0])/(2*h);checks.append(dict(direction='initial_scale',h=h,analytic=a,fd=fd,pass_check=abs(a-fd)<=1e-6*(1+abs(a))))
    # End-year transition timing and future causality, with a neutral replacement.
    last=max(tr);tr2={**tr,last:np.broadcast_to(np.eye(12),(230,12,12)).copy()}
    c=run_land1(**{**spec,'transitions':tr2})
    causal_error=float(np.max(np.abs(b.states[:last+1]-c.states[:last+1])))
    rejected=False
    try:require_nitrogen_transfer_policy(None)
    except ValueError:rejected=True
    require_nitrogen_transfer_policy(dict(identity='conservative_relocation_scenario',no_claim_of_observed_nitrogen_fates=True))
    result=dict(source_sha256={str(p):sha(p) for p in paths},states=list(LAND_STATES),reaches=230,
        event_dates='January 1 of transition field year + 1',last_field_year=2023,last_state_year=2024,
        simulation='64 annual event records, all biogeochemical probabilities and fluxes zero; not daily biological run',
        nitrogen_initial='fictional independent tracer in each of five state labels; not scientific initialization',
        yearly_area_checks=yearly,max_local_mass_error_kg=b.max_local_balance_kg,
        independent_final_max_error_kg=float(np.max(np.abs(ref-b.final))),future_causality_error=causal_error,
        gradient_checks=checks,nitrogen_fate_policy_missing_rejected=rejected,
        NSE='not applicable to transfer and gradient checks')
    result['passed']=bool(b.max_local_balance_kg<=1e-6 and causal_error==0 and rejected and all(x['pass_check'] for x in checks)
        and max(x['max_land_relative_area_error'] for x in yearly)<=1e-6)
    out=ROOT/'outputs/luh3_adapter';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'transition_matrices.npz',**{str(k):v for k,v in tr.items()})
    write_json(out/'receipt.json',result)
    print('LUH3_ADAPTER',result['passed'],b.max_local_balance_kg,flush=True)
    if not result['passed']:raise AssertionError('LUH3_ADAPTER_FAILED')
if __name__=='__main__':main()
