"""Branch-recorded refinement of a failed derivative coordinate; never opens gate."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed,sha
configure()
import numpy as np
import d29_training.annual_chain as annual
from d29_platform import precision_candidate as precise
from d29_platform.precision_transfer import transfer_expansion
annual.run_land1=precise.run_land1;annual.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training


def main(coordinate):
    ok,_=dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    scan=json.loads((ROOT/'outputs/land1_full_history_gradient_v3.json').read_text(encoding='utf-8'))
    for path,digest in scan['identity'].items():
        if sha(path)!=digest:raise RuntimeError('SCANNED_IMPLEMENTATION_CHANGED')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T0_s0')
    a=Land1Training(job,diagnostic=True);x=a.initial.copy();g=scan['analytic_gradient'][coordinate]
    native=annual.AnnualChain._run;baseline={};mode='baseline';changes=[]
    def inspect(chain,year,state,eta,correction):
        result,mask=native(chain,year,state,eta,correction);i=result._inputs;nt,nr,nl=i['shape']
        before=result.states[:-1].reshape(nt,nr*nl,5).copy()
        for t in i['transition_days']:
            low=np.zeros_like(before[t]);low[:,1:]=-i['compensation_history'][t]
            moved,_,_=transfer_expansion(before[t].reshape(nr,nl,5),low.reshape(nr,nl,5),i['matrices'][i['event'][t]])
            before[t]=moved.reshape(nr*nl,5)
        P=before[:,:,0]
        X=before[:,:,3]+i['sources'][:,:,3]+i['probabilities'][0]*before[:,:,1]+i['probabilities'][1]*before[:,:,2]
        O=i['outflows'][:,:,0]+i['outflows'][:,:,1]+i['outflows'][:,:,2]
        raw=i['target']+O-P-i['sources'][:,:,0];need=np.maximum(raw,0)
        after=np.where((raw>0)&(X>=need),i['target'],(P+i['sources'][:,:,0]-O)+np.minimum(X,need))
        branch=(raw>0).astype(np.uint8)+2*(X<=need).astype(np.uint8)+4*(after<0).astype(np.uint8)
        if mode=='baseline':baseline[year]=branch
        else:
            positions=np.argwhere(branch!=baseline[year]);witnesses=[]
            for t,u in positions[:20]:
                witnesses.append(dict(date=str(np.datetime64(f'{year}-01-01')+np.timedelta64(int(t),'D')),
                    reach=int(u//nl+1),land=int(u%nl),base_branch=int(baseline[year][t,u]),new_branch=int(branch[t,u]),
                    supply_minus_need=float(X[t,u]-need[t,u]),raw_need=float(raw[t,u])))
            if len(positions):changes.append(dict(year=year,count=len(positions),witnesses=witnesses))
        return result,mask
    annual.AnnualChain._run=inspect
    out=ROOT/f'outputs/land1_coordinate_{coordinate}_branch_refinement.json'
    started=time.monotonic()
    try:
        base,_=a.value_gradient(x,forward_only=True)
        if abs(base-scan['objective'])>1e-8*(1+abs(base)):raise RuntimeError('DIAGNOSTIC_BASE_OBJECTIVE_CHANGED')
        receipt=dict(coordinate=coordinate,name=a.names[coordinate],identity=scan['identity'],base_value=base,analytic=g,
            original_checks=scan['coordinates'][coordinate]['checks'],steps=[1e-6,3e-7,1e-7],checks=[],
            role='additional branch diagnostic; original scan failure preserved; no automatic acceptance override',NSE='not applicable')
        mode='perturbed'
        for step in receipt['steps']:
            values=[];row=dict(step=step)
            for sign in [1,-1]:
                changes.clear();point=x.copy();point[coordinate]+=sign*step
                value,_=a.value_gradient(point,forward_only=True);values.append(value)
                row['plus' if sign>0 else 'minus']=dict(value=value,branch_switch_count=sum(r['count'] for r in changes),switches=list(changes))
                write_json(out.with_name(out.stem+'_pending.json'),row)
            fd=(values[0]-values[1])/(2*step);tol=1e-6*(1+abs(g))
            row.update(fd=fd,error=abs(fd-g),tolerance=tol,passed=bool(abs(fd-g)<=tol))
            receipt['checks'].append(row);receipt['calls']=a.calls;receipt['elapsed_seconds']=time.monotonic()-started
            write_json(out,receipt);print(coordinate,step,row['error'],row['passed'],flush=True)
    finally:annual.AnnualChain._run=native


if __name__=='__main__':main(int(sys.argv[1]))
