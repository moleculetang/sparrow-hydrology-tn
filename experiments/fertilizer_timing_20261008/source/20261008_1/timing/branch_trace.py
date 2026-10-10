"""Register the exact supply/demand/capacity and realized-output branches."""
from contextlib import contextmanager
import numpy as np
from .chain import TimingChain
from d29_platform.precision_transfer import transfer_expansion

@contextmanager
def trace():
    records={};native=TimingChain._run
    def wrapped(chain,year,initial,eta,corr):
        result,mask=native(chain,year,initial,eta,corr);i=result._inputs;nt,nr,nl=i['shape']
        before=result.states[:-1].copy()
        for day in i['transition_days']:
            low=np.zeros_like(before[day]);low[...,1:]=-i['compensation_history'][day].reshape(nr,nl,4)
            before[day],_,_=transfer_expansion(before[day],low,i['matrices'][i['event'][day]])
        b=before.reshape(nt,nr*nl,5);pa,pp,*_=i['probabilities']
        O=i['outflows'][...,0]+i['outflows'][...,1]+i['outflows'][...,2]
        raw=i['target']+O-b[...,0]-i['sources'][...,0]
        X=b[...,3]+i['sources'][...,3]+pa*b[...,1]+pp*b[...,2];need=np.maximum(0.,raw);cap=i['plant_growth_capacity']
        U=np.minimum(np.minimum(X,need),cap)
        after=np.where((raw>0)&(X>=need)&(cap>=need),i['target'],(b[...,0]+i['sources'][...,0]-O)+U)
        supply=(X<=need)&(X<=cap);demand=(~supply)&(need<=cap)
        bits=(raw>0).astype('uint8')+2*supply.astype('uint8')+4*demand.astype('uint8')+8*(after<0).astype('uint8')
        records[year]=bits.reshape(nt,nr,nl)
        return result,mask
    TimingChain._run=wrapped
    try:yield records
    finally:TimingChain._run=native

def compare(reference,candidate):
    changes=[];count=0
    for year,b in reference.items():
        if year not in candidate:raise ValueError('MISSING_BRANCH_YEAR')
        ids=np.argwhere(b!=candidate[year]);count+=len(ids)
        if len(ids):changes.append(dict(year=year,count=len(ids),examples=[dict(day=int(t),reach=int(r+1),land=int(l),before=int(b[t,r,l]),after=int(candidate[year][t,r,l])) for t,r,l in ids[:6]]))
    return dict(consistent=count==0,changed_cells=count,changes=changes)
