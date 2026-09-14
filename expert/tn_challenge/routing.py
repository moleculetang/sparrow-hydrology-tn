"""Extracted original routing and adjoints; only reach dimension is generalized."""
import numpy as np
import torch
from numba import njit

@njit(cache=False)
def reservoir_scan(captured,release_fraction):
    n=len(captured)
    releases=np.empty(n);stocks=np.empty(n)
    storage=0.0
    for d in range(n):
        pre=storage+captured[d]
        releases[d]=pre*release_fraction[d]
        storage=pre-releases[d]
        stocks[d]=storage
    return releases,stocks


def route(data,local,vf=0.021952884993820768,exposure='monthly',water_replay=False):
    """Topological time-series routing. Reservoir scans are causal and conservative."""
    nd,nr=local.shape
    inlet=np.zeros_like(local);preout=np.zeros_like(local);official=np.zeros_like(local)
    removed=np.zeros_like(local)
    nrsv=len(data.metadata)
    captures=np.zeros((nd,nrsv));releases=np.zeros_like(captures);stocks=np.zeros_like(captures)
    by_control={c:i for i,r in enumerate(data.metadata) for c in r['controls']}
    seen=[set() for _ in data.metadata]
    for r in data.order:
        h=data.h_month[data.mid,r] if exposure=='monthly' else data.h_day[:,r]
        attenuation=np.exp(-vf*h)
        local_delivered=local[:,r]*np.exp(-0.5*vf*h)
        preout[:,r]=inlet[:,r]*attenuation+local_delivered
        official[:,r]=preout[:,r]
        removed[:,r]=inlet[:,r]+local[:,r]-preout[:,r]
        if r not in by_control:
            if r in data.downstream:inlet[:,data.downstream[r]]+=preout[:,r]
            continue
        k=by_control[r];meta=data.metadata[k]
        if len(meta['controls'])>1:
            captures[:,k]+=preout[:,r]
            seen[k].add(r)
            if seen[k]!=set(meta['controls']):continue
            bypass=np.zeros(nd)
        else:
            fraction=np.where(data.enabled[:,k],meta['fraction'],1.0)
            bypass=(1-fraction)*local_delivered
            captures[:,k]=preout[:,r]-bypass
        if water_replay:
            releases[:,k]=data.released_water[:,k]
        else:
            releases[:,k],stocks[:,k]=reservoir_scan(captures[:,k],data.release_fraction[:,k])
        inlet[:,meta['target']]+=releases[:,k]+bypass
        # This is precisely the mixed boundary convention of the hydro export.
        if len(meta['controls'])==1:
            official[:,r]=releases[:,k]+bypass
    return {'inlet':inlet,'preout':preout,'official':official,'channel_removed':removed,
            'captures':captures,'releases':releases,'stocks':stocks,
            'terminal':official[:,data.terminal].sum(axis=1)}


@njit(cache=False)
def reservoir_backward(grad_release,fraction):
    n=len(grad_release)
    grad_capture=np.empty(n)
    state=0.
    for d in range(n-1,-1,-1):
        state=grad_release[d]*fraction[d]+state*(1-fraction[d])
        grad_capture[d]=state
    return grad_capture


class RiverN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,local,vf,data,exposure,include_release=False):
        v=float(vf.detach())
        values=np.ascontiguousarray(local.detach().numpy())
        result=route(data,values,vf=v,exposure=exposure)
        ctx.data=data;ctx.exposure=exposure;ctx.inlet=result['inlet'];ctx.vf=v
        ctx.save_for_backward(local)
        ctx.nargs=len(ctx.needs_input_grad)
        outputs=(torch.from_numpy(data.monthly_sum(result['inlet'])),torch.from_numpy(data.monthly_sum(result['official'])))
        return outputs+(torch.from_numpy(data.monthly_sum(result['releases'])),) if include_release else outputs

    @staticmethod
    def backward(ctx,g_inlet_month,g_official_month,g_release_month=None):
        data=ctx.data
        local=ctx.saved_tensors[0].detach().numpy()
        shape=(len(data.months),data.source.shape[1])
        im=np.zeros(shape) if g_inlet_month is None else g_inlet_month.detach().numpy()
        om=np.zeros(shape) if g_official_month is None else g_official_month.detach().numpy()
        rm=np.zeros((len(data.months),len(data.metadata))) if g_release_month is None else g_release_month.detach().numpy()
        gr=rm[data.mid]
        gi=im[data.mid].copy();go=om[data.mid]
        gl=np.zeros_like(local);gv=0.
        by_control={c:k for k,m in enumerate(data.metadata) for c in m['controls']}
        capture_adjoints={}
        for r in reversed(data.order):
            h=data.h_month[data.mid,r] if ctx.exposure=='monthly' else data.h_day[:,r]
            attenuation=np.exp(-ctx.vf*h);half=np.exp(-.5*ctx.vf*h)
            if r not in by_control:
                gp=go[:,r].copy()
                if r in data.downstream:gp+=gi[:,data.downstream[r]]
                gdel=gp
            else:
                k=by_control[r];meta=data.metadata[k]
                if len(meta['controls'])>1:
                    if k not in capture_adjoints:
                        capture_adjoints[k]=reservoir_backward(np.ascontiguousarray(gi[:,meta['target']]+gr[:,k]),
                                                                np.ascontiguousarray(data.release_fraction[:,k]))
                    gp=go[:,r]+capture_adjoints[k]
                    gdel=gp
                else:
                    gout=go[:,r]+gi[:,meta['target']]
                    gp=reservoir_backward(np.ascontiguousarray(gout+gr[:,k]),np.ascontiguousarray(data.release_fraction[:,k]))
                    fraction=np.where(data.enabled[:,k],meta['fraction'],1.)
                    gdel=gp*fraction+gout*(1-fraction)
            gi[:,r]+=gp*attenuation
            gl[:,r]=gdel*half
            gv+=np.sum(-gp*h*ctx.inlet[:,r]*attenuation-.5*gdel*h*local[:,r]*half)
        return (torch.from_numpy(gl),torch.tensor(gv,dtype=torch.float64),None,None,None)[:ctx.nargs]


def boundary_mass(inlet,official,releases,local,vf,c):
    ti,ri,f,h=c['ti'],c['ri'],c['f'],c['h']
    mass=inlet[ti,ri]*torch.exp(-vf*h*f)+f*local[ti,ri]*torch.exp(-.5*vf*h*f)
    if 'boundary_code' in c:
        mass=torch.where(c['boundary_code']==1,releases[ti,c['reservoir_index']],mass)
        mass=torch.where(c['boundary_code']==2,official[ti,ri],mass)
    return mass


def monthly_sum(values,data):
    # index_add propagates all day gradients, including pre-observation history.
    index=torch.from_numpy(data.mid)
    result=torch.zeros((len(data.months),data.source.shape[1]),dtype=values.dtype)
    return result.index_add(0,index,values)
