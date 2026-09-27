"""Conservative source partition with explicit floating-point residual accounting.

The largest share receives the arithmetic residual of each proportional split.
This changes no total recurrence, source coefficient, physical flux or tolerance.
Both the rounding adjustments and per-source balances are returned for audit.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def partition(total,weights,out):
    pivot=0
    for j in range(len(weights)):
        out[j]=total*weights[j]
        if weights[j]>weights[pivot]:pivot=j
    subtotal=0.
    for j in range(len(weights)):
        if j!=pivot:subtotal+=out[j]
    before=out[pivot];out[pivot]=total-subtotal
    return abs(out[pivot]-before)

@njit(cache=True)
def balanced_scan(h,s,f,release,inputs,total_inputs,demand):
    nd,nr=h.shape;ns=inputs.shape[-1]
    fast=np.zeros_like(inputs);slow=np.zeros_like(inputs);raw=np.zeros_like(inputs)
    ms=np.zeros_like(inputs);ls=np.zeros_like(inputs);uptake=np.zeros_like(inputs);loss=np.zeros_like(inputs)
    M=np.zeros((nr,ns));L=np.zeros((nr,ns));mt=np.zeros(nr);lt=np.zeros(nr)
    w=np.zeros(ns);v=np.zeros(ns);jmass=np.zeros(ns);pre=np.zeros(ns)
    adjustment=0.;balance=0.;minimum=0.
    for t in range(nd):
        for r in range(nr):
            B=mt[r]+total_inputs[t,r];U=min(B,demand[t,r]);A=max(B-demand[t,r],0.)
            prob=-np.expm1(-min(h[t,r],700.));E=A*prob
            F=E*f[t,r];J=E*(1-f[t,r]);P=lt[r]+J;S=P*release[t,r]
            MN=A*(1-prob)*s[r];LN=P-S;D=A*(1-prob)*(1-s[r])
            rawsum=0.
            for j in range(ns):raw[t,r,j]=M[r,j]+inputs[t,r,j];rawsum+=raw[t,r,j]
            for j in range(ns):w[j]=raw[t,r,j]/rawsum if rawsum>0 else (1./ns)
            adjustment=max(adjustment,partition(U,w,uptake[t,r]))
            adjustment=max(adjustment,partition(F,w,fast[t,r]))
            adjustment=max(adjustment,partition(D,w,loss[t,r]))
            adjustment=max(adjustment,partition(MN,w,ms[t,r]))
            adjustment=max(adjustment,partition(J,w,jmass))
            pretotal=0.
            for j in range(ns):pre[j]=L[r,j]+jmass[j];pretotal+=pre[j]
            for j in range(ns):v[j]=pre[j]/pretotal if pretotal>0 else 1./ns
            adjustment=max(adjustment,partition(S,v,slow[t,r]))
            adjustment=max(adjustment,partition(LN,v,ls[t,r]))
            for j in range(ns):
                err=M[r,j]+L[r,j]+inputs[t,r,j]-uptake[t,r,j]-loss[t,r,j]-fast[t,r,j]-slow[t,r,j]-ms[t,r,j]-ls[t,r,j]
                balance=max(balance,abs(err));minimum=min(minimum,ms[t,r,j],ls[t,r,j],uptake[t,r,j],loss[t,r,j],fast[t,r,j],slow[t,r,j])
                M[r,j]=ms[t,r,j];L[r,j]=ls[t,r,j]
            mt[r]=MN;lt[r]=LN
    return fast,slow,raw,ms,ls,uptake,loss,adjustment,balance,minimum
