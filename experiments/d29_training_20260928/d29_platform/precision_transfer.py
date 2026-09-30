"""Experimental float64 expansion transfer; not enabled in formal calibration.

The largest destination retains the exact remaining row fraction. This treats
row fractions as a partition of unity, and records the tiny representational
change separately. No observed nitrogen residual is used to change a state.
Both high and low parts are physical numerical state, not an error allowance.
"""
import numpy as np
from numba import njit
from .land1 import _accurate_sum


@njit(cache=True)
def two_product(a,b):
    p=a*b
    ca=134217729.*a;ah=ca-(ca-a);al=a-ah
    cb=134217729.*b;bh=cb-(cb-b);bl=b-bh
    return p,((ah*bh-p)+ah*bl+al*bh)+al*bl


@njit(cache=True)
def transfer_expansion(high,low,matrix):
    nr,nl,ns=high.shape
    result=np.zeros_like(high);tail=np.zeros_like(high)
    terms=np.empty((nl,6*nl),np.float64)
    max_fraction_change=0.
    for r in range(nr):
        for s in range(ns):
            terms[:]=0.
            for old in range(nl):
                pivot=np.argmax(matrix[r,old]);fractions=matrix[r,old].copy()
                other=fractions.copy();other[pivot]=0.
                # Exact residual fraction represented by two float64 parts.
                fh=_accurate_sum(np.concatenate((np.ones(1),-other)))
                fl=_accurate_sum(np.concatenate((np.ones(1),-other,np.array([-fh]))))
                max_fraction_change=max(max_fraction_change,abs(fh+fl-fractions[pivot]))
                for new in range(nl):
                    fraction=fh if new==pivot else fractions[new]
                    fraclow=fl if new==pivot else 0.
                    p,e=two_product(high[r,old,s],fraction)
                    pl,el=two_product(low[r,old,s],fraction)
                    terms[new,6*old]=p;terms[new,6*old+1]=e
                    terms[new,6*old+2]=pl;terms[new,6*old+3]=el
                    terms[new,6*old+4]=high[r,old,s]*fraclow
                    terms[new,6*old+5]=low[r,old,s]*fraclow
            for new in range(nl):
                h=_accurate_sum(terms[new]);result[r,new,s]=h
                tail[r,new,s]=_accurate_sum(np.concatenate((terms[new],np.array([-h]))))
    return result,tail,max_fraction_change


def represented_balance(new,newlow,old,oldlow):
    import math
    return max(abs(math.fsum([*new[r].ravel(),*newlow[r].ravel(),*(-old[r]).ravel(),*(-oldlow[r]).ravel()])) for r in range(len(old)))
