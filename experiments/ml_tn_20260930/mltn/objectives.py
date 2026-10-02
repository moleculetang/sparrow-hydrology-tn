"""Sparse daily observation operators and exact transpose gradients."""
import numpy as np
from scipy.sparse import csr_matrix
class AggregateObjective:
    def __init__(self,month_operator,y_month,w_month,hf_operator,y_hf,w_hf,groups,counts):
        self.B=csr_matrix(month_operator);self.H=csr_matrix(hf_operator);self.ym=np.asarray(y_month);self.wm=np.asarray(w_month)
        self.yh=np.asarray(y_hf);self.wh=np.asarray(w_hf);self.groups=np.asarray(groups);self.counts=np.asarray(counts)
    def center(self,x):
        v=x.copy()
        for g in np.unique(self.groups):
            z=self.groups==g;v[z]-=np.average(x[z],weights=self.counts[z])
        return v
    def transpose_center(self,x):
        v=x.copy()
        for g in np.unique(self.groups):
            z=self.groups==g;v[z]-=self.counts[z]/self.counts[z].sum()*x[z].sum()
        return v
    def value_gradient(self,p):
        em=self.B@p-self.ym;eh=self.center(self.H@p-self.yh)
        j=.4*np.dot(self.wm,em*em)+.1*np.dot(self.wh,eh*eh)
        g=.8*self.B.T@(self.wm*em)+.2*self.H.T@self.transpose_center(self.wh*eh)
        return float(j),np.asarray(g).ravel()
    def diagonal(self):
        # Exact diagonal of Hessian, off-diagonal couplings dropped by tree learner.
        diag=.8*np.asarray(self.B.power(2).T@self.wm).ravel()
        for group in np.unique(self.groups):
            ix=np.flatnonzero(self.groups==group);a=self.counts[ix]/self.counts[ix].sum()
            c=np.eye(len(ix))-np.ones((len(ix),1))*a[None,:]
            z=csr_matrix(c)@self.H[ix];diag+=.2*np.asarray(z.power(2).T@self.wh[ix]).ravel()
        return diag
    def diagonal_majorizer(self):
        # Cauchy-Schwarz row-L1 bound: diag(sum w |v| ||v||1) >= sum w vvT.
        # Exact gradients stay unchanged; this prevents month-mean common-mode
        # curvature being underestimated by a factor equal to month length.
        b=abs(self.B);d=.8*np.asarray(b.T@(self.wm*np.asarray(b.sum(axis=1)).ravel())).ravel()
        for group in np.unique(self.groups):
            ix=np.flatnonzero(self.groups==group);a=self.counts[ix]/self.counts[ix].sum();c=np.eye(len(ix))-np.ones((len(ix),1))*a[None,:]
            z=abs(csr_matrix(c)@self.H[ix]);d+=.2*np.asarray(z.T@(self.wh[ix]*np.asarray(z.sum(axis=1)).ravel())).ravel()
        return d
