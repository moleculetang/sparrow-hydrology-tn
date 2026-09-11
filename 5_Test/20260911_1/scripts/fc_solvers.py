"""Serializable SciPy 1.17.1 reverse-communication solvers.

Preserve line-search, quasi-Newton memory and counters across resource yield.
The implementation follows installed scipy.optimize wrappers, avoiding their
non-serializable Python loop frame. Scientific tolerances match parent fits.
"""
import copy
import numpy as np
import scipy
from scipy.optimize import _lbfgsb
from scipy.optimize._slsqplib import slsqp


def version_guard():
    if scipy.__version__!='1.17.1':raise RuntimeError('Unregistered low-level SciPy ABI')


def projected_gradient(x,g,bounds):
    result=g.copy()
    for i,(lo,hi) in enumerate(bounds):
        if lo is not None and x[i]<=lo+1e-7 and g[i]>0:result[i]=0
        if hi is not None and x[i]>=hi-1e-7 and g[i]<0:result[i]=0
    return result


class LBFGSB:
    def __init__(self,x,bounds,stage=0,maxiter=1200,state=None):
        version_guard()
        if state is not None:
            self.s=copy.deepcopy(state);return
        x=np.asarray(x,dtype=np.float64).copy();n=len(x);m=20
        lo=np.zeros(n);hi=np.zeros(n);nbd=np.zeros(n,dtype=np.int32)
        for i,(a,b) in enumerate(bounds):
            if a is not None:lo[i]=a;nbd[i]=1
            if b is not None:hi[i]=b;nbd[i]=2 if a is not None else 3
        self.s={'method':'L-BFGS-B','x':x,'f':np.array(0.,dtype=np.float64),'g':np.zeros(n),'lo':lo,'hi':hi,'nbd':nbd,
                'm':m,'factr':(1e-14 if stage==0 else 0.)/np.finfo(float).eps,'pgtol':1e-8,'maxls':100,'maxiter':maxiter,
                'wa':np.zeros(2*m*n+5*n+11*m*m+8*m),'iwa':np.zeros(3*n,dtype=np.int32),
                'task':np.zeros(2,dtype=np.int32),'ln_task':np.zeros(2,dtype=np.int32),'lsave':np.zeros(4,dtype=np.int32),
                'isave':np.zeros(44,dtype=np.int32),'dsave':np.zeros(29),'iterations':0,'evaluations':0,
                'pending_eval':False,'done':False}

    def step(self,fun):
        s=self.s
        if s['done']:return 'done'
        if s['pending_eval']:
            f,g=fun(s['x'].copy())
            if not np.isfinite(f) or not np.isfinite(g).all():raise FloatingPointError('Nonfinite optimizer value/gradient')
            s['f'][...]=f;s['g'][...]=g;s['evaluations']+=1;s['pending_eval']=False
            return 'evaluation'
        _lbfgsb.setulb(s['m'],s['x'],s['lo'],s['hi'],s['nbd'],s['f'],s['g'],s['factr'],s['pgtol'],s['wa'],s['iwa'],
                       s['task'],s['lsave'],s['isave'],s['dsave'],s['maxls'],s['ln_task'])
        if s['task'][0]==3:
            s['pending_eval']=True;return 'request_evaluation'
        if s['task'][0]==1:
            s['iterations']+=1
            if s['iterations']>=s['maxiter']:
                s['task'][0]=5;s['task'][1]=504
            return 'iteration'
        s['done']=True
        return 'done'


class SLSQP:
    def __init__(self,x,bounds,maxiter=1200,state=None):
        version_guard()
        if state is not None:
            self.s=copy.deepcopy(state);return
        x=np.asarray(x,dtype=np.float64).copy();n=len(x);acc=1e-13
        st={k:0. for k in ['alpha','f0','gs','h1','h2','h3','h4','t','t0']}
        st.update(acc=acc,tol=10*acc,exact=0,inconsistent=0,reset=0,iter=0,itermax=maxiter,line=0,m=0,meq=0,mode=0,n=n)
        buffer_size=n*(n+1)//2+8*n*n+35*n+28+2*n*(n+1)
        lo=np.array([a if a is not None else np.nan for a,b in bounds]);hi=np.array([b if b is not None else np.nan for a,b in bounds])
        self.s={'method':'SLSQP','x':x,'f':0.,'g':np.zeros(n),'lo':lo,'hi':hi,'solver':st,
                'buffer':np.zeros(buffer_size),'indices':np.zeros(2*n+2,dtype=np.int32),'mult':np.zeros(2*n+2),
                'C':np.zeros((1,n),order='F'),'d':np.zeros(1),'iterations':0,'evaluations':0,
                'cache_x':None,'cache_f':None,'cache_g':None,'pending':'initial','done':False}

    def step(self,fun):
        s=self.s
        if s['done']:return 'done'
        if s['pending']:
            lo=np.where(np.isnan(s['lo']),-np.inf,s['lo']);hi=np.where(np.isnan(s['hi']),np.inf,s['hi'])
            x=np.clip(s['x'],lo,hi)
            if s['cache_x'] is None or not np.array_equal(x,s['cache_x']):
                f,g=fun(x.copy())
                if not np.isfinite(f) or not np.isfinite(g).all():raise FloatingPointError('Nonfinite optimizer value/gradient')
                s.update(cache_x=x.copy(),cache_f=float(f),cache_g=g.copy(),evaluations=s['evaluations']+1)
            if s['pending'] in ['initial','f']:s['f']=s['cache_f']
            if s['pending'] in ['initial','g']:s['g']=s['cache_g'].copy()
            s['pending']=None
            return 'evaluation'
        slsqp(s['solver'],s['f'],s['g'],s['C'],s['d'],s['x'],s['mult'],s['lo'],s['hi'],s['buffer'],s['indices'])
        s['iterations']=s['solver']['iter']
        mode=s['solver']['mode']
        if mode==1:s['pending']='f';return 'request_evaluation'
        if mode==-1:s['pending']='g';return 'request_evaluation'
        s['done']=True
        return 'done'


def restore_solver(state):
    if state['method']=='L-BFGS-B':return LBFGSB([],[],state=state)
    if state['method']=='SLSQP':return SLSQP([],[],state=state)
    raise ValueError('Unknown solver checkpoint')
