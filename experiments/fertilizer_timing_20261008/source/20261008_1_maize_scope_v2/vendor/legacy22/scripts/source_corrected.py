"""Shared source-estimate correction; full-history input adjoint, unchanged demand."""
import math
import numpy as np
import torch
from numba import njit
from hf_model import HFEndpoints
from closures import scan
from balanced_tags import balanced_scan

GROUPS={'SOURCE_UNIFIED':[0,0,0,0], 'SOURCE_GROUPED':[0,0,0,1], 'SOURCE_SEPARATE':[0,1,2,3]}

def embed(parameters, old_kind, new_kind):
    x=np.asarray(parameters,float)
    expanded=np.zeros(4) if old_kind=='D29_BE' else x[30:][GROUPS[old_kind]]
    groups=np.asarray(GROUPS[new_kind]);v=[]
    for k in range(groups.max()+1):
        a=expanded[groups==k]
        if not np.all(a==a[0]):raise ValueError('NON_NESTED_GROUPS')
        v.append(a[0])
    return np.r_[x[:30],v]

@njit(cache=True)
def reverse_input(gfast,gslow,h,s,f,release,a,p,starts,nmonths):
    nd,nr=h.shape;gh=np.zeros_like(h);gs=np.zeros_like(s);gf=np.zeros_like(f)
    gi=np.zeros((nmonths,nr));am=np.zeros(nr);al=np.zeros(nr)
    month=nmonths-1
    for t in range(nd-1,-1,-1):
        injection=t==starts[month]
        for r in range(nr):
            av=a[t,r];prob=p[t,r]
            pre=gslow[t,r]*release[t,r]+al[r]*(1-release[t,r])
            ae=gfast[t,r]*f[t,r]+pre*(1-f[t,r])
            ap=av*(ae-am[r]*s[r])
            gf[t,r]=av*prob*(gfast[t,r]-pre)
            gs[r]+=am[r]*av*(1-prob)
            if h[t,r]<700.:gh[t,r]=ap*np.exp(-h[t,r])
            aa=ae*prob+am[r]*(1-prob)*s[r]
            am[r]=aa if av>0 else 0.
            al[r]=pre
            if injection:gi[month,r]=am[r]
        if injection:month=max(0,month-1)
    return gh,gs,gf,gi

class SourceTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,s,f,source,owner):
        hn,sn,fn=[np.ascontiguousarray(x.detach().numpy()) for x in (h,s,f)]
        inp=np.zeros_like(hn);inp[owner.data.starts]=source.detach().numpy()
        ff,ss,a,p=scan(hn,sn,fn,np.ones(hn.shape[1]),owner.data.lower_release,inp,owner.demand,False)
        ctx.val=(hn,sn,fn,a,p);ctx.owner=owner
        return torch.from_numpy(ff),torch.from_numpy(ss)
    @staticmethod
    def backward(ctx,gfast,gslow):
        h,s,f,a,p=ctx.val;o=ctx.owner
        gf=np.zeros_like(h) if gfast is None else np.ascontiguousarray(gfast.numpy())
        gl=np.zeros_like(h) if gslow is None else np.ascontiguousarray(gslow.numpy())
        v=reverse_input(gf,gl,h,s,f,o.data.lower_release,a,p,o.data.starts,len(o.data.months))
        return (*[torch.from_numpy(x) for x in v],None)

class SourceCorrected(HFEndpoints):
    def __init__(self,data,train,design,kind):
        super().__init__(data,train,design)
        self.source_kind=kind;self.source_groups=torch.tensor(GROUPS[kind],dtype=torch.long)
        self.nsource=max(GROUPS[kind])+1
        self.names+=['log_source_correction_'+str(i) for i in range(self.nsource)]
        self.bounds += [(-math.log(4),math.log(4))]*self.nsource
        self.original_source=np.array(self.data.source,copy=True)
        self.original_tags=np.array(self.data.source_tags,copy=True)
        if np.max(abs(self.original_tags.sum(-1)-self.original_source))>1e-6:raise ValueError('SOURCE_IDENTITY')
        self.source_tensor=torch.tensor(self.original_source)
        self.tags_tensor=torch.tensor(self.original_tags)
    def expanded_eta(self,t):return t[30:][self.source_groups]
    def corrected_source(self,t):
        # Anchored arithmetic preserves the exact original total at all c=1.
        return self.source_tensor+torch.sum(self.tags_tensor*torch.expm1(self.expanded_eta(t))[None,None,:],dim=-1)
    def initial(self,index=0):return np.r_[super().initial(index),np.zeros(self.nsource)]
    def variable_scale(self):return np.r_[super().variable_scale()[:30],np.repeat(math.log(2),self.nsource)]
    def prior(self,t):
        extra=self.expanded_eta(t)/math.log(2)*math.sqrt(.03/17/4)
        return torch.cat([super().prior(t[:30]),extra])
    def flux_parameters(self,t):return super().flux_parameters(t[:30])
    def land_transport(self,t,h,s,f,k):return SourceTransport.apply(h,s,f,self.corrected_source(t),self)
    def ledger(self,x):
        source=self.corrected_source(torch.tensor(x)).detach().numpy()
        tags=self.original_tags*np.exp(np.asarray(x)[30:][GROUPS[self.source_kind]])[None,None,:]
        old=(self.data.source,self.data.source_tags,self.inp,self.tag_inputs)
        try:
            self.data.source=source;self.data.source_tags=tags
            self.inp=np.zeros_like(self.data.contact);self.inp[self.data.starts]=source
            self.tag_inputs=np.zeros_like(self.tag_inputs);self.tag_inputs[self.data.starts]=tags[:,self.data.pilot_indices,:]
            a=super().ledger(x)
            rr=self.data.pilot_indices
            if rr:
                with torch.no_grad():h,s,f,k=[v.numpy() for v in self.flux_parameters(torch.tensor(x))]
                v=balanced_scan(h[:,rr],s[rr],f[:,rr],self.data.lower_release[:,rr],self.tag_inputs,self.inp[:,rr],self.demand[:,rr])
                a['source_labels']={name:v[i] for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]}
                a['source_label_sum_errors']={name:float(np.max(abs(value.sum(-1)-a[name][:,rr]))) for name,value in a['source_labels'].items()}
                a['source_label_sum_errors']['per_source_balance']=v[8]
                if v[9]<-1e-7:raise ValueError('NEGATIVE_SOURCE_ACCOUNT')
                a['source_partition_rounding_max']=v[7]
            a['corrected_source_monthly']=source;a['source_multipliers']=np.exp(np.asarray(x)[30:][GROUPS[self.source_kind]])
            return a
        finally:self.data.source,self.data.source_tags,self.inp,self.tag_inputs=old
    def value_gradient(self,x):
        v,g=super().value_gradient(x)
        eta=np.asarray(x)[30:][GROUPS[self.source_kind]]
        extra=.5*(.03/17)*np.mean((eta/math.log(2))**2)
        self.last_terms.update(new_prior=float(extra),original_prior=float(self.last_terms['prior']-extra))
        return v,g
