"""Explicit kg N/day sources; no observation arguments or fitted calendars."""
import numpy as np
import torch
from numba import njit
from closures import scan, ResearchObjective
from balanced_tags import balanced_scan
from campaign_model import RUN, sha

@njit(cache=True)
def daily_reverse(gfast,gslow,h,s,f,release,a,p):
    nd,nr=h.shape
    gh=np.zeros_like(h);gs=np.zeros_like(s);gf=np.zeros_like(f);gi=np.zeros_like(h)
    am=np.zeros(nr);al=np.zeros(nr)
    for t in range(nd-1,-1,-1):
        for r in range(nr):
            av=a[t,r];prob=p[t,r]
            pre=gslow[t,r]*release[t,r]+al[r]*(1-release[t,r])
            ae=gfast[t,r]*f[t,r]+pre*(1-f[t,r])
            ap=av*(ae-am[r]*s[r])
            gf[t,r]=av*prob*(gfast[t,r]-pre)
            gs[r]+=am[r]*av*(1-prob)
            if h[t,r]<700:gh[t,r]=ap*np.exp(-h[t,r])
            aa=ae*prob+am[r]*(1-prob)*s[r]
            am[r]=aa if av>0 else 0.
            al[r]=pre;gi[t,r]=am[r]
    return gh,gs,gf,gi

class DailyTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,s,f,source,owner):
        hn,sn,fn=[np.ascontiguousarray(v.detach().numpy()) for v in (h,s,f)]
        ff,ss,a,p=scan(hn,sn,fn,np.ones(hn.shape[1]),owner.data.lower_release,source.detach().numpy(),owner.demand,False)
        ctx.val=(hn,sn,fn,a,p);ctx.owner=owner
        return torch.from_numpy(ff),torch.from_numpy(ss)
    @staticmethod
    def backward(ctx,gfast,gslow):
        h,s,f,a,p=ctx.val;o=ctx.owner
        gf=np.zeros_like(h) if gfast is None else np.ascontiguousarray(gfast.numpy())
        gl=np.zeros_like(h) if gslow is None else np.ascontiguousarray(gslow.numpy())
        return (*[torch.from_numpy(v) for v in daily_reverse(gf,gl,h,s,f,o.data.lower_release,a,p)],None)

class DailyMixin:
    def __init__(self,data,train,design,kind):
        super().__init__(data,train,design,kind)
        spec=design['daily_input'];self.input_mode=spec['mode']
        assert self.data.operator_id=='OU','Daily adapter admits frozen OU only'
        for key in ['total','tags']:
            if sha(RUN/spec[key])!=spec[key+'_sha256']:raise ValueError('DAILY_INPUT_IDENTITY')
        self.raw_daily=np.load(RUN/spec['total'],mmap_mode='r')
        self.raw_daily_tags=np.load(RUN/spec['tags'],mmap_mode='r')
        self.source_names=spec['source_names']
        assert self.raw_daily.shape==self.data.contact.shape
        assert self.raw_daily_tags.shape==(*self.raw_daily.shape,len(self.source_names))
        self.daily_tensor=torch.tensor(np.array(self.raw_daily))
        if spec.get('daily_demand',False):
            self.demand=self.data.crop[self.data.mid]/(self.data.stops-self.data.starts)[self.data.mid,None]
    def corrected_source(self,t):
        # P preserves the old anchored arithmetic, including summation order.
        if self.input_mode=='P':
            monthly=super().corrected_source(t)
            result=torch.zeros_like(self.daily_tensor);result[self.data.starts]=monthly
            return result
        return self.daily_tensor+self.daily_tensor*torch.expm1(t[30])
    def land_transport(self,t,h,s,f,k):
        return DailyTransport.apply(h,s,f,self.corrected_source(t),self)
    def ledger(self,x):
        source=self.corrected_source(torch.tensor(x)).detach().numpy()
        old=self.inp
        try:
            self.inp=source
            a=ResearchObjective.ledger(self,x)
            rr=self.data.pilot_indices
            with torch.no_grad():h,s,f,k=[v.numpy() for v in self.flux_parameters(torch.tensor(x))]
            tags=np.ascontiguousarray(self.raw_daily_tags[:,rr,:]*np.exp(x[30]))
            v=balanced_scan(h[:,rr],s[rr],f[:,rr],self.data.lower_release[:,rr],tags,source[:,rr],self.demand[:,rr])
            a['source_labels']={name:v[i] for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]}
            a['source_label_global_reaches']=[self.data.global_reach_ids[i] for i in rr]
            a['source_label_names']=self.source_names
            a['source_label_sum_errors']={name:float(np.max(abs(value.sum(-1)-a[name][:,rr]),initial=0)) for name,value in a['source_labels'].items()}
            a['source_label_sum_errors']['per_source_balance']=v[8]
            if v[9]<-1e-7:raise ValueError('NEGATIVE_SOURCE_ACCOUNT')
            a['source_partition_rounding_max']=v[7]
            a['corrected_source_monthly']=self.data.monthly_sum(source)
            a['source_multipliers']=np.repeat(np.exp(x[30]),len(self.source_names))
            return a
        finally:self.inp=old

def make_daily(data,train,kind,design):
    from source_corrected import SourceCorrected
    from regional_response import RegionalResponse
    base=RegionalResponse if kind=='REGIONAL_L3' else SourceCorrected
    assert kind in ('SOURCE_UNIFIED','REGIONAL_L3')
    cls=type('Daily'+base.__name__,(DailyMixin,base),{})
    return cls(data,train,design,kind)
