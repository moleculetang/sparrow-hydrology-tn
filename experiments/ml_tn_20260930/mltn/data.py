import numpy as np,pandas as pd
from mltn.common import ROOT,read
class Inputs:
    def __init__(self):
        self.identity=read(ROOT/'data/feature_identity.json');self.dates=pd.DatetimeIndex(self.identity['dates'])
        self.x=np.load(ROOT/'data'/self.identity.get('array_file','reach_features.npy'),mmap_mode='r');self.adj=np.load(ROOT/'data/adjacency.npy')
        self.reg=pd.read_parquet(ROOT/'data/station_registry.parquet');self.station_index=dict(zip(self.reg.station_key,range(len(self.reg))))
        self.rr=self.reg.reach_id.to_numpy(int)-1;self.water=np.load(ROOT/'data/station_water.npy',mmap_mode='r')
        self.blocks=read(ROOT/'data/spatial_blocks.json')
    def labels(self,task):
        q=pd.read_parquet(ROOT/'data'/('monthly_tn_accepted.parquet' if task=='monthly' else 'hf_daily_accepted.parquet')).copy()
        q['date']=pd.to_datetime(q.date);q['year']=q.date.dt.year;q['month']=q.date.dt.month
        q['si']=q.station_key.map(self.station_index).astype(int);q['ri']=self.rr[q.si]
        q['end_date']=q.date+pd.offsets.MonthEnd(0) if task=='monthly' else q.date
        q['ti']=self.dates.get_indexer(q.end_date);q['start_ti']=self.dates.get_indexer(q.date.dt.to_period('M').dt.to_timestamp()) if task=='monthly' else q.ti
        if (q.ti<0).any():raise ValueError('LABEL_DATE_OUTSIDE_FEATURES')
        return q.reset_index(drop=True)
    def split(self,q,cutoff,block=None):
        allowed=~q.station_key.isin([] if block is None else self.blocks[str(block)]['held_stations']+self.blocks[str(block)]['buffer_stations'])
        return q[(q.year<=cutoff)&allowed].reset_index(drop=True)
    def raw_rows(self,q,monthly=False):
        x=np.asarray(self.x[q.ti,q.ri],float).copy()
        if monthly:
            for i,row in enumerate(q.itertuples()):x[i]=np.mean(self.x[row.start_ti:row.ti+1,row.ri],axis=0,dtype=float)
        w=np.log1p(self.water[q.ti,q.si]);f=self.reg.downstream_fraction_on_reach.to_numpy()[q.si]
        if monthly:
            w=np.array([np.mean(np.log1p(self.water[r.start_ti:r.ti+1,r.si])) for r in q.itertuples()])
        return np.column_stack([x,w,f])
    def sequence(self,q,window,graph=False):
        # B x days x features; causal end index includes only available forcings.
        ix=q.ti.to_numpy()[:,None]-np.arange(window-1,-1,-1)[None,:]
        if (ix<0).any():raise ValueError('INSUFFICIENT_SEQUENCE_WARMUP')
        r=q.ri.to_numpy();x=np.asarray(self.x[ix,r[:,None]],dtype=np.float32)
        if graph:
            # Exact fixed graph filter: incoming direct nodes, shared trainable channel map later.
            neighbor=np.empty_like(x)
            for j,ri in enumerate(r):
                nodes=np.flatnonzero(self.adj[ri]);neighbor[j]=0 if not len(nodes) else np.mean(self.x[ix[j][:,None],nodes[None,:]],axis=1)
            x=np.concatenate([x,neighbor],axis=-1)
        return np.concatenate([x,np.log1p(self.water[ix,q.si.to_numpy()[:,None]])[...,None].astype('float32'),np.broadcast_to(self.reg.downstream_fraction_on_reach.to_numpy()[q.si][:,None,None],(*x.shape[:2],1)).astype('float32')],axis=-1)
    def cached_sequence(self,q,window,graph,transform):
        """Normalize each immutable day/reach once; exact FP32 parity with sequence/apply."""
        f=self.x.shape[-1];nf=f*(2 if graph else 1)+2
        # History feedback has origin-dependent extra channels and uses its own path.
        if len(transform.mean)!=nf:return None
        key=(id(transform),graph,id(self.x),id(self.water),id(self.reg))
        if getattr(self,'_neural_key',None)!=key:
            def normalize(raw,start):
                out=np.empty((*raw.shape[:-1],raw.shape[-1]*2),dtype=np.float32);n=raw.shape[-1]
                for lo in range(0,len(raw),128):
                    v=raw[lo:lo+128];bad=~np.isfinite(v);out[lo:lo+128,:,:n]=((np.where(bad,transform.median[start:start+n],v)-transform.mean[start:start+n])/transform.sd[start:start+n]).astype('float32');out[lo:lo+128,:,n:]=bad
                return out
            local=normalize(self.x,0);neighbor=None
            if graph:
                raw=np.empty(self.x.shape,dtype=np.float32)
                for ri in range(self.x.shape[1]):
                    nodes=np.flatnonzero(self.adj[ri]);raw[:,ri]=0 if not len(nodes) else np.mean(self.x[:,nodes],axis=1)
                neighbor=normalize(raw,f)
            # sequence() casts physical readout channels to FP32 before apply().
            water=np.log1p(self.water).astype('float32');bad=~np.isfinite(water);wi=nf-2
            wv=((np.where(bad,transform.median[wi],water)-transform.mean[wi])/transform.sd[wi]).astype('float32')
            frac=self.reg.downstream_fraction_on_reach.to_numpy().astype('float32');fb=~np.isfinite(frac);fv=((np.where(fb,transform.median[nf-1],frac)-transform.mean[nf-1])/transform.sd[nf-1]).astype('float32')
            self._neural_cache=(local,neighbor,wv,bad,fv,~np.isfinite(frac));self._neural_key=key
        local,neighbor,wv,wb,fv,fb=self._neural_cache
        ix=q.ti.to_numpy()[:,None]-np.arange(window-1,-1,-1)[None,:]
        if (ix<0).any():raise ValueError('INSUFFICIENT_SEQUENCE_WARMUP')
        r=q.ri.to_numpy();si=q.si.to_numpy();out=np.empty((len(q),window,nf*2),dtype=np.float32)
        v=local[ix,r[:,None]];out[:,:,:f]=v[:,:,:f];out[:,:,nf:nf+f]=v[:,:,f:]
        if graph:
            v=neighbor[ix,r[:,None]];out[:,:,f:2*f]=v[:,:,:f];out[:,:,nf+f:nf+2*f]=v[:,:,f:]
        out[:,:,nf-2]=wv[ix,si[:,None]];out[:,:,2*nf-2]=wb[ix,si[:,None]]
        out[:,:,nf-1]=fv[si,None];out[:,:,2*nf-1]=fb[si,None]
        return out
class Transform:
    def fit(self,x):
        flat=np.asarray(x,float).reshape(-1,x.shape[-1]);self.median=np.nanmedian(flat,axis=0);self.median=np.where(np.isfinite(self.median),self.median,0)
        fill=np.where(np.isfinite(flat),flat,self.median);self.mean=fill.mean(0);self.sd=fill.std(0);self.sd[self.sd==0]=1
        self.all_missing=~np.isfinite(flat).any(0)
        return self
    def apply(self,x):
        bad=~np.isfinite(x);v=(np.where(bad,self.median,x)-self.mean)/self.sd
        return np.concatenate([v,bad.astype(float)],axis=-1).astype('float32')
def weights_and_scales(train,target):
    stats=train.groupby('station_key')[target].var(ddof=0);positive=stats[stats>0]
    if positive.empty:raise ValueError('NO_POSITIVE_TRAIN_VARIANCE')
    floor=float(positive.quantile(.1));return stats.clip(lower=floor).fillna(floor).to_dict(),floor
def balanced_weights(q,scales,floor,read_weight=False):
    # Station equal, observed year equal, month equal, within-month count weighted.
    q=q.copy();keys=['station_key','year','month'];nstation=q.station_key.nunique();ny=q.groupby('station_key').year.nunique();nm=q[keys].drop_duplicates().groupby(['station_key','year']).size()
    counts=q.read_count.to_numpy(float) if read_weight else np.ones(len(q));q['_n']=counts
    totals=q.groupby(keys)._n.transform('sum').to_numpy()
    denom=np.array([nstation*ny[r.station_key]*nm.loc[(r.station_key,r.year)]*scales.get(r.station_key,floor) for r in q.itertuples()])
    return counts/totals/denom
