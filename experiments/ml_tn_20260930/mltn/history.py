"""Strict availability-before-origin TN feedback; entirely separate from main inputs."""
import numpy as np,pandas as pd
from mltn.data import Inputs
from mltn.common import ROOT
class HistoryInputs(Inputs):
    def __init__(self,task,lead,freeze_year=None,freeze_at=None):
        super().__init__();self.task=task;self.lead=lead;self.history={};q=super().labels(task)
        if freeze_year is not None:q=q[q.year<=freeze_year]
        q=q.copy()
        if task=='daily':
            availability=pd.read_parquet(ROOT/'data/hf_daily_availability.parquet')[['station_key','date','available']]
            q=q.merge(availability,on=['station_key','date'],how='left',validate='one_to_one')
            if q.available.isna().any():raise ValueError('MISSING_TN_ACQUISITION_TIME')
        else:q['available']=q.date+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)
        q['observed_at']=q.date+pd.Timedelta(days=1) if task=='daily' else q.date+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)
        if freeze_at is not None:q=q[q.available<pd.Timestamp(freeze_at)]
        elif freeze_year is not None:q=q[q.available<=pd.Timestamp(f'{freeze_year+1}-01-01')]
        for s,g in q.groupby('station_key'):self.history[s]=g[['available','observed_at','tn_mg_l']].sort_values('available').reset_index(drop=True)
    def feedback(self,q):
        rows=[]
        for row in q.itertuples():
            origin=row.date-pd.Timedelta(days=self.lead) if self.task=='daily' else row.date-pd.DateOffset(months=self.lead)
            g=self.history.get(row.station_key);v=np.full(6,np.nan);v[-1]=1
            if g is not None:
                # Side left excludes a reading whose availability equals the forecast origin.
                n=np.searchsorted(g.available.to_numpy('datetime64[ns]'),np.datetime64(origin.to_datetime64()),side='left')
                if n:
                    visible=g.iloc[:n].sort_values('observed_at');latest=visible.iloc[-1];v[0]=latest.tn_mg_l;v[1]=float((origin-latest.observed_at)/pd.Timedelta(days=1));v[-1]=0
                    for k in [2,3]:
                        if n>=k:v[k]=visible.tn_mg_l.iloc[-k]
                    z=visible[visible.observed_at>=origin-pd.Timedelta(days=90)]
                    if len(z):v[4]=z.tn_mg_l.mean()
            rows.append(v)
        return np.asarray(rows,dtype=np.float32)
    def raw_rows(self,q,monthly=False):return np.column_stack([super().raw_rows(q,monthly),self.feedback(q)])
    def sequence(self,q,window,graph=False):
        x=super().sequence(q,window,graph);h=self.feedback(q);return np.concatenate([x,np.broadcast_to(h[:,None,:],(len(q),window,h.shape[-1]))],axis=-1)
    def append_prediction(self,station,date,value):
        available=pd.Timestamp(date)+pd.Timedelta(days=1) if self.task=='daily' else pd.Timestamp(date)+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)
        row=pd.DataFrame([dict(available=available,observed_at=available,tn_mg_l=float(value))]);self.history[station]=pd.concat([self.history.get(station,row.iloc[:0]),row],ignore_index=True).sort_values('available').reset_index(drop=True)

def origins(q,task,lead):
    return pd.to_datetime(q.date)-pd.Timedelta(days=lead) if task=='daily' else pd.to_datetime(q.date)-pd.DateOffset(months=lead)
