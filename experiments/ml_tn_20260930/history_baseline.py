"""No-fit last-visible-TN comparator with the same forecast-origin gates."""
import numpy as np
import pandas as pd
from mltn.common import ROOT,write,sha
from mltn.history import HistoryInputs,origins
from mltn.data import balanced_weights,weights_and_scales

def main():
    for task in ['monthly','daily']:
        for stage,cutoff in [('F23',2022),('F24',2023)]:
            ready=pd.Timestamp(f'{cutoff+1}-01-02')
            for lead in ([1] if task=='monthly' else [1,7,30]):
                d=HistoryInputs(task,lead);q=d.labels(task);tr=d.split(q,cutoff)
                if task=='daily':
                    av=pd.read_parquet(ROOT/'data/hf_daily_availability.parquet')[['station_key','date','available']]
                    tr=tr.merge(av,on=['station_key','date'],validate='one_to_one')
                    tr=tr[tr.available<ready].reset_index(drop=True)
                else:tr=tr[tr.date+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)<ready].reset_index(drop=True)
                scales,floor=weights_and_scales(tr,'tn_mg_l');w=balanced_weights(tr,scales,floor,task=='daily')
                fallback=float(np.average(tr.tn_mg_l,weights=w))
                ev=q[q.year.eq(cutoff+1)].copy();ev=ev[origins(ev,task,lead)>=ready].reset_index(drop=True)
                closed=HistoryInputs(task,lead,freeze_year=cutoff,freeze_at=ready)
                root=ROOT/'jobs'/f'aux_{stage}_{task}_persistence_lead{lead}';root.mkdir(parents=True,exist_ok=True)
                for dd,filename,mode in [(d,'prediction.parquet','rolling_observed_before_origin'),(closed,'closed_loop_prediction.parquet','annual_frozen_last_observation')]:
                    p=dd.feedback(ev)[:,0];out=ev.copy();out['prediction']=np.where(np.isfinite(p),p,fallback);out['feedback_mode']=mode
                    out.to_parquet(root/filename,index=False)
                write(root/'result.json',dict(status='complete',role='simple_reference',model='no-fit last-visible TN',task=task,lead=lead,model_ready_time=str(ready),training_cutoff=cutoff,missing_history_fallback='train-only weighted pooled TN mean',closed_loop='frozen last observation; no evaluation labels or missingness',code_sha256=sha(ROOT/'history_baseline.py')))

if __name__=='__main__':main()
