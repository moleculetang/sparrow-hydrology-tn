"""Frozen historical descriptive comparators; never candidates for new selection."""
from pathlib import Path
import numpy as np,pandas as pd
from mltn.common import ROOT,write,sha
from mltn.data import Inputs
def main():
    d=Inputs();old=ROOT.parent/'20260928_1/outputs/jobs';receipts=[]
    for model in ['U','LAND1']:
        for fold,year in [('F23',2023),('F24',2024)]:
            folder=old/f'{model}_{fold}_T0_s0'
            if not (folder/'prediction_freeze.json').exists():continue
            for task,n in [('daily','frozen_station_days.parquet'),('monthly','frozen_station_months.parquet')]:
                path=folder/n
                if not path.exists():continue
                p=pd.read_parquet(path)
                if task=='monthly':p['date']=pd.to_datetime(dict(year=p.year,month=p.month,day=np.ones(len(p),int)))
                q=d.labels(task);q=q[q.year.eq(year)].merge(p[['station_key','date','prediction_mg_l']],on=['station_key','date'],validate='one_to_one');q=q.rename(columns={'prediction_mg_l':'prediction'})
                jid=f'{fold}_{task}_reference_{model}_T0_frozen';dest=ROOT/'jobs'/jid;dest.mkdir(parents=True,exist_ok=True);q.to_parquet(dest/'prediction.parquet',index=False);write(dest/'result.json',dict(job_id=jid,status='complete',role='descriptive historical reference only',source=str(path),source_sha256=sha(path),inputs='different source/history/process objective; not a pure structural or algorithm effect',selection='frozen entry0, no new evaluation-based selection'));receipts.append(dict(path=str(path),sha256=sha(path)))
    path=ROOT.parent/'20260916_2/outputs/diagnostic/T24_tree_A0_1729/predictions_2024.parquet'
    q=d.labels('daily');q=q[q.year.eq(2024)].merge(pd.read_parquet(path).rename(columns={'p':'prediction'}),on=['station_key','date'],validate='one_to_one');jid='F24_daily_reference_old_tree_A0';folder=ROOT/'jobs'/jid;folder.mkdir(exist_ok=True);q.to_parquet(folder/'prediction.parquet',index=False);write(folder/'result.json',dict(job_id=jid,status='complete',role='descriptive legacy comparator',source=str(path),source_sha256=sha(path),D29_derived_features=True,not_pure_ML=True,selection_eligible=False));receipts.append(dict(path=str(path),sha256=sha(path)))
    write(ROOT/'evidence/historical_reference_identity.json',receipts)
if __name__=='__main__':main()
