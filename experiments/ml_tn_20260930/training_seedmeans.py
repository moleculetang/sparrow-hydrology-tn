"""Actual three-seed training prediction mean, kept separate from holdout metrics."""
import re
import numpy as np,pandas as pd
from mltn.common import ROOT,read,write,sha
from mltn.metrics import station_table

def main():
    roles=read(ROOT/'outputs/result_roles.json');groups={};out=ROOT/'outputs/training_seedmeans';out.mkdir(exist_ok=True);receipts=[];tables=[]
    for name in roles['include']:
        match=re.search(r'_s(1729|1730|1731)(?=_B|_lead|$)',name)
        if match is None:continue
        key=name[:match.start()]+'_seedmean'+name[match.end():]
        folder=ROOT/'jobs'/name
        files=[('monthly','training_prediction_monthly.parquet'),('daily','training_prediction_daily_hf.parquet')] if name.startswith('joint_') else [('monthly' if '_monthly_' in name else 'daily','training_prediction.parquet')]
        for task,file in files:
            path=folder/file
            if path.exists():groups.setdefault((key,task),[]).append((int(match[1]),path))
    for (key,task),parts in sorted(groups.items()):
        if {seed for seed,_ in parts}!={1729,1730,1731} or len(parts)!=3:continue
        parts=sorted(parts);keys=['station_key','date'];base=pd.read_parquet(parts[0][1]).sort_values(keys).reset_index(drop=True);predictions=[]
        for seed,path in parts:
            q=pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
            assert not q.duplicated(keys).any() and base[keys].equals(q[keys]),'TRAIN_SEED_SUPPORT_MISMATCH'
            np.testing.assert_array_equal(base.tn_mg_l,q.tn_mg_l)
            if task=='daily':np.testing.assert_array_equal(base.read_count,q.read_count)
            assert np.isfinite(q.prediction).all() and q.prediction.ge(0).all()
            predictions.append(q.prediction.to_numpy(float))
        base['prediction']=np.mean(np.stack(predictions),axis=0);path=out/f'{key}_{task}.parquet';base.to_parquet(path,index=False)
        q=base.rename(columns={'tn_mg_l':'observed'});q['date']=pd.to_datetime(q.date)
        for period,g in [('all_training',q),*[(str(y),g) for y,g in q.groupby(q.date.dt.year)]]:
            t=station_table(g,predictions=('prediction',),daily=task=='daily',minimum_coverage=True);t['job']=key;t['task']=task;t['period']=period;tables.append(t)
        receipts.append(dict(configuration=key,task=task,prediction_file=path.relative_to(ROOT).as_posix(),prediction_sha256=sha(path),parents=[dict(file=p.relative_to(ROOT).as_posix(),sha256=sha(p),seed=seed) for seed,p in parts],rows=len(base)))
    assert receipts,'NO_COMPLETE_THREE_SEED_TRAINING_GROUP'
    pd.concat(tables,ignore_index=True).to_csv(ROOT/'outputs/evaluation/training_seedmean_station_metrics.csv',index=False,encoding='utf-8-sig')
    write(out/'manifest.json',dict(records=receipts,rule='mean of predictions before computing metrics, not average NSE; all three seeds exact station/date/label/read-count support',role='training reconstruction only; no prediction validation or new fit'))

if __name__=='__main__':main()
