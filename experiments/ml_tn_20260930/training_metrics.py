"""Training reconstruction separated from holdouts, by year and station."""
import pandas as pd
from mltn.common import ROOT,read,write
from mltn.metrics import station_table

def main():
    roles=read(ROOT/'outputs/result_roles.json');rows=[]
    for name in roles['include']:
        folder=ROOT/'jobs'/name
        files=[('monthly','training_prediction_monthly.parquet'),('daily','training_prediction_daily_hf.parquet')] if name.startswith('joint_') else [('monthly' if '_monthly_' in name else 'daily','training_prediction.parquet')]
        for task,file in files:
            if not (folder/file).exists():continue
            q=pd.read_parquet(folder/file).rename(columns={'tn_mg_l':'observed'});q['date']=pd.to_datetime(q.date)
            if q.empty:continue
            for period,g in [('all_training',q),*[(str(y),g) for y,g in q.groupby(q.date.dt.year)]]:
                t=station_table(g,predictions=('prediction',),daily=task=='daily',minimum_coverage=True);t['job']=name;t['task']=task;t['period']=period;rows.append(t)
    out=ROOT/'outputs/evaluation';out.mkdir(exist_ok=True,parents=True)
    if rows:pd.concat(rows,ignore_index=True).to_csv(out/'training_station_metrics.csv',index=False,encoding='utf-8-sig')
    write(out/'training_metrics_receipt.json',dict(rows=sum(len(t) for t in rows),interpretation='training reconstruction; not prediction or spatial holdout performance'))
if __name__=='__main__':main()
