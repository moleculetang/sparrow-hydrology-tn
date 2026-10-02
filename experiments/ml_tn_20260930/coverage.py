"""All certified observations, calendar support and frozen spatial exclusions."""
import pandas as pd
from mltn.common import ROOT,write
from mltn.data import Inputs

def main():
    d=Inputs();out=ROOT/'outputs/coverage';out.mkdir(exist_ok=True,parents=True);summary=[];station=[];blocks=[]
    for task in ['monthly','daily']:
        q=d.labels(task)
        q.groupby('year').agg(accepted_rows=('date','size'),accepted_stations=('station_key','nunique')).assign(task=task).reset_index().to_csv(out/f'{task}_annual_coverage.csv',index=False,encoding='utf-8-sig')
        g=q.groupby(['station_key','year']).agg(rows=('date','size'),months=('month','nunique')).reset_index();g['task']=task;station.append(g)
        summary.append(dict(task=task,rows=len(q),stations=q.station_key.nunique(),first=str(q.date.min()),last=str(q.date.max())))
        for block,v in d.blocks.items():
            for role,keys in [('held',v['held_stations']),('buffer',v['buffer_stations'])]:
                for key in keys:
                    sub=q[q.station_key.eq(key)];blocks.append(dict(block=block,task=task,role=role,station_key=key,training_rows_removed=int((sub.year<=2023).sum()),eval2023_rows=int(sub.year.eq(2023).sum()),eval2024_rows=int(sub.year.eq(2024).sum())))
    pd.concat(station).to_csv(out/'station_year_support.csv',index=False,encoding='utf-8-sig');pd.DataFrame(blocks).to_csv(out/'spatial_removed_and_held_support.csv',index=False,encoding='utf-8-sig')
    reads=pd.read_parquet(ROOT/'data/hf_readings.parquet');reads.groupby(['year','eligible']).size().rename('count').reset_index().to_csv(out/'four_hour_read_coverage.csv',index=False,encoding='utf-8-sig')
    write(out/'identity.json',dict(support=summary,monthly_definition='automatic valid-reading arithmetic average confirmed by user; equal-day model approximation without complete official counts',NH4_DO='excluded',spatial='held and downstream buffer removed for every training year; no residual-selected stations',new_sites=0))
if __name__=='__main__':main()
