"""Events are defined before concentration evaluation, from H1 water only."""
import numpy as np,pandas as pd
def define_events(dates,water,names):
 rows=[];thresholds={};ref=dates.year<=2020
 for j,s in enumerate(names):
  threshold=float(np.quantile(water[ref,j],.9));thresholds[s]=threshold
  high=water[:,j]>threshold;edges=np.diff(np.r_[False,high,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
  for rank,(a,b) in enumerate(zip(starts,ends)):
   if dates[a].year<2021 or dates[b].year>2024:continue
   rows.append(dict(station_key=s,event_rank=rank,start=dates[a],end=dates[b],background_start=dates[a]-pd.Timedelta(days=7),event_days=b-a+1,threshold=threshold,water_peak=float(water[a:b+1,j].max()),period='2024' if dates[a].year==2024 else '2021-2023',cross_period=dates[a].year<2024<=dates[b].year))
 return pd.DataFrame(rows),thresholds
def adjacent_pairs(events):
 result=[]
 for s,g in events.sort_values(['station_key','event_rank']).groupby('station_key'):
  records=g.to_dict('records')
  for a,b in zip(records,records[1:]):
   gap=(pd.Timestamp(b['start'])-pd.Timestamp(a['end'])).days
   if b['event_rank']==a['event_rank']+1 and 0<gap<=30 and a['period']==b['period']:
    result.append(dict(station_key=s,previous_rank=a['event_rank'],event_rank=b['event_rank'],gap_days=gap,period=b['period']))
 return pd.DataFrame(result)
def fixtures():
 ds=pd.date_range('2023-12-20',periods=80);w=np.zeros((80,1));w[20:22]=3;w[30:32]=3
 d=pd.DataFrame([dict(station_key='s',event_rank=i,start=pd.Timestamp('2024-01-01')+pd.Timedelta(days=i*5),end=pd.Timestamp('2024-01-02')+pd.Timedelta(days=i*5),period='2024') for i in (0,2,3)])
 a=adjacent_pairs(d);assert len(a)==1 and a.iloc[0].previous_rank==2
 x=np.array([.1,.2,.6]);assert np.median(x)==np.median(np.tile(x,2))
 leap=pd.date_range('2024-02-28',periods=3);assert leap[1].day==29
 return dict(nonadjacent_rejected=True,duplicated_sample_median_invariant=True,leap_day=True)

if __name__=='__main__':
 import os
 os.environ['WET_PHYSICS_ONLY']='1'
 from runtime import *
 c=build('H1');m=c['model'];mm=c['mm'];d=c['d'];names=read(R/'data/hf_stations.json')['stations']
 ix=[int(np.flatnonzero(c['meta'].station_key.eq(s))[0]) for s in names];ri=mm['ri'].numpy()[ix];f=mm['f'].numpy()[ix]
 assert (mm['boundary_code'].numpy()[ix]==0).all(),'HF_BOUNDARY_NOT_ORDINARY'
 water=m.daily_water['inlet'][:,ri]+(d.fast_water[:,ri]+d.slow_water[:,ri])*f
 assert np.isfinite(water).all() and water.min()>0
 events,thresholds=define_events(d.dates,water,names)
 events.to_parquet(R/'data/events_frozen.parquet',index=False)
 put(R/'data/events_freeze.json',dict(sha256=sha(R/'data/events_frozen.parquet'),thresholds=thresholds,count=len(events),tests=fixtures(),input_hydro='H1',no_TN=True))
 print('EVENTS_FROZEN',len(events))
