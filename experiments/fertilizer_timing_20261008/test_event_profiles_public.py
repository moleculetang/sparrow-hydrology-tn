from pathlib import Path
import sys,calendar
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'source/20261008_1'))
from timing.profiles import profile,monthly_partition,event_window
def main():
 seasons=[dict(share=1.,plant='03-01',harvest='09-01',events=[dict(share=.5,date='03-01',sigma_days=3.),dict(share=.5,date='05-01',sigma_days=3.)])]
 count=0
 for year in (2023,2024):
  w,d,g,o=profile(year,seasons);months=np.repeat(np.arange(1,13),[calendar.monthrange(year,m)[1] for m in range(1,13)])
  pi,c1,q=monthly_partition(w,months)
  assert abs(w.sum()-1)<1e-12 and abs(c1.sum()-1)<1e-12
  for m in range(1,13):assert abs(w[months==m].sum()-c1[months==m].sum())<1e-12
  for h in (1e-3,1e-4,1e-5):
   slope=(profile(year,seasons,h)[0]-profile(year,seasons,-h)[0])/(2*h)
   assert np.max(abs(slope-d))<1e-8
  restricted=event_window(len(w),60.,2.,50,70);p,u,q=monthly_partition(restricted,months)
  for m in np.flatnonzero(p==0)+1:assert np.isnan(q[months==m]).all() and np.all(u[months==m]==0)
  assert len(w)==(366 if year==2024 else 365);count+=1
 print('synthetic_event_profile_cases',count)
if __name__=='__main__':main()
