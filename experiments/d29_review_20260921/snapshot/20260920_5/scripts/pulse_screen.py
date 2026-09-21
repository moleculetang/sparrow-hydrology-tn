"""Label-free one-kg newly mobile pulse at every Jan/Apr/Jul/Oct start."""
from runtime import *
def main():
 a={k:np.load(R/'data'/f'{k}.npy',mmap_mode='r') for k in ['s','gu','pf','gs','gf']};nd,nr=a['gu'].shape;dates=pd.date_range('1961-01-01','2024-12-31');starts=np.flatnonzero((dates.day==1)&dates.month.isin([1,4,7,10])&(np.arange(nd)+90<=nd));rows=[]
 for c in read(R/'data/protocol.json')['configs']:
  if c['family']=='REFERENCE':continue
  pi=c['pi'];f=np.full((len(starts),nr),pi,dtype=np.float64);m=np.full_like(f,1-pi);lower=np.zeros_like(f);fast=lower.copy();slow=lower.copy();loss=lower.copy()
  if c['family']=='C':extra={k:np.load(R/'data'/f'C_{c["omega"]:g}_{k}.npy',mmap_mode='r') for k in ['xf','xp','ef']}
  for day in range(90):
   ix=starts+day
   if c['family']=='B':x=.5*c['rho']*(f-m);f-=x;m+=x
   if c['family']=='C':
    e=extra['ef'][ix];x=np.where(e>=0,-e*m,-e*f);f-=x;m+=x;ff=f*extra['xf'][ix];mf=np.zeros_like(f);j=m*extra['xp'][ix]
   else:ff=f*a['gf'][ix];mf=m*a['gu'][ix]*a['pf'][ix];j=m*a['gu'][ix]*(1-a['pf'][ix])
   f-=ff;m-=mf+j;fast+=ff+mf;lower+=j;sl=lower*a['gs'][ix];lower-=sl;slow+=sl;loss+=(f+m)*(1-a['s']);f*=a['s'];m*=a['s']
   assert np.max(abs(f+m+lower+fast+slow+loss-1))<1e-12
   if day+1 in [1,7,30,90]:
    for name,value in [('fast_export',fast),('slow_export',slow),('upper_remaining',f+m),('lower_remaining',lower),('loss',loss)]:
     quant=np.quantile(value,[.1,.5,.9]);rows.append(dict(arm=c['id'],days=day+1,metric=name,p10=quant[0],median=quant[1],p90=quant[2],mean=float(value.mean()),pulse_count=value.size))
 pd.DataFrame(rows).to_csv(R/'reports/pulse_screen.csv',index=False)
 put(R/'reports/pulse_protocol.json',dict(starts=dates[starts].strftime('%Y-%m-%d').tolist(),all_reaches=230,unit='one kg newly mobile',uptake='zero demand diagnostic',loss='inherited s_M',no_TN=True,no_additional_science_configs=True))
 print('PULSE_SCREEN_COMPLETE')
if __name__=='__main__':main()
