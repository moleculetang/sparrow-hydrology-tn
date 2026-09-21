"""No fitting. Finite forward matrix and label-free dynamic root solves."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from kernel import scan,tags,transfer_total,probabilities
from scipy.optimize import brentq
PRO=read(R/'data/protocol.json')
assert sha(R/'data/protocol.json')==(R/'data/protocol.sha256').read_text().strip()
NAMES=read(R/'data/hf_stations.json')['stations']
def qmake(c,g,k,clim=False):
 w=c['W']
 if clim:
  if 'clim' not in c:
   md=c['d'].dates.strftime('%m-%d');frame=pd.DataFrame(w[c['ref']],index=md[c['ref']]);table=frame.groupby(level=0).mean()
   c['clim']=np.ascontiguousarray(table.loc[md].to_numpy())
  w=c['clim']
 return probabilities(w,g,k)
def runland(c,q):
 m=c['model'];f=c['frac'];return scan(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],q)
def transfer(c,q):
 return float(transfer_total(c['model'].inp,c['model'].demand,c['s'],c['frac']['gu'],q,int(c['ref'].sum())).sum())
def modelmean(c,q):
 a=runland(c,q);mass,w,_=boundary(c,a[0]+a[1]);sel=c['meta'].station_key.isin(NAMES).to_numpy()
 assert sel.sum()==15
 return float((1000*mass[c['ref']][:,sel]/w[c['ref']][:,sel]).mean(axis=0).mean())
def root(c,name,target,g,clim=False,level=False):
 path=R/'outputs'/f'{name}_root.json'
 if path.exists():
  rec=read(path)
  if rec['status']=='PASS':return rec['k']
  raise ValueError('PREVIOUS_ROOT_FAILED '+name)
 memo={}
 def fun(k):
  key=float(k)
  if key in memo:return memo[key]
  total=sum(1 for _ in (R/'logs/roots.jsonl').open()) if (R/'logs/roots.jsonl').exists() else 0
  if total>=PRO['root_budget']:raise ValueError('ROOT_BUDGET_EXHAUSTED')
  log('roots.jsonl',dict(event='evaluation_start',name=name,index=total,k=key))
  q=qmake(c,g,key,clim);v=modelmean(c,q) if level else transfer(c,q)
  err=(v-target)/target;memo[key]=err
  log('root_results.jsonl',dict(name=name,k=key,value=v,target=target,relative_error=err));return err
 grid=np.logspace(-8,2,PRO['root_grid_points']);values=[fun(k) for k in grid]
 intervals=[(grid[i],grid[i+1]) for i in range(len(grid)-1) if values[i]*values[i+1]<0]
 exact=[grid[i] for i,v in enumerate(values) if v==0]
 if len(intervals)+len(exact)!=1:
  put(path,dict(status='BLOCKED',reason='NO_UNIQUE_BRACKET',intervals=intervals,exact=exact,target=target));raise ValueError('NO_UNIQUE_ROOT '+name)
 k=float(exact[0]) if exact else float(brentq(fun,*intervals[0],xtol=1e-15,rtol=1e-12,maxiter=60))
 error=fun(k);assert abs(error)<=1e-8,(name,error)
 put(path,dict(status='PASS',k=k,target=target,relative_error=error,grid=grid.tolist(),grid_errors=values,full_recursive=True,evaluations=len(memo)))
 return k
def save_output(c,arm,a,q=None):
 name=arm['id'];out=R/'outputs'/name;out.mkdir(exist_ok=True)
 local=a[0]+a[1];mass,w,river=boundary(c,local);conc=1000*mass/w
 m=c['model'];d=c['d'];sel=c['meta'].station_key.isin(NAMES).to_numpy();refmean=float(conc[c['ref']][:,sel].mean(axis=0).mean())
 net=float(local.sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum());scale=float(local.sum())
 gate=dict(network_balance_kg=net,network_scale_kg=scale,network_pass=bool(abs(net)<=scale*1e-10))
 if q is not None:
  stock=a[2]+a[3]+a[4];bal=m.inp-a[5]-a[6]-local-np.diff(stock,axis=0,prepend=np.zeros_like(stock[:1]))
  gate.update(local_balance_max_kg=float(abs(bal).max()),nonnegative_min_kg=float(a[:9].min()),uptake_excess_kg=float(np.max(a[5]-m.demand)))
  rr=np.asarray(m.data.pilot_indices,int);f=c['frac'];tag=tags(m.tag_inputs,m.tag_demand,c['s'][rr],np.ascontiguousarray(f['gu'][:,rr]),np.ascontiguousarray(f['phi_f'][:,rr]),np.ascontiguousarray(f['gs'][:,rr]),np.ascontiguousarray(q[:,rr]))
  # Inherited hard scope: total M plus slow pool and flux channels; extra split diagnostics reported separately.
  diffs={n:float(abs(tag[k].sum(-1)-a[k][:,rr]).max()) for k,n in enumerate(['fast','slow','legacy','mobile','L','uptake','loss','transfer'])}
  diffs['M']=float(abs((tag[2]+tag[3]).sum(-1)-(a[2]+a[3])[:,rr]).max())
  gate['source_sum_errors']=diffs;gate['tag_pass']=max(diffs[n] for n in ('fast','slow','M','L','uptake','loss'))<=1e-6
  gate['pass']=bool(gate['network_pass'] and gate['local_balance_max_kg']<=1e-6 and gate['nonnegative_min_kg']>=-1e-7 and gate['uptake_excess_kg']<=1e-7 and gate['tag_pass'])
  np.savez_compressed(out/'source_tags.npz',values=tag,reaches=rr+1)
  np.savez_compressed(out/'land_history.npz',fast=a[0],slow=a[1],legacy=a[2],mobile=a[3],lower=a[4],uptake=a[5],loss=a[6],transfer=a[7],mobile_pre=a[8])
  np.save(out/'q.npy',q,allow_pickle=False)
 else:
  ledger=m.ledger(c['x']);gate.update(local_balance_max_kg=float(ledger['local_balance_max_kg']),source_sum_errors=ledger.get('source_label_sum_errors',{}))
  gate['pass']=bool(gate['network_pass'] and gate['local_balance_max_kg']<=1e-6 and max(gate['source_sum_errors'].values(),default=0)<=1e-6)
  np.savez_compressed(out/'land_history.npz',fast=ledger['fast'],slow=ledger['slow'],M=ledger['M'],lower=ledger['L'],uptake=ledger['uptake'],loss=ledger['mineral_loss'])
 np.savez_compressed(out/'river_history.npz',terminal=river['terminal'],stocks=river['stocks'],loss=river['channel_removed'])
 dates=d.dates;stations=c['meta'].station_key.to_numpy();ns=len(stations)
 ev=dates.year>=2021
 df=pd.DataFrame(dict(date=np.repeat(dates[ev].to_numpy(),ns),station_key=np.tile(stations,int(ev.sum())),mass_kg=mass[ev].reshape(-1),water_m3=w[ev].reshape(-1),p=conc[ev].reshape(-1)))
 df.to_parquet(out/'daily_station.parquet',index=False)
 hf=pd.DataFrame(dict(date=np.repeat(dates.to_numpy(),int(sel.sum())),station_key=np.tile(stations[sel],len(dates)),mass_kg=mass[:,sel].reshape(-1),water_m3=w[:,sel].reshape(-1),p=conc[:,sel].reshape(-1)))
 hf.to_parquet(out/'daily_HF_history.parquet',index=False)
 # Daily all-reach local flux is an internal diagnostic, not a station concentration.
 yearly=[]
 for year in range(1961,2025):
  ix=dates.year==year
  yearly.append(dict(year=year,fast=float(a[0,ix].sum()),slow=float(a[1,ix].sum()),demand=float(m.demand[ix].sum()),uptake=float((a[5] if q is not None else ledger['uptake'])[ix].sum()),loss=float((a[6] if q is not None else ledger['mineral_loss'])[ix].sum())))
 pd.DataFrame(yearly).to_csv(out/'annual_ledger.csv',index=False)
 summary=dict(arm=arm,gate=gate,reference_station_mean=refmean,reference_transfer=None if q is None else float(a[7,c['ref']].sum()),runtime_status='COMPLETE',physical_status='PASS' if gate['pass'] else 'BLOCKED')
 put(out/'summary.json',summary);return summary
def main(hydro):
 lock=R/'work'/f'{hydro}.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  c=build(hydro);np.save(R/'data'/f'wetness_{hydro}.npy',c['W'],allow_pickle=False)
  put(R/'data'/f'stations_{hydro}.json',c['meta'].to_dict('records'))
  baseline=qmake(c,0,1/200);target=transfer(c,baseline);levelk=None;leveltarget=None
  for arm in [a for a in PRO['arms'] if a['hydro']==hydro]:
   out=R/'outputs'/arm['id'];out.mkdir(exist_ok=True)
   if (out/'summary.json').exists():continue
   cpu,ram=resource('before_'+arm['id'])
   while cpu>=90 or ram>=90:
    time.sleep(30);cpu,ram=resource('resource_wait')
    if cpu<85 and ram<85:break
   log('tasks.jsonl',dict(event='start',arm=arm['id']))
   try:
    if arm['kind']=='D29':
     import closures
     with torch.no_grad():fast,slow=closures.Transport.apply(torch.tensor(c['h']),torch.tensor(c['s']),torch.tensor(c['f']),torch.tensor(c['k']),c['model'])
     a=np.stack([fast.numpy(),slow.numpy()]);result=save_output(c,arm,a)
    else:
     if arm['kind']=='Q1':q=np.ones_like(c['W']);k=None
     elif arm['kind']=='level':
      if levelk is None:
       lev=read(R/'outputs'/f'{hydro}_D29/summary.json')['reference_station_mean']
       levelk=root(c,hydro+'_LEVEL_MEAN',lev,0,level=True);leveltarget=transfer(c,qmake(c,0,levelk))
      k=levelk if arm['gamma']==0 else root(c,arm['id'],leveltarget,arm['gamma']);q=qmake(c,arm['gamma'],k)
     else:
      k=1/200 if arm['gamma']==0 else root(c,arm['id'],target,arm['gamma'],arm['kind']=='clim');q=qmake(c,arm['gamma'],k,arm['kind']=='clim')
     a=runland(c,q);result=save_output(c,dict(arm,k=k),a,q);del a,q
    log('tasks.jsonl',dict(event='exit',arm=arm['id'],status=result['physical_status']))
   except Exception as e:
    import traceback
    put(out/'failure.json',dict(reason=str(e),traceback=traceback.format_exc()));log('tasks.jsonl',dict(event='exit',arm=arm['id'],status='FAILED',reason=str(e)))
   resource('after_'+arm['id'])
 finally:lock.unlink()
if __name__=='__main__':main(sys.argv[1])
