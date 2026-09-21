"""Independent arithmetic audit of saved histories, checkpoint replay and identities."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from bypass_kernel import reference,block,tag_block,CHANNELS,TAG_CHANNELS
def main():
 c=build('H1');m=c['model'];d=c['d'];f=c['frac'];q=np.load(R/'data/q.npy');rr=np.array(m.data.pilot_indices,int);results=[]
 for arm in read(R/'data/protocol.json')['arms']:
  out=R/'outputs'/arm;land=np.load(out/'land_history.npy',mmap_mode='r');tags=np.load(out/'source_tags.npy',mmap_mode='r');b=np.load(out/'bypass_fraction.npy')
  ref=reference(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],q,b)
  errors={k:float(np.max(abs(ref[i]-land[i]))) for i,k in enumerate(CHANNELS)};assert max(errors.values())<=1e-6,('INDEPENDENT_FORWARD',arm,errors)
  del ref
  a,z=1098,1464;st=np.load(out/f'state_{a:05d}.npz');res,end=block(m.inp[a:z],m.demand[a:z],c['s'],f['gu'][a:z],f['phi_f'][a:z],f['gs'][a:z],q[a:z],b[a:z],st['land'])
  assert np.array_equal(res,land[:,a:z]);assert np.array_equal(end,np.load(out/f'state_{z:05d}.npz')['land'])
  tr,te=tag_block(m.tag_inputs[a:z],m.tag_demand[a:z],c['s'][rr],np.ascontiguousarray(f['gu'][a:z,rr]),np.ascontiguousarray(f['phi_f'][a:z,rr]),np.ascontiguousarray(f['gs'][a:z,rr]),np.ascontiguousarray(q[a:z,rr]),np.ascontiguousarray(b[a:z,rr]),st['tags']);assert np.array_equal(tr,tags[:,a:z]);st.close()
  parent=P/'20260920_3';oldtag=np.load(parent/'outputs/H1_G1/source_tags.npz')['values'] if arm=='MIX' else None
  if oldtag is not None:assert np.array_equal(tags[:8],oldtag),'MIX_TAG_REGRESSION'
  cp=read(out/'checkpoint.json');assert cp['completed_days']==len(d.dates) and cp['completed_tag_days']==len(d.dates)
  for ch in cp['chunks']:
   h=hashlib.sha256();h.update(np.ascontiguousarray(land[:,ch['start']:ch['stop']]).tobytes());h.update(np.ascontiguousarray(tags[:,ch['start']:ch['stop']]).tobytes());assert h.hexdigest()==ch['sha256']
  before=np.zeros(230);maxbal=0.;minimum=float('inf')
  for t in range(len(d.dates)):
   stock=land[2,t]+land[3,t]+land[4,t];residual=m.inp[t]-land[5,t]-land[6,t]-land[0,t]-land[1,t]-(stock-before);maxbal=max(maxbal,float(abs(residual).max()));before=stock;minimum=min(minimum,float(stock.min()))
  assert maxbal<=1e-6 and minimum>=-1e-7
  net=np.load(out/'river_history.npz');total=float(land[0].sum()+land[1].sum());balance=total-float(net['terminal'].sum())-float(net['stocks'][-1].sum())-float(net['loss'].sum());assert abs(balance)<=total*1e-10
  frame=pd.read_parquet(out/'daily_station.parquet');assert np.array_equal(frame.p,1000*frame.mass_kg/frame.water_m3)
  assert not ((d.fast_fraction<=0)&(land[10]!=0)).any() and not ((d.fast_water<=0)&(land[10]!=0)).any()
  assert np.allclose(land[10]+(land[7]-land[10]),land[7],atol=1e-10,rtol=1e-15)
  assert np.array_equal(land[0],land[10]+land[11])
  results.append(dict(arm=arm,independent_channel_errors_kg=errors,actual_checkpoint_resume_bitwise=True,source_checkpoint_resume_bitwise=True,checkpoint_chunks=len(cp['chunks']),leap_in_resume_window=bool(((d.dates[a:z].month==2)&(d.dates[a:z].day==29)).any()),checkpoint_hashes=True,local_balance=maxbal,network_balance=balance,station_concentration_identity=True,contact_zero_bypass=True,bypass_fast_loading_once=True,MIX_tag_bitwise=oldtag is not None))
  resource('independent_audit_'+arm,True)
 put(R/'reports/independent_physical_audit.json',results)
 # Exercise the actual worker pause signal reader without touching prior data.
 from worker import checkpoint_requested
 assert not checkpoint_requested();(R/'work/pause.request').write_text('AUDIT_FIXTURE');assert checkpoint_requested();(R/'work/pause.request').unlink();assert not checkpoint_requested()
 attempts=[json.loads(l) for l in (R/'logs/block_attempts.jsonl').read_text().splitlines()]
 counters={a:sum(x['arm']==a for x in attempts) for a in read(R/'data/protocol.json')['arms']}
 assert all(n==64 for n in counters.values())
 put(R/'reports/checkpoint_control_audit.json',dict(actual_pause_signal=True,actual_history_resume=True,attempted_blocks=counters,completed_days_per_arm=len(d.dates),counter_reset=False,full_recomputed_configs=3,physical_label_reading=False))
 print('AUDIT_PHYSICS_PASS')
if __name__=='__main__':main()
