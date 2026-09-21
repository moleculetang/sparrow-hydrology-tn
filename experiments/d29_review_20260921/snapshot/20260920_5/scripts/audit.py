"""Independent full-history arithmetic audit; does not call the production forward worker."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from family_kernel import reference,block,CHANNELS
def main():
 arrays={k:np.load(R/'data'/f'{k}.npy',mmap_mode='r') for k in ['inp','demand','s','gu','pf','gs','q','gf','tag_inputs','tag_demand','pilot_indices']};a=arrays;nd,nr=a['inp'].shape;args=[a[k] for k in ['gu','pf','gs','q','gf']];zero=np.zeros((nd,nr));records=[]
 for conf in read(R/'data/protocol.json')['configs']:
  name=conf['id'];out=R/'outputs'/name
  if conf['family']=='REFERENCE':continue
  assert (out/'summary.json').exists(),name
  fam={'MIX':0,'A':0,'B':1,'C':2}[conf['family']]
  extra=[np.load(R/'data'/f'C_{conf["omega"]:g}_{k}.npy',mmap_mode='r') for k in ['xf','xp','ef']] if fam==2 else [a['gf'],a['gu'],zero]
  saved=np.load(out/'land_history.npy',mmap_mode='r');ref=reference(a['inp'],a['demand'],a['s'],*args,*extra,fam,conf['pi'],conf['rho'])
  errs={CHANNELS[i]:float(abs(ref[i]-saved[i]).max()) for i in range(21)};assert max(errs.values())<=1e-6,(name,errs)
  state=saved[2]+saved[3]+saved[4]+saved[10];bal=a['inp']-saved[5]-saved[6]-saved[0]-saved[1]-np.diff(state,axis=0,prepend=np.zeros_like(state[:1]));assert abs(bal).max()<=1e-6
  # Saved block covering 1964 leap day; replay actual checkpoint rather than synthetic state.
  start=1098;stop=1464;snap=np.load(out/f'state_{start:05d}.npz');val,end=block(a['inp'][start:stop,:,None],a['demand'][start:stop],a['s'],*[x[start:stop] for x in args+extra],fam,conf['pi'],conf['rho'],snap['land'])
  assert np.array_equal(val[:,:,:,0],saved[:,start:stop])
  rr=a['pilot_indices'].astype(int);tv,te=block(a['tag_inputs'][start:stop],a['tag_demand'][start:stop],a['s'][rr],*[np.ascontiguousarray(x[start:stop,rr]) for x in args+extra],fam,conf['pi'],conf['rho'],snap['tags'])
  tags=np.load(out/'source_tags.npy',mmap_mode='r');assert np.array_equal(tv,tags[:,start:stop]);cp=read(out/'checkpoint.json');assert cp['completed_days']==nd and len(cp['chunks'])==64
  for ch in cp['chunks']:
   i,j=ch['start'],ch['stop'];assert hashlib.sha256(saved[:,i:j].tobytes()+tags[:,i:j].tobytes()).hexdigest()==ch['sha256']
  record=dict(arm=name,reference_errors_kg=errs,local_balance_max_kg=float(abs(bal).max()),actual_land_resume_bitwise=True,actual_source_resume_bitwise=True,all_chunk_hashes=True,completed_days=nd)
  put(R/'reports'/f'independent_{name}.json',record);records.append(record);del ref,state,bal,val,tv
  resource('independent_audit_'+name,True)
 put(R/'reports/independent_physical_audit.json',records);print('FULL_INDEPENDENT_AUDIT_PASS',len(records))
if __name__=='__main__':main()
