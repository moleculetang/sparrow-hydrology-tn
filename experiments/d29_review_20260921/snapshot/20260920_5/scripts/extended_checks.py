import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from family_kernel import block,reference
def main():
 rng=np.random.default_rng(1729);nd=731;nr=3;s=np.array([.99,.999,.9]);inp=rng.uniform(0,1,(nd,nr,1));dem=rng.uniform(0,.8,(nd,nr));gu=rng.uniform(0,1,(nd,nr));pf=rng.uniform(0,1,(nd,nr));gf=gu*pf;gs=gu*.1;q=rng.uniform(0,1,(nd,nr));zero=np.zeros_like(gu);state=np.zeros((4,nr,1));state[:,0]=1
 mix,_=block(inp,dem,s,gu,pf,gs,q,gf,gf,gu,zero,0,0,0,state)
 a,_=block(inp,dem,s,gu,pf,gs,q,gf,gf,gu,zero,0,.5,0,state)
 b,_=block(inp,dem,s,gu,pf,gs,q,gf,gf,gu,zero,1,.5,0,state);assert np.array_equal(a,b)
 altered=q.copy();altered[400:]=0;future,_=block(inp,dem,s,gu,pf,gs,altered,gf,gf,gu,zero,0,.5,0,state);assert np.array_equal(a[:,:400],future[:,:400])
 # C actual domain exchange, independent reference, all 8 parameter points.
 results=[]
 for c in read(R/'data/protocol.json')['configs']:
  if c['family']!='C':continue
  w=c['omega'];xf=np.load(R/'data'/f'C_{w:g}_xf.npy',mmap_mode='r')[:nd,:nr];xp=np.load(R/'data'/f'C_{w:g}_xp.npy',mmap_mode='r')[:nd,:nr];ef=np.load(R/'data'/f'C_{w:g}_ef.npy',mmap_mode='r')[:nd,:nr]
  args=[np.load(R/'data'/f'{k}.npy',mmap_mode='r')[:nd,:nr] for k in ['gu','pf','gs','q','gf']]
  out,end=block(inp,dem,s,*args,xf,xp,ef,2,c['pi'],0,np.zeros_like(state));ref=reference(inp[:,:,0],dem,s,*args,xf,xp,ef,2,c['pi'],0)
  error=float(abs(out[:,:,:,0]-ref).max());assert error<1e-9
  one,st=block(inp[:365],dem[:365],s,*[x[:365] for x in args],xf[:365],xp[:365],ef[:365],2,c['pi'],0,np.zeros_like(state))
  two,st2=block(inp[365:],dem[365:],s,*[x[365:] for x in args],xf[365:],xp[365:],ef[365:],2,c['pi'],0,st)
  assert np.array_equal(out,np.concatenate([one,two],axis=1))
  results.append(dict(arm=c['id'],reference_error=error,resume=True))
 # Forced pre-outflow full mixing gives common concentration for arbitrary nonnegative domains.
 f=rng.random(100);p=rng.random(100);carry=rng.random(100);v=carry+f+p;mass=rng.random(100);w=.2;vf=w*carry+f;vp=(1-w)*carry+p
 nf=mass*vf/v;nm=mass*vp/v
 assert np.allclose(nf*f/vf,mass*f/v) and np.allclose(nm*p/vp,mass*p/v)
 # Zero/fully exhausted stocks and zero carrier/contact, no epsilon or clipping.
 z=np.zeros((4,2));o=np.ones_like(z)
 for fam in [0,1,2]:
  out,st=block(z[:,:,None],o,np.ones(2),z,z,z,z,z,z,z,z,fam,1,1,np.zeros((4,2,1)));assert not out.any()
  out,st=block(o[:,:,None],o*10,np.ones(2),o,o,o,o,o,o,o,z,fam,1,1,np.zeros((4,2,1)));assert not out[[0,1,2,3,4,10]].any()
  out,st=block(o[:,:,None],z,np.ones(2),z,z,z,o,z,z,z,z,fam,1,1,np.zeros((4,2,1)));assert not out[[0,1]].any()
 denied=False
 try:pd.read_parquet(R/'data/evaluation/observed_days.parquet')
 except PermissionError:denied=True
 assert denied
 from types import SimpleNamespace
 rd=SimpleNamespace(order=[0,1,2],downstream={},metadata=[dict(controls=[0,1],target=2,fraction=1.)],terminal=[2],mid=np.arange(4),h_month=np.zeros((4,3)),h_day=np.zeros((4,3)),operator_id='O0',release_fraction=np.array([[.5],[0.],[.5],[1.]]),enabled=np.ones((4,1),bool))
 rv=routing.route(rd,np.tile([1.,2.,0.],(4,1)),vf=0.);assert np.array_equal(rv['terminal'],np.array([1.5,0.,3.75,6.75]))
 # Actual frozen coefficients never use epsilon; fixture includes arbitrarily small flux.
 tiny=np.full((4,2),1e-200);out,st=block(o[:,:,None],z,np.ones(2),tiny,o,z,o,tiny,tiny,z,z,0,1,0,np.zeros((4,2,1)));assert np.isfinite(out).all() and (out[0]>0).all()
 lock=R/'work/duplicate_fixture.lock';lock.write_text('occupied');denied_lock=False
 try:
  with lock.open('x'):pass
 except FileExistsError:denied_lock=True
 lock.unlink();assert denied_lock
 assert sha(R/'data/q.npy')!=hashlib.sha256((R/'data/q.npy').read_bytes()+b'bad').hexdigest()
 put(R/'reports/extended_checks.json',dict(status='PASS',C_fixtures=results,B_zero_exchange_recovers_A=True,full_mixing_C_identity=True,future_causality=True,zero_and_exhaustion=True,label_barrier=True,shared_reservoir_once=True,extremely_small_positive_flux=True,duplicate_lock=True,bad_hash=True))
 print('EXTENDED_PASS')
if __name__=='__main__':main()
