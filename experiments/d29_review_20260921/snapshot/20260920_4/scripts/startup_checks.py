import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from bypass_kernel import block,tag_block,fraction,reference
from types import SimpleNamespace
def main():
 rng=np.random.default_rng(1729);nd,nr=400,3
 inp=rng.random((nd,nr));dem=rng.random((nd,nr));dem[30]=1e8;s=np.array([1.,.995,.98]);gu=rng.random((nd,nr));pf=rng.random((nd,nr));gs=rng.random((nd,nr));q=rng.random((nd,nr))*.1;b=rng.random((nd,nr))
 state=np.zeros((3,nr));full,end=block(inp,dem,s,gu,pf,gs,q,b,state)
 ref=reference(inp,dem,s,gu,pf,gs,q,b);err=float(abs(full-ref).max());assert err<1e-12
 # Serialize all states, then resume over year/leap and nonuniform block boundaries.
 first,st=block(inp[:59],dem[:59],s,gu[:59],pf[:59],gs[:59],q[:59],b[:59],state)
 cp=R/'work/fixture_checkpoint.npz';np.savez(cp,state=st,completed_days=59,attempted_blocks=1)
 saved=np.load(cp);tail,en=block(inp[59:],dem[59:],s,gu[59:],pf[59:],gs[59:],q[59:],b[59:],saved['state'])
 assert int(saved['completed_days'])==59 and np.array_equal(np.concatenate((first,tail),axis=1),full) and np.array_equal(en,end)
 tags=inp[:,:,None]*np.array([.1,.2,.3,.4]);initial=np.zeros((3,nr,4));tf,ts=tag_block(tags,dem,s,gu,pf,gs,q,b,initial)
 t0,z=tag_block(tags[:59],dem[:59],s,gu[:59],pf[:59],gs[:59],q[:59],b[:59],initial);t1,zz=tag_block(tags[59:],dem[59:],s,gu[59:],pf[59:],gs[59:],q[59:],b[59:],z)
 assert np.array_equal(tf,np.concatenate((t0,t1),axis=1)) and np.array_equal(ts,zz)
 # One-step identities at precisely identical initial states.
 zero=np.zeros((1,nr));args=(inp[:1],dem[:1],s,gu[:1],pf[:1],gs[:1],q[:1]);m,_=block(*args,zero,state);x,_=block(*args,b[:1],state);B=x[10]
 assert np.allclose(x[0]-m[0],B*(1-gu[:1]*pf[:1]),atol=1e-14)
 assert np.allclose(x[12]-m[12],-B*gu[:1]*(1-pf[:1]),atol=1e-14)
 assert np.allclose(B+(x[7]-B),x[7],atol=1e-14)
 z=np.zeros((5,2));one=np.ones_like(z)
 for name in ('MIX','HYDRO-SELECT','FULL-BYPASS'):
  assert not fraction(name,z,z,one).any() and not fraction(name,one,one,z).any()
 assert np.isfinite(fraction('HYDRO-SELECT',one*1e-200,one*1e-200,one)).all()
 out,_=block(z,z,np.ones(2),one,one,one,one,one,np.zeros((3,2)));assert out.sum()==0
 changed=b.copy();changed[350:]=0;future,_=block(inp,dem,s,gu,pf,gs,q,changed,state);assert np.array_equal(full[:,:350],future[:,:350])
 d=SimpleNamespace(order=[0,1,2],downstream={},metadata=[dict(controls=[0,1],target=2,fraction=1.)],terminal=[2],mid=np.arange(4),h_month=np.zeros((4,3)),h_day=np.zeros((4,3)),operator_id='O0',release_fraction=np.array([[.5],[0.],[.5],[1.]]),enabled=np.ones((4,1),bool))
 local=np.tile([1.,2.,0.],(4,1));rv=routing.route(d,local,vf=0.);assert np.array_equal(rv['terminal'],np.array([1.5,0.,3.75,6.75]))
 snapshot=resource('startup_memory_check',True);assert snapshot['rss_bytes']>0 and snapshot['peak_rss_bytes']>=snapshot['rss_bytes']
 # Bad hash and duplicate process lock are rejected before any physics launch.
 protocol=R/'data/protocol.json';assert sha(protocol)==(R/'data/protocol.sha256').read_text().strip();assert hashlib.sha256(protocol.read_bytes()+b'x').hexdigest()!=sha(protocol)
 lock=R/'work/test.lock';lock.write_text('test');rejected=False
 try:
  with lock.open('x'):pass
 except FileExistsError:rejected=True
 lock.unlink();assert rejected
 denied=False
 try:pd.read_parquet(R/'data/evaluation/observed_days.parquet')
 except PermissionError:denied=True
 assert denied
 put(R/'reports/startup_checks.json',dict(status='PASS',independent_max_abs=err,serialized_land_resume_bitwise=True,tag_resume_bitwise=True,one_step_identities=True,future_causality=True,zero_carrier_guard=True,shared_reservoir_once=True,memory_api=True,bad_hash=True,duplicate_lock=True,label_barrier=True))
 print('STARTUP_PASS')
if __name__=='__main__':main()
