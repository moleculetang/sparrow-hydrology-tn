import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from kernel import scan,tags,transfer_total,probabilities
import closures_dp2 as parent
import closures_dp as immediate

def independent(inp,demand,s,gu,pf,gs,q):
 nd,nr=inp.shape;out=np.zeros((10,nd,nr));L=np.zeros(nr);A=np.zeros(nr);B=np.zeros(nr)
 for t in range(nd):
  oldA=A+inp[t];total=oldA+B;u=np.minimum(total,demand[t]);share=np.divide(oldA,total,out=np.zeros(nr),where=total>0)
  a=oldA-u*share;b=B-(u-u*share);T=q[t]*a;a-=T;b+=T
  export=b*gu[t];f=export*pf[t];lp=L+export*(1-pf[t]);slow=lp*gs[t]
  loss=(a+b-export)*(1-s);A=a*s;B=(b-export)*s;L=lp-slow
  out[:,t]=[f,slow,A,B,L,u,loss,T,b,total-u]
 return out
def main():
 rng=np.random.default_rng(1729);n=370;r=3
 inp=rng.random((n,r))*2;inp[10:20]=0;dem=rng.random((n,r))*3;dem[50]=1e8
 s=np.array([1.,.99,.9]);gu=rng.random((n,r));pf=rng.random((n,r));gs=rng.random((n,r));gu[:5]=0
 W=rng.random((n,r));W[0]=0;q=probabilities(W,1,.05)
 a=scan(inp,dem,s,gu,pf,gs,q);b=independent(inp,dem,s,gu,pf,gs,q)
 assert np.max(abs(a-b))<1e-12
 assert (q[0]==0).all() and a[7,1:5].sum()>0
 assert np.array_equal(probabilities(W,0,.005),probabilities(W*.1,0,.005))
 for gamma in (.5,1,2,4):assert np.allclose(probabilities(np.full_like(W,.4),gamma,.005/.4**gamma),probabilities(W,0,.005),rtol=1e-15,atol=0)
 assert np.allclose(transfer_total(inp,dem,s,gu,q,200),a[7,:200].sum(0),rtol=1e-14)
 z=np.zeros_like(inp);assert scan(z,z,s,gu,pf,gs,q).sum()==0
 c=build('H0');m=c['model'];f=c['frac'];q=np.full_like(c['W'],-np.expm1(-1/200))
 a=scan(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],q)
 old=parent.scan_dp2(c['h'],c['s'],c['f'],c['k'],c['d'].lower_release,m.inp,m.demand,m.cap,f['gu'],f['phi_f'],f['gs'],float(q[0,0]))
 assert np.array_equal(a[0],old[0]) and np.array_equal(a[1],old[1]) and np.array_equal(a[9],old[2]) and np.array_equal(a[8],old[4])
 a1=scan(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],np.ones_like(q))
 old1=immediate.scan_dp(c['h'],c['s'],c['f'],c['k'],c['d'].lower_release,m.inp,m.demand,m.cap,f['gu'],f['phi_f'],f['gs'])
 assert np.array_equal(a1[0],old1[0]) and np.array_equal(a1[1],old1[1])
 # Perturb only the future daily probability: exact prefix preservation.
 q2=q.copy();q2[-10:]=.9
 fut=scan(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],q2)
 assert np.array_equal(a[:,:-10],fut[:,:-10])
 mass,w,rv=boundary(c,a[0]+a[1]);assert np.isfinite(mass).all()
 # Compare actual station readout, not only kernel outputs.
 oldframe=pd.read_parquet(P/'20260920_2/reports/daily_dense_pL.parquet')
 print('OLD_COLUMNS',oldframe.columns.tolist(),flush=True)
 put(R/'reports/preflight.json',dict(status='PASS',independent_max_abs=float(np.max(abs(b-scan(inp,dem,s,gu,pf,gs,probabilities(W,1,.05))))),H0_constant_bitwise=True,Q1_parent_bitwise=True,future_prefix_bitwise=True,zero_source=True,zero_wetness=True,constant_wetness_scaling=True,no_contact_transfer=True,station_mass_finite=True))
 print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
