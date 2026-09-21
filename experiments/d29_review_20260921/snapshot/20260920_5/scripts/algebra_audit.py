from runtime import *
from family_kernel import block
def main():
 a={k:np.load(R/'data'/f'{k}.npy',mmap_mode='r')[:500,:5] for k in ['inp','demand','gu','pf','gs','q','gf','volume_mm','fast_mm','percol_mm','post_mm']};s=np.load(R/'data/s.npy')[:5];z=np.zeros_like(a['gu']);st=np.zeros((4,5,1));args=[a[k] for k in ['gu','pf','gs','q','gf']]
 x,_=block(a['inp'][:,:,None],a['demand'],s,*args,z,z,z,1,.5,1,st);y,_=block(a['inp'][:,:,None],a['demand'],s,*args,z,z,z,1,1,1,st)
 # Internal exchange amount differs because it exactly cancels the different allocation.
 err=float(abs(x[[i for i in range(21) if i!=14]]-y[[i for i in range(21) if i!=14]]).max());assert err<1e-6
 aa,_=block(a['inp'][:1,:,None],a['demand'][:1],s,*[v[:1] for v in args],z[:1],z[:1],z[:1],0,0,0,st)
 bb,_=block(a['inp'][:1,:,None],a['demand'][:1],s,*[v[:1] for v in args],z[:1],z[:1],z[:1],0,1,0,st);assert np.allclose(aa[0],bb[0],atol=1e-10,rtol=1e-14)
 # One-day C proportional inventories and input give equal concentration before/after water transfer.
 checks={}
 for w in [.2,.5,.8]:
  v=a['volume_mm'];f=a['fast_mm'];p=a['percol_mm'];h=(1-w)*f-w*p;vf=w*a['post_mm']+f;vp=(1-w)*a['post_mm']+p
  total=np.ones_like(v);af=w*total;am=(1-w)*total;ef=h/np.where(h>=0,(1-w)*v,w*v);xx=np.where(ef>=0,-ef*am,-ef*af);af-=xx;am+=xx
  e=float(max(abs(af*f/vf-total*f/v).max(),abs(am*p/vp-total*p/v).max()));assert e<1e-12;checks[str(w)]=e
 put(R/'reports/algebraic_equivalence.json',dict(B_rho1_pi_independent_max_error_kg=err,A_same_day_fast_identity=True,C_pi_equals_omega_equal_concentration_invariant_when_contact_continuous=checks,C_contact_guard_breaks_invariant='G=0 blocks nitrogen outputs but water still leaves; subsequent differences can arise from inherited decoupling',not_independent_evidence=['B-P0.5-R1 and B-P1-R1'],interpretation='C main is a conditional null-like control; source-enriched points test joint source accessibility and domains, not pure mixing'))
 print('ALGEBRA_AUDIT_PASS')
if __name__=='__main__':main()
