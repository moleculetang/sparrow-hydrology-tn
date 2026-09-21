"""One documented arithmetic correction: restore inherited passive-tag spelling."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from kernel import tags
import closures_dp2 as old
def main():
 records=[]
 for hydro in ('H0','H1'):
  c=build(hydro);m=c['model'];rr=np.array(m.data.pilot_indices,int);f=c['frac']
  gu=np.ascontiguousarray(f['gu'][:,rr]);pf=np.ascontiguousarray(f['phi_f'][:,rr]);gs=np.ascontiguousarray(f['gs'][:,rr]);s=c['s'][rr]
  for arm in read(R/'data/protocol.json')['arms']:
   if arm['hydro']!=hydro or arm['kind']=='D29':continue
   out=R/'outputs'/arm['id'];summary=read(out/'summary.json');q=np.load(out/'q.npy')[:,rr].copy()
   result=tags(m.tag_inputs,m.tag_demand,s,gu,pf,gs,q)
   if arm['kind'] in ('Q1',) or arm['gamma']==0:
    ref=old._tag_dp2_nb(np.ascontiguousarray(c['h'][:,rr]),np.ascontiguousarray(s),np.ascontiguousarray(c['f'][:,rr]),m.tag_release,m.tag_inputs,m.tag_demand,gu,pf,gs,float(q[0,0]))
    for i,j in ((0,0),(1,1),(4,4),(5,5),(6,6)):assert np.array_equal(result[i],ref[j]),('TAG_OLD_REGRESSION',i,arm['id'])
    assert np.array_equal(result[2]+result[3],ref[3])
   with np.load(out/'land_history.npz') as land:
    names=['fast','slow','legacy','mobile','lower','uptake','loss','transfer'];diffs={n:float(abs(result[i].sum(-1)-land[n][:,rr]).max()) for i,n in enumerate(names)}
    diffs['L']=diffs.pop('lower');diffs['M']=float(abs((result[2]+result[3]).sum(-1)-(land['legacy']+land['mobile'])[:,rr]).max())
   previous=summary['gate'];put(out/'initial_source_gate.json',previous)
   gate=dict(previous);gate['source_sum_errors']=diffs;gate['tag_pass']=max(diffs[n] for n in ('fast','slow','M','L','uptake','loss'))<=1e-6
   gate['pass']=bool(gate['network_pass'] and gate['local_balance_max_kg']<=1e-6 and gate['nonnegative_min_kg']>=-1e-7 and gate['uptake_excess_kg']<=1e-7 and gate['tag_pass'])
   summary['gate']=gate;summary['physical_status']='PASS' if gate['pass'] else 'BLOCKED';summary['source_arithmetic_revision']='Restored original 20_2 tag recursion before TN evaluation; physics and parameters unchanged'
   np.savez_compressed(out/'source_tags.npz',values=result,reaches=rr+1);put(out/'summary.json',summary)
   records.append(dict(arm=arm['id'],before=previous['source_sum_errors'],after=diffs,pass_gate=gate['pass']))
 put(R/'reports/source_arithmetic_review.json',records)
 print('SOURCE_REVIEW_COMPLETE',sum(r['pass_gate'] for r in records),len(records))
if __name__=='__main__':main()
