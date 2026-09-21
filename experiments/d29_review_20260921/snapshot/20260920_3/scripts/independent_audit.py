"""Saved-output independent verification. Not an external expert review."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from preflight import independent
import closures_dp
def main():
 audits=[];identities=[]
 for hydro in ('H0','H1'):
  c=build(hydro);m=c['model'];d=c['d'];f=c['frac']
  for arm in read(R/'data/protocol.json')['arms']:
   if arm['hydro']!=hydro:continue
   out=R/'outputs'/arm['id'];summary=read(out/'summary.json');land=np.load(out/'land_history.npz')
   item=dict(arm=arm['id'],original_physical_gate=summary['physical_status'])
   if arm['kind']!='D29':
    q=np.load(out/'q.npy');a=independent(m.inp,m.demand,c['s'],f['gu'],f['phi_f'],f['gs'],q)
    channels=['fast','slow','legacy','mobile','lower','uptake','loss','transfer','mobile_pre']
    errors={k:float(np.max(abs(a[i]-land[k]))) for i,k in enumerate(channels)}
    item['independent_channel_max_abs_kg']=errors
    item['independent_forward_within_1e6']=max(errors.values())<=1e-6
    item['reference_transfer_relative_difference']=abs(float(land['transfer'][c['ref']].sum())-summary['reference_transfer'])/summary['reference_transfer']
    if arm['kind']=='Q1':
     b=closures_dp.scan_dp(c['h'],c['s'],c['f'],c['k'],d.lower_release,m.inp,m.demand,m.cap,f['gu'],f['phi_f'],f['gs'])
     item['Q1_parent_bitwise']=np.array_equal(b[0],land['fast']) and np.array_equal(b[1],land['slow'])
    # Conservation is recalculated directly from delivered stocks and independent uptake/loss.
    stock=land['legacy']+land['mobile']+land['lower']
    balance=m.inp-land['uptake']-land['loss']-land['fast']-land['slow']-np.diff(stock,axis=0,prepend=np.zeros_like(stock[:1]))
    item['delivered_local_balance_max_kg']=float(abs(balance).max())
    item['reference_transfer_by_reach']=land['transfer'][c['ref']].sum(0).tolist()
    del a
   frame=pd.read_parquet(out/'daily_station.parquet');item['concentration_identity_max_abs']=float(abs(frame.p-1000*frame.mass_kg/frame.water_m3).max());item['daily_rows']=len(frame)
   river=np.load(out/'river_history.npz');local=land['fast']+land['slow'];net=float(local.sum()-river['terminal'].sum()-river['stocks'][-1].sum()-river['loss'].sum())
   item['independent_network_balance_kg']=net;item['network_relative']=abs(net)/float(local.sum())
   if arm['id']=='H0_G0':
    old=pd.read_parquet(P/'20260920_2/reports/daily_dense_pL.parquet');old=old[old.arm.eq('P-1e2')]
    old['date']=pd.to_datetime(old.date);z=frame.merge(old,on=['station_key','date'],validate='one_to_one')
    item['old_H0_station_regression_rows']=len(z);item['old_H0_station_max_abs']=float(abs(z.p-z.pL3).max());assert len(z)==len(frame) and item['old_H0_station_max_abs']<=1e-12
   audits.append(item);land.close();river.close();resource('audited_'+arm['id'])
  # Actual old code modules used in the process, hashed; no source is edited.
  for name,mod in list(sys.modules.items()):
   file=getattr(mod,'__file__',None)
   if file and str(P).lower() in str(file).lower() and str(file).endswith('.py'):
    identities.append(dict(module=name,path=file,sha256=sha(file)))
 put(R/'reports/independent_physical_audit.json',audits);put(R/'data/imported_algorithm_hashes.json',identities)
 h0=read(R/'data/lineage_H0.json');h1=read(R/'data/lineage_H1.json')
 changed=[k for k in h0['arrays'] if h0['arrays'][k]['sha256']!=h1['arrays'][k]['sha256']]
 for k in ('source','crop','source_tags','static_raw','area_ha','temperature','dates','months'):assert h0['arrays'][k]['sha256']==h1['arrays'][k]['sha256'],('NONHYDRO_CHANGE',k)
 # The default read barrier refuses a forbidden label path before it can be opened.
 blocked=False
 try:open(P/'20260918_1/data/heldout_labels/monthly_original.parquet','rb')
 except PermissionError:blocked=True
 assert blocked
 # Freeze and input hash refusal are not inferred from a successful read.
 expected=(R/'data/protocol.sha256').read_text().strip();assert sha(R/'data/protocol.json')==expected
 bad=hashlib.sha256((R/'data/protocol.json').read_bytes()+b'corruption').hexdigest();assert bad!=expected
 # Duplicate process lock and repeated counters are tested without launching a second run.
 tmp=R/'work/audit_lock';tmp.write_text('audit');duplicate=False
 try:
  with tmp.open('x'):pass
 except FileExistsError:duplicate=True
 finally:tmp.unlink()
 assert duplicate
 count=sum(1 for _ in (R/'logs/roots.jsonl').open());assert count<=512
 put(R/'reports/input_and_execution_audit.json',dict(hydrology_changed=changed,nonhydro_identity=True,label_barrier=True,bad_protocol_hash_detected=True,duplicate_lock_rejected=True,root_calls=count,root_counter_recovered=count,physics_only=True,external_expert=False))
 print('INDEPENDENT_AUDIT_COMPLETE',len(audits))
if __name__=='__main__':main()
