"""Post-fit selected-point full-history physical replays, no optimization or labels."""
import gc,time
import native_runtime as rt
from campaign_model import *
from routing import route
from validate_global import physical
R=RUN
def main():
 selected=rt.read(R/'reports/prediction_freeze_manifest.json')['selected'];cfgs=rt.read(R/'configs/folds.json');proof={}
 for config,tag in selected.items():
  rec=rt.read(R/'outputs'/tag/'model.json');cfg=cfgs[config];d=load_data(cfg['domain']);des=dict(rec['design']);des.update(observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(R/'data/prediction_registry.json'))
  m=make_model(d,None,rec['job']['kind'],des);d=m.data;x=np.array(rec['parameters']);assert len(x)==len(m.names),(tag,len(x),len(m.names));p,a=physical(m,x);net=route(d,a['fast']+a['slow'],float(x[2]));assert np.allclose(net['channel_removed'],a['channel_loss'],rtol=1e-12,atol=1e-9);folder=R/'reports/selected_routing'/tag;folder.mkdir(parents=True,exist_ok=True)
  rr=len(d.global_reach_ids);q=pd.DataFrame(dict(year=np.repeat(d.months.year,rr),month=np.repeat(d.months.month,rr),reach_id=np.tile(d.global_reach_ids,len(d.months))))
  for name in ['inlet','preout','official','channel_removed']:q[name+'_kg']=d.monthly_sum(net[name]).ravel()
  q.to_parquet(folder/'monthly_reach_routing.parquet',index=False)
  count=len(d.metadata);v=pd.DataFrame(dict(year=np.repeat(d.months.year,count),month=np.repeat(d.months.month,count),reservoir_index=np.tile(np.arange(count),len(d.months))))
  for name in ['captures','releases']:v[name+'_kg']=d.monthly_sum(net[name]).ravel()
  v['end_stock_kg']=net['stocks'][d.stops-1].ravel();v.to_parquet(folder/'monthly_reservoir_routing.parquet',index=False)
  # Daily response chain over observed years, with unchanged month-start source inputs.
  mask=d.dates.year>=2021;nd=int(mask.sum());q=pd.DataFrame(dict(date=np.repeat(d.dates[mask],rr),reach_id=np.tile(d.global_reach_ids,nd)))
  for name in ['fast','slow','uptake','mineral_loss']:q[name+'_kg']=a[name][mask].ravel()
  for name in ['inlet','preout','official']:q[name+'_kg']=net[name][mask].ravel()
  inp=np.zeros_like(a['fast']);inp[d.starts]=d.source;q['source_input_kg']=inp[mask].ravel();q.to_parquet(folder/'daily_source_to_outlet.parquet',index=False)
  proof[tag]=dict(physical=p,parameter_hash=rt.sha(R/'outputs'/tag/'model.json'),files={f.name:rt.sha(f) for f in folder.glob('*.parquet')},start_year=1961,end_year=cfg['end_year'],no_labels=True)
  del m,d,a,net,inp,q,v;gc.collect()
 rt.write(R/'reports/selected_routing_recompute.json',dict(status='PASS_SELECTED_ROUTING_RECOMPUTE',paths=proof))
 print('SELECTED_ROUTING_RECOMPUTED',len(proof),flush=True)
if __name__=='__main__':main()
