"""Four registered old points: diagnostic only, never new initialization."""
import gc
import native_runtime as rt
from campaign_model import *
R=RUN;OLD=R.parent/'20260915_5'
def main():
 tags=['F23_M_HF_s1','F23_D_HF_s0','F24_M_HF_s1','F24_D_HF_s1'];dest=R/'diagnostics/old_extrapolation';dest.mkdir(parents=True,exist_ok=True)
 des=rt.read(R/'data/frozen_design.json');des.update(observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(R/'data/prediction_registry.json'),operator_id='OU')
 d=load_data('FULL24');m=make_model(d,None,'D29_BE',des);meta=pd.read_parquet(R/'data/prediction_calendar.parquet');meta=meta[meta.station_key.eq('棉江')&meta.year.le(2024)].copy();assert len(meta)==48
 ref=np.array([56,59,158,190,213])-1;ix=112;raw=d.static_raw;rows=[];sources={}
 env=pd.DataFrame(dict(feature=np.arange(raw.shape[1]),mianjiang_raw=raw[ix],training_min=raw[ref].min(0),training_max=raw[ref].max(0),standardized=m.x.numpy()[ix],outside_seven_station_support=(raw[ix]<raw[ref].min(0))|(raw[ix]>raw[ref].max(0))))
 env.to_csv(dest/'environment_support.csv',index=False)
 for tag in tags:
  path=OLD/'outputs'/tag/'model.json';sources[str(path)]=rt.sha(path);rec=rt.read(path);x=np.array(rec['parameters']);h,s,f,_=m.flux_parameters(torch.tensor(x));h=h.detach().numpy();f=f.detach().numpy();led=m.ledger(x)
  with torch.no_grad():b=m.daily_boundary(torch.tensor(x),meta)
  pd.DataFrame(dict(date=d.dates[b['day_index'].numpy()],mass_kg=b['mass'].numpy(),water_m3=b['water'].numpy(),concentration_mg_l=(1000*b['mass']/b['water']).numpy())).to_parquet(dest/f'{tag}_mianjiang_daily.parquet',index=False)
  for r in [56,59,158,190,213,113]:
   for year in [2021,2022,2023,2024]:
    sel=d.dates.year==year;i=r-1;uptake=led['uptake'][sel,i].sum();demand=led['demand'][sel,i].sum()
    rows.append(dict(old_tag=tag,reach_id=r,year=year,bfi=float(d.bfi[i]),pi=float(m.pi[i]),beta=float((1-m.pi[i])*x[1]+m.pi[i]*x[29]),hazard_p95=float(np.quantile(h[sel,i],.95)),hazard_max=float(h[sel,i].max()),fast_partition_mean=float(f[sel,i].mean()),M_end=float(led['M'][sel,i][-1]),L_end=float(led['L'][sel,i][-1]),fast_kg=float(led['fast'][sel,i].sum()),slow_kg=float(led['slow'][sel,i].sum()),loss_kg=float(led['mineral_loss'][sel,i].sum()),uptake_demand=float(uptake/demand) if demand else None))
  del h,f,led,b;gc.collect()
 pd.DataFrame(rows).to_csv(dest/'mapping_inventory_risk.csv',index=False);rt.write(dest/'provenance.json',dict(sources=sources,old_points_used_as_initialization=False,inventory_warm_start=False,new_inputs='v3 FULL24',fixed_formulas=True))
 print('OLD_EXTRAPOLATION_DIAGNOSTICS_COMPLETE',flush=True)
if __name__=='__main__':main()
