"""Post-freeze provenance, NSE identity, common-support and numerical audit."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd
from d29_platform.legacy import build_legacy,SNAP
from d29_platform.coupling import route_and_sample,sampling_from_inference_model
RUN=ROOT/'outputs/supply_limited_reference';OUT=RUN/'audit';OUT.mkdir(parents=True,exist_ok=True)
def main():
    receipt=json.loads((RUN/'receipt.json').read_text(encoding='utf-8'))
    p=pd.read_parquet(RUN/'evaluation/paired_HF_days.parquet');rows=[]
    for (station,year),g in p.groupby(['station_key',p.date.dt.year]):
        w=g.read_count.to_numpy(float);w/=w.sum();y=g.truth.to_numpy();ym=w@y;sy=np.sqrt(w@(y-ym)**2)
        for name in ('baseline','candidate'):
            pred=g[name].to_numpy();pm=w@pred;sp=np.sqrt(w@(pred-pm)**2)
            if sy==0:continue
            r=float(w@((y-ym)*(pred-pm))/(sy*sp)) if sp>0 else 0.;a=float(sp/sy);b=float((pm-ym)/sy)
            nse=float(1-w@(pred-y)**2/sy**2);identity=2*r*a-a*a-b*b
            rows.append(dict(station_key=station,year=int(year),model=name,nse=nse,correlation=r,amplitude_ratio=a,standardized_bias=b,
                correlation_term=2*r*a,amplitude_penalty=a*a,bias_penalty=b*b,identity_error=abs(nse-identity),
                increasing_amplitude_at_fixed_mean_would_improve=bool(a<r),n_common_days=len(g),
                observed_mean=ym,predicted_mean=pm,water_correlation=float(np.corrcoef(y,g.water_m3)[0,1]) if np.std(g.water_m3)>0 else None))
    w=pd.DataFrame(rows);w.to_csv(OUT/'station_year_nse_decomposition.csv',index=False)
    annual=pd.read_csv(RUN/'annual_mass_ledger.csv');annual['source_mass_relative_annual_residual']=annual.mass_residual_kg.abs()/annual.sources_total_kg
    annual['unmet_export_fraction']=annual.unmet_plant_export_kg/annual.planned_crop_export_kg
    annual['unmet_return_fraction']=annual.unmet_plant_return_kg/(annual.planned_internal_crop_return_kg+annual.planned_internal_vegetation_return_kg)
    annual.to_csv(OUT/'annual_balance_and_activity_deficits.csv',index=False)
    label=np.load(RUN/'local_source_labels_kg_day.npy',mmap_mode='r');fast=np.load(RUN/'local_fast_kg_day.npy',mmap_mode='r');slow=np.load(RUN/'local_slow_kg_day.npy',mmap_mode='r')
    dates=pd.date_range('1961-01-01','2024-12-31');names=['deposition','fertilizer','field_manure','crop_BNF','initial_research_state']
    fractions=[]
    for year in range(1961,2025):
        mass=label[dates.year==year].sum(axis=(0,1));total=mass.sum()
        for name,v in zip(names,mass):fractions.append(dict(year=year,label=name,local_export_kg=float(v),fraction=float(v/total)))
    pd.DataFrame(fractions).to_csv(OUT/'annual_local_export_source_fractions.csv',index=False)
    # Frozen routing is linear in mass at fixed hydraulic and removal parameters.
    # Propagate each provenance tracer through the SAME OU operator and compare sums.
    m,x,_=build_legacy('F23','U',inference_only=True)
    meta=pd.read_parquet(SNAP/'data/prediction_calendar.parquet');meta=meta[meta.year.between(2021,2024)].reset_index(drop=True)
    m.registry=json.loads((SNAP/'data/prediction_registry.json').read_text(encoding='utf-8'))['records'];m._daily_meta_cache={}
    sampling=sampling_from_inference_model(m,meta);c,rec,weights,n,op=sampling
    result=pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[rec.numpy()],date=m.data.dates[c['ti'].numpy()]))
    stations=pd.read_parquet(RUN/'station_days.parquet')
    if not np.array_equal(result.station_key,stations.station_key) or not np.array_equal(result.date,stations.date):raise ValueError('ROUTING_SAMPLE_IDENTITY')
    for i,name in enumerate(names):
        v=route_and_sample(m.data,np.ascontiguousarray(label[:,:,i]),x[2],sampling)
        result[name+'_mg_l']=1000*v['sample_mass_kg']/v['sample_water_m3']
    columns=[name+'_mg_l' for name in names]
    result['tracer_sum_mg_l']=result[columns].sum(axis=1)
    result['physical_mg_l']=stations.candidate.to_numpy();result.to_parquet(OUT/'station_daily_source_contributions.parquet',index=False)
    contribution_error=float(abs(result.tracer_sum_mg_l-result.physical_mg_l).max())
    physical_ledger=dict(full_history_days=23376,all_reach_count=230,all_station_count=int(result.station_key.nunique()),
        NSE_identity_max_error=float(w.identity_error.max()),local_tag_export_max_difference_kg=float(abs(label.sum(2)-fast-slow).max()),
        routed_tracer_concentration_sum_max_error_mg_l=contribution_error,
        max_annual_source_relative_residual=float(annual.source_mass_relative_annual_residual.max()),
        absolute_mass_contract_met=receipt['strict_mass_pass'],numerical_micro_errors_not_TN_bias_explanation=bool(contribution_error<1e-6),
        note='Relative error descriptions do not replace registered absolute mass acceptance. Real-data full run remains diagnostic.')
    # JSON keys cannot start with a digit as Python keyword arguments.
    physical_ledger['year_2024']=annual[annual.year==2024].to_dict('records')[0]
    write_json(OUT/'independent_result_audit.json',physical_ledger)
    print('result audit',json.dumps({k:v for k,v in physical_ledger.items() if k!='year_2024'}),flush=True)
if __name__=='__main__':main()
