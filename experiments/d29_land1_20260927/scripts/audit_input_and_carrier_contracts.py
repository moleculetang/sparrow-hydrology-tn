"""Actual-data adapter closures plus fixed H1 water-carrier diagnostics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd
from d29_platform.conditional_inputs import ConditionalInputs
from d29_platform.legacy import build_legacy
OUT=ROOT/'outputs/input_contract_audit';OUT.mkdir(parents=True,exist_ok=True)
RUN=ROOT/'outputs/supply_limited_reference'
def main():
    m,x,_=build_legacy(inference_only=True);ci=ConditionalInputs(m.data.dates);checks=[]
    checks.append(dict(name='LUH_all230_all64years_12physical_classes',passed=ci.area.shape==(64,230,12) and np.isfinite(ci.area).all()))
    dep=ci.dep;comp=['drynhx_kg_n','drynoy_kg_n','wetnhx_kg_n','wetnoy_kg_n']
    err=float(abs(dep[comp].sum(axis=1)-dep.total_kg_n).max());checks.append(dict(name='deposition_four_components',error_kg=err,passed=err<=1e-6))
    known=dep[~dep.clcd_class.isin([0,5])].groupby(['year','month','reach_id'])[comp].sum()
    saved=np.load(ROOT.parent/'20260926_1/outputs/deposition/monthly_land_components_kg_n.npy')
    aligned=known.reindex(pd.MultiIndex.from_product([range(1961,2025),range(1,13),range(1,231)],names=['year','month','reach_id'])).to_numpy().reshape(768,230,4)
    err=float(abs(aligned-saved).max());checks.append(dict(name='known_land_deposition_restores_audited_months',error_kg=err,passed=err<=1e-6))
    dep.groupby(['year','clcd_class','clcd_class_name'])[comp+['total_kg_n']].sum().to_csv(OUT/'deposition_known_water_unknown_annual.csv')
    totals=[]
    # All annual generator fields have frozen byte identities before label evaluation.
    identities=json.loads((RUN/'annual_input_hashes.json').read_text(encoding='utf-8'))
    checks.append(dict(name='annual_input_hashes_precede_evaluation',passed=len(identities)==64 and [r['year'] for r in identities]==list(range(1961,2025))))
    start=ci.initial();end=np.load(RUN/'annual_end_stocks.npy')
    for year in (1961,2020,2024):
        initial=start if year==1961 else end[year-1962]
        plant=initial[:,:,0].copy();idx=np.flatnonzero(m.data.dates.year==year)
        if int(idx[0]) in ci.transitions:plant[:,:12]=np.einsum('ri,rij->rj',plant[:,:12],ci.transitions[int(idx[0])])
        a=ci.annual(year,plant);err=float(abs(a['tags'].sum(3)-a['sources']).max())
        outside=[0,1,2,3,4,10,11,12]
        crop_leak=float(abs(a['tags'][:,:,outside,1:4,:]).max())
        checks.extend([dict(name=f'actual_{year}_source_entry_sum',error_kg=err,passed=err<=1e-6),dict(name=f'actual_{year}_crop_sources_do_not_enter_noncrop',error_kg=crop_leak,passed=crop_leak==0.)])
        for name in ('sources','plant_target','plant_outflows'):
            import hashlib
            digest=hashlib.sha256(memoryview(np.ascontiguousarray(a[name]))).hexdigest()
            checks.append(dict(name=f'actual_{year}_{name}_reproducible_hash',passed=digest==identities[year-1961][name]))
        mass=a['tags'].sum(axis=(0,1,2,4));totals.append(dict(year=year,**dict(zip(ci.labels,map(float,mass)))))
    pd.DataFrame(totals).to_csv(OUT/'reconstructed_source_totals.csv',index=False)
    daily=[]
    for name,water in [('fast',m.data.fast_water),('slow',m.data.slow_water)]:
        flux=np.load(RUN/f'local_{name}_kg_day.npy',mmap_mode='r')
        bad=(flux>1e-7)&(water<=0);inds=np.argwhere(bad)
        for t,r in inds:daily.append(dict(carrier=name,date=str(m.data.dates[t].date()),reach_id=int(r+1),mass_kg=float(flux[t,r]),water_m3=float(water[t,r])))
        checks.append(dict(name=f'{name}_positive_N_requires_positive_carrier_water',positive_N_without_carrier=len(inds),passed=len(inds)==0))
    pd.DataFrame(daily,columns=['carrier','date','reach_id','mass_kg','water_m3']).to_csv(OUT/'mass_without_carrier.csv',index=False)
    station=pd.read_parquet(RUN/'station_days.parquet');valid=station.water_m3>0
    error=float(abs(station.loc[valid,'candidate']-1000*station.loc[valid,'mass_kg']/station.loc[valid,'water_m3']).max())
    checks.append(dict(name='OU_C_equals_1000_same_boundary_M_over_Q',error_mg_l=error,passed=error==0.))
    checks.append(dict(name='zero_water_concentration_undefined',passed=station.loc[~valid,'candidate'].isna().all()))
    checks.append(dict(name='all116_station_boundaries_retained',passed=station.station_key.nunique()==116))
    spatial=pd.read_csv(ROOT.parent/'20260926_1/outputs/spatial_topology/reach_input_spatial_join_230.csv')
    spatial[spatial.H1_area_mismatch_gt1pct].to_csv(OUT/'H1_area_mismatch_unresolved.csv',index=False)
    for c in checks:c['passed']=bool(c['passed'])
    write_json(OUT/'receipt.json',dict(passed=all(c['passed'] for c in checks),checks=checks,
        H1_area_mismatch_count=int(spatial.H1_area_mismatch_gt1pct.sum()),measured_same_boundary_Q_certified=False,
        source_complete=False,unknown_surface_policy='quarantined explicit stored mass, not vegetation or direct river load',
        CCD_driver_enabled=False,CCD_reason='partial crop classes and cross-province footprint ambiguity retained; not summed into current sources',
        NSE='not applicable; this is an input and transport-unit audit'))
    print('contracts',all(c['passed'] for c in checks),checks,flush=True)
if __name__=='__main__':main()
