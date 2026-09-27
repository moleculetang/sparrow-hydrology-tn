from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd,xarray as xr
RAW=Path(r'E:\SPARROW\0_reach_topology\data\raw\atmosphere\meteorology\era5_land')
OUT=ROOT/'outputs/soil_temperature';OUT.mkdir(parents=True,exist_ok=True)
def main():
    rows=[];months=set();files=[]
    for p in sorted((RAW/'data/monthly_prb_buffer').glob('*.nc')):
        print('monthly',p.name,flush=True)
        with xr.open_dataset(p) as d:
            times=pd.DatetimeIndex(d.valid_time.values);files.append(dict(path=str(p),sha256=sha(p),variables=list(d.data_vars)))
            good=set(('stl1','stl2','stl3','stl4'))<=set(d.data_vars)
            if good:months.update(str(x.to_period('M')) for x in times)
            for v in ('stl1','stl2','stl3','stl4'):
                if v not in d:continue
                a=d[v].values;unit=d[v].attrs.get('units');valid=np.isfinite(a)
                for i,t in enumerate(times):
                    rows.append(dict(year=t.year,month=t.month,variable=v,unit=unit,
                        min_kelvin=float(np.min(a[i][valid[i]])),max_kelvin=float(np.max(a[i][valid[i]])),
                        nonfinite_grid_cells=int((~valid[i]).sum()),total_grid_cells=int(valid[i].size),
                        filename=p.name))
                if unit!='K':raise ValueError('UNEXPECTED_SOIL_T_UNIT')
    full=set(str(x) for x in pd.period_range('1961-01','2024-12',freq='M'))
    missing=sorted(full-months)
    # Hourly PET files are checked for variables, not presumed to contain soil T.
    hourly=[]
    for p in sorted((RAW/'hourly_pet_prb').glob('*.nc')):
        if len(hourly)%24==0:print('hourly header',len(hourly),p.name,flush=True)
        with xr.open_dataset(p) as d:
            hourly.append(dict(filename=p.name,variables=list(d.data_vars),soil_temperature_variables=[x for x in d.data_vars if x.startswith('stl') or 'soil_temperature' in x]))
    pd.DataFrame(rows).to_csv(OUT/'monthly_layer_inventory.csv',index=False)
    pd.DataFrame({'missing_month':missing}).to_csv(OUT/'missing_full_history_months.csv',index=False)
    decision=dict(selected_full_history_equation='fixed_first_order',
        reason='Local monthly soil T does not cover 1961–2024; no reconstruction or year substitution authorized by this conditional branch.',
        requested_months=768,available_months=len(full&months),missing_months=len(missing),
        first_available=min(months),last_available=max(months),spatial_resolution_degrees=.1,
        temporal_resolution='monthly mean, not daily soil-temperature dynamics',
        ERA5_layers_cm=[[0,7],[7,28],[28,100],[100,289]],
        possible_0_30cm_mapping='7/30 stl1 + 21/30 stl2 + 2/30 stl3; effective thickness average, not observed profile',
        monthly_to_daily='not performed; repeating monthly means would be an explicit scenario, not daily observations',
        hourly_pet_files_checked=len(hourly),hourly_pet_files_with_soil_temperature=sum(bool(x['soil_temperature_variables']) for x in hourly),
        source_files=files,hourly_inventory=hourly,
        official_source='https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview',
        NSE='not applicable: forcing inventory, not water quality skill',
        future_temperature_mode='implemented but disabled until full intended period, soil depth, spatial support and units pass gate')
    write_json(OUT/'decision.json',decision)
    print(json.dumps({k:decision[k] for k in ['available_months','missing_months','first_available','last_available','hourly_pet_files_checked','hourly_pet_files_with_soil_temperature','selected_full_history_equation']}),flush=True)
if __name__=='__main__':main()
