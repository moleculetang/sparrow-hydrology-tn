"""Reproduce legacy invalid-rate-to-zero exposure without certifying zero N.

Uses original spatial operator solely to diagnose the previous mainline input.
2021–2023 use 2020 harvested area explicitly for this bridge audit, not as a
new-source product. Unknown area and unknown rates are separate dimensions.
"""
from pathlib import Path
import sys,ast,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np,pandas as pd,h5py,xarray as xr
from scipy.sparse import csr_matrix
BASE=Path(r'E:\SPARROW\0_reach_topology\data\raw\agriculture\nitrogen_inputs')
AREA=BASE/'crop_n_fertilization/global_crop_specific_1961_2020/data/Harvested_area_1961-2020.h5'
SCRIPT=ROOT.parent/'20260815_2/scripts/run_stage2.py'
WEIGHTS=ROOT.parent/'20260815_2/outputs/crop_harvest_area_grid_overlap_weights.parquet'
OUT=ROOT/'outputs/crop_rate_audit';OUT.mkdir(parents=True,exist_ok=True)
def main():
    ok,res=dispatch_allowed(reserve_bytes=2_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_PAUSED')
    tree=ast.parse(SCRIPT.read_text(encoding='utf-8'));cross=None
    for x in tree.body:
        if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CROP_CROSSWALK' for t in x.targets):cross=ast.literal_eval(x.value)
    if cross is None:raise ValueError('NO_LEGACY_CROSSWALK')
    w=pd.read_parquet(WEIGHTS);inferred=None
    for nx in range(int(w.lon_index.max()-w.lon_index.min()+1),int(w.lon_index.max()-w.lon_index.min()+17)):
        lonstart=w.lon_index.to_numpy(int)-w.cell_pos.to_numpy(int)%nx
        latstart=w.lat_index.to_numpy(int)-w.cell_pos.to_numpy(int)//nx
        if np.unique(lonstart).size==1 and np.unique(latstart).size==1:
            inferred=(nx,int(lonstart[0]),int(latstart[0]));break
    if inferred is None:raise ValueError('GRID_OPERATOR_UNKNOWN')
    nx,l0,a0=inferred;ny=int(w.cell_pos.max()//nx+1)
    lon=-180+(np.arange(l0,l0+nx)+.5)/12;lat=90-(np.arange(a0,a0+ny)+.5)/12
    op=csr_matrix((w.fraction_of_cell.to_numpy(),(w.reach_id.to_numpy(int)-1,w.cell_pos.to_numpy(int))),shape=(230,nx*ny))
    rows=[];crop_summary=[]
    with h5py.File(AREA) as f:
        for crop,hani in cross.items():
            a=np.asarray(f[crop][:,l0:l0+nx,a0:a0+ny],dtype='float64').transpose(0,2,1)
            area_valid=np.isfinite(a)&(a>=0)
            apath=BASE/f'hani_crop_1961_2023/data/HaNi-Crop-{hani}.nc'
            with xr.open_dataset(apath,decode_times=False) as d:
                yi=np.asarray(d.time.values,int)
                ilat=np.argmin(abs(d.lat.values[:,None]-lat[None,:]),axis=0)
                ilon=np.argmin(abs(d.lon.values[:,None]-lon[None,:]),axis=0)
                fields={}
                for field in ('Nfer','Nmanure'):
                    if d[field].attrs.get('units')!='kg N ha-1 yr-1':raise ValueError('CROP_RATE_UNIT_CHANGED')
                    b=d[field].isel(lat=slice(ilat.min(),ilat.max()+1),lon=slice(ilon.min(),ilon.max()+1)).values.astype('float64')
                    fields[field]=b[:,ilat-ilat.min()][:,:,ilon-ilon.min()]
                for y in range(1961,2024):
                    ay=min(y,2020);ix=ay-1961;known=np.where(area_valid[ix],a[ix],0.).reshape(-1)
                    unsupported=np.asarray(op@(~area_valid[ix]).reshape(-1).astype(float))
                    harvested=np.asarray(op@known)
                    yy=np.flatnonzero(yi==y)
                    if len(yy)!=1:raise ValueError('RATE_YEAR_MISSING')
                    for field,b in fields.items():
                        rate=b[yy[0]].reshape(-1);valid=np.isfinite(rate)&(rate>=0)
                        missing_area=np.asarray(op@(known*(~valid)))
                        partial=np.asarray(op@(known*np.where(valid,rate,0.)))
                        for r in range(230):
                            unknown=(unsupported[r]>0) or (missing_area[r]>0)
                            rows.append(dict(year=y,harvest_area_year=ay,crop=crop,HaNi_crop=hani,reach_id=r+1,field=field,known_harvested_area_ha=harvested[r],positive_area_without_rate_ha=missing_area[r],unknown_harvest_cell_fraction_sum=unsupported[r],known_partial_mass_kg=partial[r],complete_mass_kg=None if unknown else partial[r],status='unknown_not_zero' if unknown else 'complete_on_registered_operator'))
            sub=pd.DataFrame(rows[-63*230*2:]);crop_summary.append(dict(crop=crop,rate_crop=hani,unknown_reach_year_component_rows=int(sub.status.eq('unknown_not_zero').sum()),positive_area_without_rate_ha_sum=float(sub.positive_area_without_rate_ha.sum())))
            print(crop,crop_summary[-1]['unknown_reach_year_component_rows'],flush=True)
            write_json(OUT/'progress.json',dict(completed_crops=len(crop_summary),total_crops=len(cross)))
    table=pd.DataFrame(rows);table.to_parquet(OUT/'legacy_crop_unknown_exposure.parquet',index=False)
    pd.DataFrame(crop_summary).to_csv(OUT/'crop_gap_summary.csv',index=False)
    missing=table[table.positive_area_without_rate_ha>0]
    missing.groupby(['year','field'],as_index=False).agg(unknown_harvest_area_ha=('positive_area_without_rate_ha','sum'),affected_crop_reach_rows=('reach_id','size')).to_csv(OUT/'annual_unknown_rate_support.csv',index=False)
    write_json(OUT/'receipt.json',dict(status='legacy_zero_substitution_exposure_quantified',rows=len(table),unknown_rows=int(table.status.eq('unknown_not_zero').sum()),
        positive_area_missing_rate_rows=len(missing),source_script_sha256=sha(SCRIPT),operator_sha256=sha(WEIGHTS),
        rate_gap_mass='not identifiable from absent rates; known_partial_mass must not be called complete source',
        inherited_crosswalk=cross,crosswalk_limitation='Sweetpotato->Others is inherited proxy; Pulses has no dedicated harvest key. Not certified as a new crop mapping.',
        bridge_only=True,source_claim='No TN used; no invalid rate imputed as a scientifically known zero',NSE='not applicable'))
if __name__=='__main__':main()
