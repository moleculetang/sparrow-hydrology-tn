"""Modern SoilGrids total-N reference, not an observed 1961 SON initial state.

Native equal-area pixel centres identify reaches; density is reported only on
the valid support. No missing pixel is silently assigned zero nitrogen.
"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
os.environ.setdefault('GDAL_DATA',str(Path(sys.executable).parent/'Library/share/gdal'))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import shapely,pyogrio,geopandas as gpd,numpy as np,pandas as pd,rasterio
from rasterio.features import rasterize
from d29_platform.source_definitions import soil_total_n_stock_kg_m2
BASE=Path(r'E:\SPARROW\0_reach_topology\data\raw\soil\soilgrids_v2')
CATCH=Path(r'E:\SPARROW\5_Test\20260810_1\inputs\spatial_corrected\reach_catchments.shp')
OUT=ROOT/'outputs/soil_reference';OUT.mkdir(parents=True,exist_ok=True)
DEPTHS=[('0-5cm',.05),('5-15cm',.1),('15-30cm',.15),('30-60cm',.3),('60-100cm',.4)]
def main():
    allowed,res=dispatch_allowed(reserve_bytes=2_000_000_000)
    if not allowed:raise RuntimeError('RESOURCE_PAUSED '+str(res))
    g=gpd.read_file(CATCH);rows=[];hashes={};sumdensity=None;allvalid=None;counts=None
    for depth,thick in DEPTHS:
        arrays=[]
        for prop,part in [('nitrogen','prb_buffer_legacy_n'),('bdod','prb_buffer_legacy_n'),('cfvo','prb_buffer_hydraulic')]:
            path=BASE/part/'data'/prop/f'{prop}_{depth}_mean_prb_buffer.tif';hashes[str(path)]=sha(path)
            with rasterio.open(path) as src:
                if counts is None:
                    transform,crs,shape=src.transform,src.crs,src.shape
                    zones=rasterize([(x.geometry,int(x.reach_id)) for x in g.to_crs(crs).itertuples()],out_shape=shape,transform=transform,fill=0,dtype='uint16')
                    counts=np.bincount(zones.ravel(),minlength=231);pixel=abs(transform.a*transform.e)
                    sumdensity=np.zeros(shape,np.float64);allvalid=zones>0
                if src.shape!=shape or src.crs!=crs or src.transform!=transform:raise ValueError('SOIL_GRIDS_NOT_IDENTICAL')
                arrays.append(src.read(1,masked=True).astype('float64').filled(np.nan))
        n,bd,cf=arrays
        valid=(zones>0)&np.isfinite(n)&np.isfinite(bd)&np.isfinite(cf)&(n>=0)&(bd>0)&(cf>=0)&(cf<=1000)
        density=np.full(shape,np.nan)
        density[valid]=soil_total_n_stock_kg_m2(n[valid],bd[valid],cf[valid],thick)
        cnt=np.bincount(zones[valid],minlength=231);tot=np.bincount(zones[valid],weights=density[valid]*pixel,minlength=231)
        sumdensity[valid]+=density[valid];allvalid &= valid
        for r in range(1,231):rows.append(dict(reach_id=r,depth=depth,thickness_m=thick,pixel_support_area_m2=float(counts[r]*pixel),valid_area_m2=float(cnt[r]*pixel),missing_area_m2=float((counts[r]-cnt[r])*pixel),total_n_mass_kg_on_valid_pixels=float(tot[r]),total_n_kg_m2_on_valid_pixels=float(tot[r]/(cnt[r]*pixel)) if cnt[r] else None))
        if depth in ('15-30cm','60-100cm'):
            cnt=np.bincount(zones[allvalid],minlength=231);tot=np.bincount(zones[allvalid],weights=sumdensity[allvalid]*pixel,minlength=231)
            for r in range(1,231):rows.append(dict(reach_id=r,depth='0-30cm' if depth=='15-30cm' else '0-100cm',thickness_m=.3 if depth=='15-30cm' else 1.,pixel_support_area_m2=float(counts[r]*pixel),valid_area_m2=float(cnt[r]*pixel),missing_area_m2=float((counts[r]-cnt[r])*pixel),total_n_mass_kg_on_valid_pixels=float(tot[r]),total_n_kg_m2_on_valid_pixels=float(tot[r]/(cnt[r]*pixel)) if cnt[r] else None))
        print('soil',depth,'valid fraction',float(valid.sum()/(zones>0).sum()),flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'reach_total_n_reference.csv',index=False)
    write_json(OUT/'receipt.json',dict(status='reference_aggregated_not_historical_SON',source_hashes=hashes,output_sha256=sha(OUT/'reach_total_n_reference.csv'),
        formula='N_raw*1e-5 * bdod_raw*10 * thickness_m * (1-cfvo_raw/1000)',
        spatial_method='native equal-area pixel-centre reach assignment; support area and missing area explicit; not exact edge intersection',
        organic_fraction='not measured by SoilGrids total-N layer; scenario required',year='modern multi-year soil prediction, not 1961 observation',
        units_source='https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html',
        depth_choice='0-30 and 0-100 cm reference products; no automatic scientific selection',NSE='not applicable'))
if __name__=='__main__':main()
