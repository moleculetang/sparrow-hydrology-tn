"""SWAT C:N=14 initialization adaptation on modern SoilGrids SOC, not 1961 truth."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
from d29_platform.runtime import configure,write_json,sha
configure()
import shapely,pyogrio,geopandas as gpd,numpy as np,pandas as pd,rasterio
from rasterio.features import rasterize
BASE=Path('E:/SPARROW/0_reach_topology/data/raw/soil/soilgrids_v2')
OUT=ROOT/'outputs/soil_reference'
def main():
    geo=gpd.read_file('E:/SPARROW/5_Test/20260810_1/inputs/spatial_corrected/reach_catchments.shp')
    zones=None;rows=[];hashes={}
    for depth,thick in [('0-5cm',.05),('5-15cm',.1),('15-30cm',.15),('30-60cm',.3),('60-100cm',.4)]:
        fields=[]
        for prop,part in [('soc','prb_buffer_legacy_n'),('bdod','prb_buffer_legacy_n'),('cfvo','prb_buffer_hydraulic')]:
            path=BASE/part/'data'/prop/f'{prop}_{depth}_mean_prb_buffer.tif';hashes[str(path)]=sha(path)
            with rasterio.open(path) as src:
                if zones is None:
                    trans,shape,crs=src.transform,src.shape,src.crs
                    zones=rasterize([(g,int(r)) for g,r in zip(geo.to_crs(crs).geometry,geo.reach_id)],out_shape=shape,transform=trans,dtype='uint16')
                    support=np.bincount(zones.ravel(),minlength=231);pix=abs(trans.a*trans.e)
                    total=np.zeros(shape);validall=zones>0
                if (src.transform,src.shape,src.crs)!=(trans,shape,crs):raise ValueError('GRID_IDENTITY')
                fields.append(src.read(1,masked=True).astype(float).filled(np.nan))
        soc,bd,cf=fields;valid=(zones>0)&np.isfinite(soc)&np.isfinite(bd)&np.isfinite(cf)&(soc>=0)&(bd>0)&(cf>=0)&(cf<=1000)
        density=soc[valid]*1e-4/14*(bd[valid]*10)*thick*(1-cf[valid]/1000)
        total[valid]+=density;validall &= valid
        if depth in ('15-30cm','60-100cm'):
            mass=np.bincount(zones[validall],weights=total[validall]*pix,minlength=231);counts=np.bincount(zones[validall],minlength=231)
            for r in range(1,231):
                if counts[r]==0:raise ValueError('REACH_SOC_MISSING')
                rows.append(dict(reach_id=r,depth='0-30cm' if depth=='15-30cm' else '0-100cm',son_reference_kg_m2=mass[r]/counts[r]/pix,valid_area_m2=counts[r]*pix,missing_area_m2=(support[r]-counts[r])*pix))
    table=pd.DataFrame(rows);tn=pd.read_csv(OUT/'reach_total_n_reference.csv')
    table=table.merge(tn[['reach_id','depth','total_n_kg_m2_on_valid_pixels']],on=['reach_id','depth'],validate='one_to_one')
    table['son_to_modern_total_n_ratio']=table.son_reference_kg_m2/table.total_n_kg_m2_on_valid_pixels
    table.to_csv(OUT/'son_soc14_reference.csv',index=False)
    write_json(OUT/'son_soc14_receipt.json',dict(status='research_initialization_not_historical_measurement',
        formula='SOC_raw(dg/kg)*1e-4 /14 * bulk_density_raw*10 * depth_m * (1-cfvo/1000)',
        source='SWAT 2009 theory 3:1.1.2; active fraction 0.02 from 3:1.1.3–4; adapted to LAND1 mixed active pool',
        spatial_extrapolation='valid reach mean density applied to LUH3 terrestrial physical area; missing pixel support reported, not zero',
        total_N_conflict='SOC/14 is a separate research estimate. Ratios to total N reported without clipping; maps do not jointly identify initial organic N.',
        max_son_to_total_n_ratio=float(table.son_to_modern_total_n_ratio.max()),hashes=hashes,NSE='not applicable'))
    print('SON reference',table.son_to_modern_total_n_ratio.describe().to_dict(),flush=True)
if __name__=='__main__':main()
