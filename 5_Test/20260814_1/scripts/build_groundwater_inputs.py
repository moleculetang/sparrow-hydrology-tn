from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Geod
from shapely.geometry import Polygon

from runtime_guard import assert_sparrow_runtime


warnings.filterwarnings("ignore")
assert_sparrow_runtime()
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\SPARROW\0_reach_topology\data\raw\hydrology\groundwater\groundwater_level_china_1km_monthly_2005_2022")
CATCHMENTS = ROOT / "inputs" / "spatial" / "reach_catchments.shp"
WEIGHTS = ROOT / "inputs" / "spatial" / "groundwater_grid_overlap_weights.parquet"
MONTHLY = ROOT / "inputs" / "spatial" / "groundwater_catchment_monthly_2005_2018.parquet"
REPORT = ROOT / "reports" / "groundwater_input_gate.json"
MANIFEST = ROOT / "reports" / "groundwater_source_manifest.csv"
DATE_RE = re.compile(r"GWs_(\d{4})-(\d{2})\.tif$", re.I)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dense_box(x0: float, y0: float, x1: float, y1: float, parts: int = 8) -> Polygon:
    xmin, xmax = sorted((x0, x1)); ymin, ymax = sorted((y0, y1))
    xs = np.linspace(xmin, xmax, parts + 1); ys = np.linspace(ymin, ymax, parts + 1)
    points = ([(float(x), ymin) for x in xs] + [(xmax, float(y)) for y in ys[1:]] +
              [(float(x), ymax) for x in xs[-2::-1]] + [(xmin, float(y)) for y in ys[-2:0:-1]])
    return Polygon(points)


def tiff_grid(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        scale = image.tag_v2.get(33550)
        tie = image.tag_v2.get(33922)
        nodata_tag = image.tag_v2.get(42113)
        nodata = float(str(nodata_tag).strip("\x00")) if nodata_tag is not None else -1e30
        xres, yres = float(scale[0]), float(scale[1])
        xmin, ymax = float(tie[3]), float(tie[4])
        return {"width":image.width,"height":image.height,"xres":xres,"yres":yres,"xmin":xmin,"ymax":ymax,
                "xmax":xmin+image.width*xres,"ymin":ymax-image.height*yres,"nodata":nodata,
                "mode":image.mode,"compression":str(image.info.get("compression"))}


def signature(grid: dict[str, object]) -> str:
    keys = ["width","height","xres","yres","xmin","ymax","xmax","ymin","nodata","mode"]
    return sha256_bytes(json.dumps({k:grid[k] for k in keys}, sort_keys=True).encode())


def discover() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.rglob("*.tif")):
        match = DATE_RE.search(path.name)
        if not match: continue
        year, month = map(int, match.groups())
        grid = tiff_grid(path)
        rows.append({"year":year,"month":month,"path":str(path),"size_bytes":path.stat().st_size,
                     "grid_signature_sha256":signature(grid),**grid,
                     "access_role":"development_values" if year<=2018 else "locked_metadata_only"})
    frame = pd.DataFrame(rows).sort_values(["year","month"]).reset_index(drop=True)
    if len(frame)!=216 or frame[["year","month"]].duplicated().any() or frame.grid_signature_sha256.nunique()!=1:
        raise RuntimeError("Groundwater raster discovery/grid signature gate failed")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST,index=False,encoding="utf-8-sig")
    return frame


def build_weights(grid: dict[str, object]) -> tuple[pd.DataFrame,pd.DataFrame,float]:
    catch = gpd.read_file(CATCHMENTS)[["reach_id","inc_km2","geometry"]].copy().sort_values("reach_id")
    if len(catch)!=230 or catch.reach_id.nunique()!=230 or catch.crs is None:
        raise RuntimeError("Catchment gate failed")
    catch_ll = catch.to_crs(4326)
    rows=[]; closure=[]
    xmin=float(grid["xmin"]); ymax=float(grid["ymax"]); xres=float(grid["xres"]); yres=float(grid["yres"])
    width=int(grid["width"]); height=int(grid["height"])
    for metric, ll in zip(catch.itertuples(index=False), catch_ll.itertuples(index=False)):
        minx,miny,maxx,maxy=ll.geometry.bounds
        c0=max(0,int(math.floor((minx-xmin)/xres))-1); c1=min(width-1,int(math.floor((maxx-xmin)/xres))+1)
        r0=max(0,int(math.floor((ymax-maxy)/yres))-1); r1=min(height-1,int(math.floor((ymax-miny)/yres))+1)
        rec=[]; geoms=[]
        for r in range(r0,r1+1):
            y1=ymax-r*yres; y0=y1-yres
            for c in range(c0,c1+1):
                x0=xmin+c*xres; x1=x0+xres
                rec.append((r,c,r*width+c)); geoms.append(dense_box(x0,y0,x1,y1,8))
        cells=gpd.GeoSeries(geoms,crs=4326).to_crs(catch.crs)
        areas=cells.intersection(metric.geometry).area.to_numpy(float)
        keep=areas>1e-6
        if not keep.any(): raise RuntimeError(f"No overlap for reach {metric.reach_id}")
        selected=np.asarray(rec,dtype=np.int64)[keep]; a=areas[keep]; total=float(a.sum()); geom_area=float(metric.geometry.area)
        w=a/total
        for (r,c,cell),area,weight in zip(selected,a,w):
            rows.append({"reach_id":int(metric.reach_id),"raster_row":int(r),"raster_col":int(c),"cell_id":int(cell),
                         "overlap_area_m2":float(area),"catchment_area_m2":geom_area,"catchment_area_km2":float(metric.inc_km2),
                         "base_weight":float(weight),"area_method":"dense8_pixel_footprint_Albers_exact_intersection"})
        closure.append({"reach_id":int(metric.reach_id),"n_intersecting_cells":int(keep.sum()),"overlap_area_m2":total,
                        "catchment_area_m2":geom_area,"coverage_fraction":total/geom_area,"weight_sum":float(w.sum())})
    weights=pd.DataFrame(rows).sort_values(["reach_id","raster_row","raster_col"]).reset_index(drop=True)
    close=pd.DataFrame(closure).sort_values("reach_id")
    grid_sig=str(grid["grid_signature_sha256"])
    geometry_hash=sha256_bytes(b"".join(catch.geometry.to_wkb()))
    crs_hash=sha256_bytes(catch.crs.to_wkt().encode())
    weights["grid_signature_sha256"]=grid_sig; weights["geometry_wkb_sha256"]=geometry_hash; weights["area_crs_wkt_sha256"]=crs_hash
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True); weights.to_parquet(WEIGHTS,index=False)
    close.to_csv(ROOT/"reports"/"groundwater_area_closure.csv",index=False,encoding="utf-8-sig")

    geod=Geod(ellps="WGS84")
    relative=[]
    for metric,ll in zip(catch.itertuples(index=False),catch_ll.itertuples(index=False)):
        geod_area=abs(float(geod.geometry_area_perimeter(ll.geometry)[0])); metric_area=float(metric.geometry.area)
        relative.append(abs(metric_area-geod_area)/geod_area)
    return weights,close,float(max(relative))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order=np.argsort(values,kind="mergesort"); v=values[order]; w=weights[order]
    return float(v[np.searchsorted(np.cumsum(w),0.5*w.sum(),side="left")])


def aggregate(manifest: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    dev=manifest[manifest.year<=2018].copy()
    minr,maxr=int(weights.raster_row.min()),int(weights.raster_row.max())
    minc,maxc=int(weights.raster_col.min()),int(weights.raster_col.max())
    local_r=weights.raster_row.to_numpy(int)-minr; local_c=weights.raster_col.to_numpy(int)-minc
    groups={rid:idx.to_numpy(int) for rid,idx in weights.groupby("reach_id").groups.items()}
    out=[]
    for row in dev.itertuples(index=False):
        path=Path(row.path)
        with Image.open(path) as image:
            crop=np.asarray(image.crop((minc,minr,maxc+1,maxr+1)),dtype=float)
        values=crop[local_r,local_c]
        valid=np.isfinite(values)&(values>-1e30)
        for rid,idx in groups.items():
            base=weights.loc[idx,"base_weight"].to_numpy(float); good=valid[idx]
            valid_fraction=float(base[good].sum()); mean=np.nan; median=np.nan; renorm=np.nan
            if valid_fraction>=0.95 and good.any():
                ww=base[good]/valid_fraction; vv=values[idx][good]; renorm=float(ww.sum())
                mean=float(np.sum(ww*vv)); median=weighted_median(vv,ww)
            out.append({"reach_id":int(rid),"year":int(row.year),"month":int(row.month),
                        "groundwater_area_mean_m":mean,"groundwater_area_weighted_median_m":median,
                        "coverage_fraction":float(weights.loc[idx,"overlap_area_m2"].sum()/weights.loc[idx,"catchment_area_m2"].iloc[0]),
                        "valid_area_fraction":valid_fraction,"n_intersecting_cells":int(len(idx)),"n_valid_cells":int(good.sum()),
                        "static_weight_sum":float(base.sum()),"renormalized_valid_weight_sum":renorm,
                        "area_method":"dense8_pixel_footprint_Albers_exact_intersection",
                        "grid_signature_sha256":str(row.grid_signature_sha256)})
    frame=pd.DataFrame(out).sort_values(["reach_id","year","month"]).reset_index(drop=True)
    frame.to_parquet(MONTHLY,index=False)
    return frame


def main() -> None:
    manifest=discover(); grid=manifest.iloc[0].to_dict(); grid["grid_signature_sha256"]=manifest.iloc[0].grid_signature_sha256
    weights,closure,geod_error=build_weights(grid)
    monthly=aggregate(manifest,weights)
    gate={
        "rasters_discovered":int(len(manifest)),"grid_signatures":int(manifest.grid_signature_sha256.nunique()),
        "locked_rasters_metadata_only":int((manifest.access_role=="locked_metadata_only").sum()),
        "weight_rows":int(len(weights)),"weight_reaches":int(weights.reach_id.nunique()),
        "coverage_min":float(closure.coverage_fraction.min()),"coverage_max":float(closure.coverage_fraction.max()),
        "weight_sum_max_abs_error":float((closure.weight_sum-1).abs().max()),"albers_geodesic_max_relative_error":geod_error,
        "monthly_rows":int(len(monthly)),"monthly_unique_keys":int(monthly[["reach_id","year","month"]].drop_duplicates().shape[0]),
        "valid_fraction_min":float(monthly.valid_area_fraction.min()),"all_values_end_by":f"{int(monthly.year.max())}-{int(monthly[monthly.year.eq(monthly.year.max())].month.max()):02d}",
        "weights_sha256":file_sha256(WEIGHTS),"monthly_sha256":file_sha256(MONTHLY),
    }
    gate["pass"]=bool(gate["rasters_discovered"]==216 and gate["grid_signatures"]==1 and gate["locked_rasters_metadata_only"]==48 and
                      gate["weight_reaches"]==230 and gate["coverage_min"]>=0.999 and gate["coverage_max"]<=1.001 and
                      gate["weight_sum_max_abs_error"]<=1e-10 and geod_error<=1e-5 and gate["monthly_rows"]==38640 and
                      gate["monthly_unique_keys"]==38640 and gate["all_values_end_by"]=="2018-12")
    REPORT.write_text(json.dumps(gate,indent=2),encoding="utf-8")
    print(json.dumps(gate,indent=2))
    if not gate["pass"]: raise RuntimeError(f"Groundwater input gate failed: {gate}")


if __name__=="__main__":
    main()
