"""Recheck historical facility IDs on H1 polygons without nearest-only relocation.

The original master/coordinates are supplied by the caller and not bundled.
The result does not certify actual outfalls. Coordinates must use EPSG:4326;
the supplied polygons and river lines must have the same projected metre CRS.
"""
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd

def map_facilities(master,catchments,lines):
    assert master.chen_fid.is_unique
    assert catchments.index.name==lines.index.name=='reach_id'
    assert catchments.index.is_unique and lines.index.is_unique
    assert catchments.crs==lines.crs and catchments.crs.is_projected
    assert all(axis.unit_conversion_factor==1 for axis in catchments.crs.axis_info)
    points=gpd.GeoDataFrame(master[['chen_fid']],geometry=gpd.points_from_xy(master.longitude,master.latitude),crs='EPSG:4326').to_crs(catchments.crs)
    records=[]
    for i,p in enumerate(points.geometry):
        m=master.iloc[i]
        containing=catchments.index[catchments.geometry.covers(p)].tolist()
        distances=lines.geometry.distance(p)
        distance=float(distances.min())
        nearest=lines.index[np.isclose(distances.to_numpy(),distance,rtol=0.,atol=1e-8)].tolist()
        rid=int(containing[0]) if len(containing)==1 else None
        nid=int(nearest[0]) if len(nearest)==1 else None
        agrees=rid is not None and nid is not None and rid==nid
        quality=('A' if distance<=2000 else 'B' if distance<=5000 else 'C') if agrees else 'D'
        historic=int(m.model_reach_id) if pd.notna(m.model_reach_id) else None
        retained=historic is not None and agrees and rid==historic
        state='retained_same_reach_candidate' if retained else 'outside_current_domain' if not containing else 'ambiguous_boundary_or_nearest' if len(containing)!=1 or len(nearest)!=1 else 'excluded_catchment_nearest_disagreement' if not agrees else 'new_geometric_candidate_requires_separate_review'
        records.append(dict(chen_fid=int(m.chen_fid),source_chen_fids=m.source_group_chen_fids,
            historical_catchment_reach_id=int(m.catchment_reach_id),historical_model_reach_id=historic,
            historical_assignment_quality=m.reach_assignment_quality,current_catchment_reach_id=rid,
            nearest_archived_line_reach_id=nid,nearest_archived_line_distance_m=distance,
            current_assignment_quality=quality,current_candidate_reach_id=rid if retained else None,
            historical_inclusion_preserved=retained,mapping_status=state,
            current_spatial_version='H1_unified_20260915_v3',outfall_location_certified=False))
    result=pd.DataFrame(records)
    for c in ['historical_model_reach_id','current_catchment_reach_id','nearest_archived_line_reach_id','current_candidate_reach_id']:
        result[c]=result[c].astype('Int64')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--master',required=True)
    parser.add_argument('--h1-catchments',required=True)
    parser.add_argument('--archived-lines',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    master=pd.read_parquet(args.master).sort_values('chen_fid').reset_index(drop=True)
    catchments=gpd.read_file(args.h1_catchments).set_index('reach_id').sort_index()
    lines=gpd.read_file(args.archived_lines).set_index('reach_id').sort_index()
    map_facilities(master,catchments,lines).to_csv(args.output,index=False,lineterminator='\n')
