"""Causal station-geometry features, with shared local and upstream information.

The only station-dependent quantities are physical location/support and frozen
water at the observation boundary. No categorical station identifier is a
model feature. Source rates remain frozen S1 and do not use observed TN.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from fc_data import GEOMETRY_FIELDS, Scaler, daily_drivers, causal_rolling_mean, training_support
from fc_legacy import route


def geometry_keys(metadata):
    return [tuple(row) for row in metadata[GEOMETRY_FIELDS].itertuples(index=False, name=None)]


def geometries(metadata):
    result=metadata[GEOMETRY_FIELDS].drop_duplicates().sort_values(GEOMETRY_FIELDS).reset_index(drop=True)
    return result


def locate_geometry(metadata, geometry):
    lookup={key:i for i,key in enumerate(geometry_keys(geometry))}
    return np.array([lookup[k] for k in geometry_keys(metadata)],dtype=np.int64)


@dataclass
class FeatureSet:
    geometry: pd.DataFrame
    static: np.ndarray
    dynamic: np.ndarray
    static_fields: list
    dynamic_fields: list
    starts: np.ndarray
    stops: np.ndarray


def raw_features(data, metadata):
    if 'tn_mg_l' in metadata:
        raise ValueError('Feature construction takes label-free metadata only')
    geometry=geometries(metadata)
    ri=geometry.reach_id.to_numpy(int)-1
    local,fields=daily_drivers(data)
    support=data.support.astype(float).copy()
    np.fill_diagonal(support,0.)
    area=support@data.area_ha
    weights=np.divide(support*data.area_ha[None,:],area[:,None],out=np.zeros_like(support),where=area[:,None]>0)
    # Compute in month-sized blocks to avoid a full 230-reach upstream cube.
    nr=len(geometry);nd=len(data.dates);nf=len(fields)
    dynamic=np.empty((nd,nr,2*nf+1),dtype=np.float32)
    for a,b in zip(data.starts,data.stops):
        dynamic[a:b,:,:nf]=local[a:b,ri]
        dynamic[a:b,:,nf:2*nf]=np.einsum('qr,drk->dqk',weights[ri],local[a:b],optimize=True)
    del local
    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    types=geometry.station_type.to_numpy()
    fraction=np.where(types=='predam',1.,geometry.downstream_fraction_on_reach.to_numpy(float))
    ridx=np.maximum(0,geometry.reservoir_index.to_numpy(int))
    volume=water['inlet'][:,ri]+fraction[None,:]*(data.fast_water+data.slow_water)[:,ri]
    volume=np.where((types=='dam_outlet')[None,:],water['releases'][:,ridx],
                    np.where((types=='postdam_mixed')[None,:],water['official'][:,ri],volume))
    boundary_area=area[ri]+fraction*data.area_ha[ri]
    # Pure upstream headwaters can have zero support. Expose zero volume as zero
    # and retain explicit support area; no fitted water information is invented.
    depth=np.divide(volume,10*boundary_area[None,:],out=np.zeros_like(volume),where=boundary_area[None,:]>0)
    dynamic[:,:,-1]=np.log1p(depth)
    static=np.concatenate([data.static_raw[ri],weights[ri]@data.static_raw,
                           np.log1p(data.area_ha[ri,None]),np.log1p(area[ri,None]),fraction[:,None],
                           np.column_stack([types=='predam',types=='dam_outlet',types=='postdam_mixed'])],axis=-1)
    static_fields=['local_'+n for n in data.static_fields]+['upstream_'+n for n in data.static_fields]+[
        'log_local_area_ha','log_upstream_area_ha','downstream_fraction','is_predam','is_dam_outlet','is_postdam_mixed']
    dynamic_fields=['local_'+n for n in fields]+['upstream_'+n for n in fields]+['log_boundary_mm']
    assert np.isfinite(dynamic).all() and np.isfinite(static).all()
    return FeatureSet(geometry,static,dynamic,static_fields,dynamic_fields,data.starts,data.stops)


def fit_scalers(features, data, train):
    positions=np.unique(locate_geometry(train,features.geometry))
    # Time-only floor: observed training years, and physical geometry training
    # support only. No heldout station rows or values enter scale estimates.
    day_mask=(data.dates.year>=int(train.year.min()))&(data.dates.year<=int(train.year.max()))
    static=Scaler.fit(features.static[positions],clip=False)
    dynamic=Scaler.fit(features.dynamic[day_mask][:,positions],clip=False)
    return {'static':static.as_dict(),'dynamic':dynamic.as_dict(),
            'training_observation_ids':train.observation_id.tolist(),
            'training_geometry_indices':positions.tolist(),
            'training_support_reach_ids':(training_support(data,train)+1).tolist(),
            'training_year_range':[int(train.year.min()),int(train.year.max())],
            'static_fields':features.static_fields,'dynamic_fields':features.dynamic_fields}


def monthly_tree_features(features):
    x=features.dynamic.astype(np.float64)
    groups=[];names=[]
    for label,fn in [('mean',np.mean),('sd',np.std),('min',np.min),('max',np.max)]:
        monthly=np.stack([fn(x[a:b],axis=0) for a,b in zip(features.starts,features.stops)])
        groups.append(monthly);names.extend([f'current_{label}_{n}' for n in features.dynamic_fields])
    means=groups[0]
    # Current month stats and strictly past 1/3/12/60 months are distinct.
    for width in [1,3,12,60]:
        groups.append(causal_rolling_mean(means,width,include_current=False))
        names.extend([f'past{width}_mean_{n}' for n in features.dynamic_fields])
    groups.append(np.broadcast_to(features.static,(len(means),)+features.static.shape))
    names.extend(features.static_fields)
    return np.concatenate(groups,axis=-1).astype(np.float32),names


def gather_tree_rows(cube, metadata, geometry):
    if 'tn_mg_l' in metadata:
        raise ValueError('Prediction gather takes label-free metadata only')
    ti=((metadata.year.to_numpy()-1961)*12+metadata.month.to_numpy()-1).astype(int)
    gi=locate_geometry(metadata,geometry)
    return cube[ti,gi]
