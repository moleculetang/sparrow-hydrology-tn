"""Read-only LUH3 area operator adapter. No nitrogen density/fate is inferred."""
from pathlib import Path
import numpy as np
import pandas as pd

LAND_STATES=('primf','primn','secdf','secdn','urban','c3ann','c4ann','c3per','c4per','c3nfx','pastr','range')

def load_area_transfers(states_path, operator_path, dates):
    states=pd.read_csv(states_path)
    op=pd.read_csv(operator_path)
    dates=pd.DatetimeIndex(dates)
    land=states[states.state.isin(LAND_STATES)]
    index=pd.MultiIndex.from_product([range(1961,2025),range(1,231),LAND_STATES],names=['year','reach_id','state'])
    area=land.set_index(['year','reach_id','state']).physical_area_m2.reindex(index)
    if area.isna().any() or (area<0).any():raise ValueError('INCOMPLETE_LUH3_PHYSICAL_AREA')
    area=area.to_numpy().reshape(64,230,12)
    matrices={};state_index={s:i for i,s in enumerate(LAND_STATES)}
    for year,g in op.groupby('year',sort=True):
        if year not in range(1961,2024):raise ValueError('OPERATOR_YEAR_OUTSIDE_HISTORY')
        if not set(g.source_state)|set(g.target_state)<=set(LAND_STATES):raise ValueError('UNKNOWN_LAND_STATE')
        matrix=np.zeros((230,12,12),dtype=np.float64)
        r=g.reach_id.to_numpy(dtype=int)-1
        i=g.source_state.map(state_index).to_numpy();j=g.target_state.map(state_index).to_numpy()
        f=g.source_fraction.to_numpy(dtype=float)
        if not np.isfinite(f).all() or (f<0).any():raise ValueError('INVALID_SOURCE_FRACTION')
        np.add.at(matrix,(r,i,j),f)
        old=area[int(year)-1961]
        # Known empty area, not unknown input: identity is the neutral extension.
        empty=(old==0)&(matrix.sum(-1)==0)
        er,ei=np.where(empty);matrix[er,ei,ei]=1.
        if not np.allclose(matrix.sum(-1),1.,rtol=0,atol=1e-12):raise ValueError('INCOMPLETE_TRANSFER_ROWS')
        effective=pd.Timestamp(int(year)+1,1,1)
        match=np.flatnonzero(dates==effective)
        if len(match)!=1:raise ValueError('TRANSFER_EFFECTIVE_DATE_MISSING')
        matrices[int(match[0])]=matrix
    if len(matrices)!=63:raise ValueError('MISSING_YEAR_TRANSITIONS')
    return area,matrices

def require_nitrogen_transfer_policy(policy):
    if policy is None or policy.get('identity')!='conservative_relocation_scenario':
        raise ValueError('AREA_TRANSFER_IS_NOT_NITROGEN_FATE_POLICY')
    if not policy.get('no_claim_of_observed_nitrogen_fates'):
        raise ValueError('SCENARIO_NOT_IDENTIFIED')
    return True
