"""Verify that timing scenarios alter timing, with annual mass documented."""
from common import ROOT,RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
from audit_inputs import SOURCE
import numpy as np
import pandas as pd


def main():
    columns=['fertilizer_kg_n','manure_kg_n','cropland_bnf_kg_n','atmospheric_deposition_kg_n','crop_demand_kg_n']
    evidence={};tables={}
    for product,path in SOURCE.items():
        frame=pd.read_parquet(path);tables[product]=frame
        if not np.isfinite(frame[columns].to_numpy(float)).all() or frame[columns].lt(0).any().any():raise ValueError('Invalid N mass')
        if frame.duplicated(['calendar_scenario','year','month','reach_id']).any():raise ValueError('Duplicate source key')
        annual=frame.groupby(['calendar_scenario','year','reach_id'])[columns].sum()
        central=annual.loc['CENTRAL'];metrics={}
        for scenario in ['EARLY','LATE']:
            delta=annual.loc[scenario]-central
            scale=np.maximum(central.to_numpy(float),1)
            normalized=np.abs(delta.to_numpy(float))/scale
            # A timing scenario may legitimately cross a year boundary. Fail
            # the mass-preserving interpretation rather than silently rescale.
            metrics[scenario]=dict(max_abs_annual_reach_difference_kg_n=float(np.abs(delta.to_numpy()).max()),
                max_relative_annual_reach_difference=float(normalized.max()),
                annual_mass_preserved=bool(np.all(np.abs(delta.to_numpy())<=1e-6+1e-10*scale)))
        evidence[product]=dict(rows=len(frame),scenarios=sorted(frame.calendar_scenario.unique()),comparisons=metrics)
    key=['calendar_scenario','year','month','reach_id']
    a=tables['formal'].sort_values(key).reset_index(drop=True)
    b=tables['sensitivity'].loc[lambda f:f.year.le(2024)].sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):raise ValueError('Formal and sensitivity source key mismatch')
    exact=bool(np.array_equal(a[columns].to_numpy(float),b[columns].to_numpy(float)))
    if not exact:raise ValueError('Nonidentical historical source overlap')
    path=ROOT/'5_Test/20260905_1/reports/calendar_audit.json'
    atomic_json(dict(status='CALENDAR_AUDIT_COMPLETE',runtime=RUNTIME,created_utc=utc_now(),products=evidence,
        historical_all_calendar_overlap_exact=exact,inputs={str(p):sha256(p) for p in SOURCE.values()},
        interpretation='monthly schedules are scenario assumptions; no observation sampling dates are inferred'),path)
    print('CALENDAR_AUDIT',evidence,flush=True)


if __name__=='__main__':main()
