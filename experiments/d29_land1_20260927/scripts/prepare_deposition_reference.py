"""Preserve strict product interface and separately seal a monthly-only scenario."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np,pandas as pd
from d29_platform.deposition import distribute_month
OLD=ROOT.parent/'20260926_1/outputs/deposition';OUT=ROOT/'outputs/deposition_reference';OUT.mkdir(parents=True,exist_ok=True)
def main():
    ok,res=dispatch_allowed(reserve_bytes=600_000_000)
    if not ok:raise RuntimeError('RESOURCE_PAUSED')
    mass=np.load(OLD/'monthly_land_components_kg_n.npy');rain=np.load(OLD/'h1_precipitation_mm.npy',mmap_mode='r')
    dates=pd.DatetimeIndex(np.load(OLD/'daily_dates.npy'));ids=np.load(OLD/'reach_ids.npy')
    out=np.lib.format.open_memmap(OUT/'daily_land_components_uniform_kg_n.npy',mode='w+',dtype='float64',shape=(len(dates),len(ids),4))
    month_err=[];conflicts=[];worst_strict=0.
    previous=np.load(OLD/'daily_land_components_kg_n.npy',mmap_mode='r')
    for j,ym in enumerate(pd.period_range('1961-01','2024-12',freq='M')):
        ix=np.flatnonzero((dates.year==ym.year)&(dates.month==ym.month))
        strict,c=distribute_month(mass[j],rain[ix],ym.year,ym.month)
        if not np.array_equal(np.isnan(strict),np.isnan(previous[ix])):raise AssertionError('STRICT_MISSING_MASK_CHANGED')
        worst_strict=max(worst_strict,float(np.nanmax(abs(strict-previous[ix]))))
        conflicts.extend(c)
        uniform,_=distribute_month(mass[j],rain[ix],ym.year,ym.month,mode='monthly_uniform_reference')
        out[ix]=uniform
        month_err.append(float(np.max(abs(uniform.sum(0)-mass[j]))))
    out.flush()
    # A controlled zero-rain case must remain blocked in strict mode.
    example=np.ones((1,4));zero=np.zeros((29,1))
    strict,c=distribute_month(example,zero,2024,2)
    check=bool(np.isnan(strict[:,:,2:]).all() and len(c)==2)
    uniform,_=distribute_month(example,zero,2024,2,mode='monthly_uniform_reference')
    check &= bool(np.allclose(uniform.sum(0),example,rtol=0,atol=1e-12))
    receipt=dict(passed=check and max(month_err)<=1e-6 and worst_strict<=1e-10,
        scenario_id='input4MIPs_monthly_uniform_reference_all_months',
        strict_H1_conflicts_preserved=len(conflicts),strict_regression_max_kg=worst_strict,
        monthly_closure_max_kg=max(month_err),days=len(dates),reaches=len(ids),
        source_sha256=sha(OLD/'monthly_land_components_kg_n.npy'),output_sha256=sha(OUT/'daily_land_components_uniform_kg_n.npy'),
        scientific_status='Research timing assumption from monthly information only; neither observed daily deposition nor corrected H1 precipitation.',
        land_entry_status='Aggregated land mass retains original scope. Land-class/impervious/unknown mapping must be applied before LAND1 soil entry.',
        source_definition='https://input4mips-cvs.readthedocs.io/en/latest/dataset-overviews/nitrogen-deposition/',
        limitations=['2023–2024 ScenarioMIP','early CLCD1985 mask scenario','does not resolve source meteorology versus H1 meteorology discrepancy'],
        NSE='not applicable')
    write_json(OUT/'receipt.json',receipt);print({k:receipt[k] for k in ['passed','strict_H1_conflicts_preserved','strict_regression_max_kg','monthly_closure_max_kg']},flush=True)
    if not receipt['passed']:raise AssertionError('DEPOSITION_REFERENCE_FAILED')
if __name__=='__main__':main()
