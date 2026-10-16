"""Frozen all-crop budget -> month -> day allocation, without water-quality reads.

Daily vectors are stored once per unique year/profile, budgets refer to them.
The exact residual/index used by the production generator is saved separately,
so a compact artifact reconstructs every actual daily crop/source mass vector.
There is no full-history crop/land/source tensor and no invented calendar.
"""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import write_json,sha


def allocation(budget,weights,month):
    weights=np.asarray(weights,dtype=np.float64);month=np.asarray(month,int)
    if not np.isfinite(budget) or budget<0 or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('FINITE_NONNEGATIVE_BUDGET_AND_PROFILE_REQUIRED')
    if weights.ndim!=1 or not len(weights) or weights.shape!=month.shape or abs(float(weights.sum())-1.)>1e-12:
        raise ValueError('NORMALIZED_DAY_PROFILE_REQUIRED')
    daily=float(budget)*weights;index=int(np.argmax(weights));residual=float(budget)-float(daily.sum())
    daily[index]+=residual
    if daily.min()<0:raise ValueError('NEGATIVE_DAILY_ALLOCATION')
    masses=np.array([daily[month==m].sum() for m in range(1,13)])
    if abs(float(daily.sum())-budget)>1e-6 or abs(float(masses.sum())-budget)>1e-6:
        raise ValueError('ANNUAL_DAILY_MONTHLY_MASS_NOT_CLOSED')
    return daily,masses,index,residual


def reconstruct(row,weights,month,conditional_month=None):
    """Return exact days, or conditional q; a zero-mass month has no q."""
    daily=np.asarray(weights,dtype=np.float64)*float(row['budget_kg'])
    daily[int(row['residual_day_index'])]+=float(row['residual_kg'])
    if conditional_month is None:return daily
    mask=np.asarray(month)==conditional_month;total=float(daily[mask].sum())
    return None if total==0. else daily[mask]/total


def main(jid):
    from train_job import validate_admission
    from timing.inputs import TimingInputs
    _,gate_hash=validate_admission()
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    folder=ROOT/'outputs/jobs'/jid;freeze=json.loads((folder/'prediction_freeze.json').read_text(encoding='utf8'))
    point=folder/'frozen_parameters.npy'
    if not freeze.get('numerically_sufficient') or freeze['formal_gate_sha256']!=gate_hash or sha(point)!=freeze['parameter_sha256']:
        raise RuntimeError('SUFFICIENT_IMMUTABLE_FROZEN_POINT_REQUIRED')
    x=np.load(point,allow_pickle=False)
    a=TimingInputs(pd.date_range('1961-01-01','2024-12-31'),job['calendar'],job['plant_version'],x[22:])
    if x.shape!=(22+len(a.names),):raise RuntimeError('FIXED_MASS_COORDINATE_IDENTITY')
    destination=folder/'input_allocation';destination.mkdir(exist_ok=True)
    artifacts={};records=[];maxima=dict(annual=0.,month_sum=0.,reconstruction=0.)
    for year in range(1961,2025):
        a._profiles.clear();dates=a._year_dates[year];months=dates.month.to_numpy()
        budgets=a.ag[a.ag.year==year];profiles={};rows=[]
        if budgets.duplicated(['reach_id','crop','field']).any():raise RuntimeError('DUPLICATE_ANNUAL_SOURCE')
        if budgets.mass_kg.isna().any() or (budgets.mass_kg<0).any():raise RuntimeError('UNKNOWN_OR_NEGATIVE_BUDGET')
        for item in budgets.itertuples():
            weights,_,identity,admitted=a._daily_profile(year,item.crop,item.field,int(item.reach_id))
            daily,monthly,index,residual=allocation(float(item.mass_kg),weights,months)
            profile_id=hashlib.sha256(weights.astype('<f8',copy=False).tobytes()).hexdigest()
            if profile_id not in profiles:
                profiles[profile_id]=dict(profile_id=profile_id,daily_weights=weights.tolist(),
                    monthly_calendar_weights=[float(weights[months==m].sum()) for m in range(1,13)])
            row=dict(year=year,reach_id=int(item.reach_id),crop=item.crop,source=item.field,
                budget_kg=float(item.mass_kg),profile_id=profile_id,residual_day_index=index,residual_kg=residual,
                monthly_kg=monthly.tolist(),month_mass_status=['positive_mass' if v>0 else 'not_applicable_zero_mass' for v in monthly],
                pi=(monthly/float(item.mass_kg)).tolist() if item.mass_kg>0 else None,
                annual_identity='frozen_input_product_no_source_multiplier',date_identity=identity,
                admitted_date_evidence=bool(admitted),calendar_profile_applied=bool(admitted and job['calendar']!='C0'),
                original_area_year=int(item.original_area_year),
                rate_year=int(item.rate_year),rate_donor_identity=item.rate_donor_identity,
                daily_mass_sha256=hashlib.sha256(daily.astype('<f8',copy=False).tobytes()).hexdigest())
            restored=reconstruct(row,weights,months)
            if not np.array_equal(restored,daily):raise RuntimeError('DAILY_ALLOCATION_RECONSTRUCTION_FAILED')
            maxima['annual']=max(maxima['annual'],abs(float(daily.sum())-float(item.mass_kg)))
            maxima['month_sum']=max(maxima['month_sum'],abs(float(monthly.sum())-float(item.mass_kg)))
            rows.append(row)
        budgetfile=destination/f'budgets_{year}.parquet';profilefile=destination/f'profiles_{year}.parquet'
        pd.DataFrame(rows).to_parquet(budgetfile,index=False)
        pd.DataFrame(list(profiles.values())).to_parquet(profilefile,index=False)
        for p in (budgetfile,profilefile):artifacts[p.name]=sha(p)
        records.append(dict(year=year,crops=int(budgets.crop.nunique()),records=len(rows),unique_profiles=len(profiles),
            annual_budget_kg=float(budgets.mass_kg.sum()),evidence_admitted_budget_kg=float(sum(r['budget_kg'] for r in rows if r['admitted_date_evidence'])),
            unsupported_uniform_budget_kg=float(sum(r['budget_kg'] for r in rows if not r['admitted_date_evidence']))))
        write_json(destination/'progress.json',dict(year=year,records=records,maxima_kg=maxima))
    receipt=dict(passed=max(maxima.values())<=1e-6,job=jid,scope='all64year_crop_source_allocation_from_frozen_mass_and_calendar_not_new_fit',
        parameter_sha256=sha(point),formal_gate_sha256=gate_hash,source_multiplier=1.,calendar=job['calendar'],
        daily_representation='budget * stored daily profile, then residual at recorded index; float64 little-endian digest',
        C0_scope='per-crop reconstruction of uniform support; aggregate parent implementation may differ by tolerated summation roundoff',
        month_support='pi=monthly mass/annual budget for positive annual budget; q=day mass/month mass only on positive-mass months',
        zero_annual_budget='known zero, pi not applicable; daily mass exactly zero',units='kg N',
        manure_entry_ratios=a.config['manure'],BNF_and_deposition_unchanged=True,records=records,maxima_kg=maxima,artifact_sha256=artifacts,
        evidence_available_is_not_profile_applied_for_C0=True,
        observation_data_read=False,identities={p:sha(ROOT/p) for p in ('timing/inputs.py','timing/profiles.py','scripts/export_timing_allocations.py')})
    write_json(destination/'complete.json',receipt);print(json.dumps(dict(passed=receipt['passed'],years=len(records),files=len(artifacts),maxima=maxima)))


if __name__=='__main__':main(sys.argv[1])
