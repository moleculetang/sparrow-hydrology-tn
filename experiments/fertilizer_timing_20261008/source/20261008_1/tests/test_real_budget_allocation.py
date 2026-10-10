"""Real all-crop budgets; no TN used and no simulation scores computed."""
import sys,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from timing.inputs import TimingInputs
from d29_platform.runtime import write_json

def main():
    start=time.monotonic();dates=pd.date_range('1961-01-01','2024-12-31')
    a=TimingInputs(dates);records=[];worst=dict(annual=0.,monthly=0.,crop=0.,plant=0.)
    years=list(range(2016,2025)) if '--all-years' in sys.argv else [2016,2021,2024]
    suffix='_all_intervention_years' if '--all-years' in sys.argv else ''
    for year in years:
        p=np.zeros((230,13));a.calendar='C0';base=a.annual(year,p)
        refmass=base['sources'].sum(0);del base
        a.calendar='C1';one=a.annual(year,p);month=dates[one['indices']].month
        monthly=np.array([one['sources'][month==m].sum(0) for m in range(1,13)])
        bycrop1={(r['reach_id'],r['crop'],r['source']):r for r in a.last_allocation}
        mass1=one['sources'].sum(0);del one
        a.calendar='C2';two=a.annual(year,p)
        crop_error=max(abs(r['budget_kg']-r['day_sum_kg']) for r in a.last_allocation)
        crop_month=max(max(abs(np.asarray(r['monthly_kg'])-bycrop1[r['reach_id'],r['crop'],r['source']]['monthly_kg'])) for r in a.last_allocation)
        annual=max(float(np.max(abs(mass1-refmass))),float(np.max(abs(two['sources'].sum(0)-refmass))))
        montherr=max(crop_month,float(np.max(abs(np.array([two['sources'][month==m].sum(0) for m in range(1,13)])-monthly))))
        assert crop_error<=1e-6 and annual<=1e-6 and montherr<=1e-6,(year,crop_error,annual,montherr)
        table=pd.DataFrame([{**r,**{f'month_{m:02d}_kg':r['monthly_kg'][m-1] for m in range(1,13)}} for r in a.last_allocation]).drop(columns='monthly_kg')
        folder=ROOT/'outputs/input_allocation';folder.mkdir(parents=True,exist_ok=True)
        table.to_parquet(folder/f'C2_{year}_all_crops.parquet',index=False)
        del two
        a.plant_version='V1';stage=a.annual(year,p)
        cap=stage['plant_growth_capacity'][:,:,5:10];out=stage['plant_outflows'][:,:,5:10].sum(-1)
        error=float(np.max(abs(cap.sum(0)-out.sum(0))))
        assert error<=1e-6 and stage['plant_target'].min()>=0,(year,error)
        assert np.all(stage['plant_target'][-1,:,5:10]==0.)
        worst['annual']=max(worst['annual'],annual);worst['monthly']=max(worst['monthly'],montherr)
        worst['crop']=max(worst['crop'],crop_error);worst['plant']=max(worst['plant'],error)
        records.append(dict(year=year,crops=int(table.crop.nunique()),rows=len(table),max_annual_error_kg=annual,
            max_C1_C2_month_error_kg=montherr,plant_plan_closure_kg=error,
            admitted_mass_kg=float(table[table.admitted_date_evidence].budget_kg.sum()),
            unsupported_mass_kg=float(table[~table.admitted_date_evidence].budget_kg.sum())))
        write_json(ROOT/f'evidence/real_budget_allocation{suffix}_progress.json',dict(year=year,records=records))
        a.plant_version='V0';del stage
    receipt=dict(passed=True,scope='all_intervention_years_all_crops_not_full_history_gate' if suffix else 'three_real_years_all_crops_not_full_history_gate',records=records,
       maxima_kg=worst,seconds=time.monotonic()-start,no_evaluation_labels=True,coverage_is_not_mass_multiplier=True)
    write_json(ROOT/f'evidence/real_budget_allocation{suffix}.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
