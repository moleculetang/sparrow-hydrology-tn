from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from d29_platform.source_definitions import *
from d29_platform.runtime import write_json,sha
OLD=ROOT.parent/'20260926_1/outputs';OUT=ROOT/'outputs/input_conflicts';OUT.mkdir(parents=True,exist_ok=True)
checks=[]
def check(name,value):
    checks.append(dict(name=name,passed=bool(value)))
def rejects(name,fn):
    try:fn();check(name,False)
    except ValueError:check(name,True)

# Independent hand arithmetic, selected published equation values and adversarial cases.
check('soil_units_2g_perkg_1p3g_cm3_10pct_rock_30cm',np.isclose(soil_total_n_stock_kg_m2(200,130,100,.3),.702,rtol=0,atol=1e-14))
check('rice_harvested_area_no_second_cropping_multiplier',crop_bnf_kg('rice',200,1200)==5000)
check('soybean_formula_asia',np.isclose(crop_bnf_kg('soybean',100,250),250000/(.2775+.1178*np.log(2.5))*(.03913-.00118*2.5/(.2775+.1178*np.log(2.5)))*1.4*.61))
check('zero_area_known_zero_legume',crop_bnf_kg('soybean',0,0)==0)
rejects('unknown_BNF_not_zero',lambda:crop_bnf_kg('unknown',100,20))
rejects('missing_yield_not_imputed',lambda:crop_bnf_kg('soybean',100,np.nan))
rejects('zero_yield_no_epsilon',lambda:crop_bnf_kg('soybean',100,0))
rejects('soil_invalid_rock_rejected',lambda:soil_total_n_stock_kg_m2(200,130,1001,.3))
rejects('residue_double_count_rejected',lambda:mutually_exclusive_residue_fates(100,returned=.76,removed=.4,burned=0,grazed=0,other=0))
check('national_growth_applied_once',np.isclose(apply_national_ratio_once(100,110,100,[20,30]).sum(),110))
check('area_scale_only_changes_distribution',np.array_equal(apply_national_ratio_once(100,110,100,[20,30]),apply_national_ratio_once(100,110,100,[40,60])))
rejects('positive_crop_area_missing_rate_not_zero',lambda:crop_mass_from_rates(2.,np.nan,unit='kg N ha-1 yr-1'))
rejects('area_NaN_not_automatically_absence',lambda:crop_mass_from_rates(np.nan,20.,unit='kg N ha-1 yr-1'))
rejects('rate_mass_filename_not_unit_proof',lambda:crop_mass_from_rates(2.,20.,unit='kg N per grid'))
rejects('known_zero_contradiction_rejected',lambda:crop_mass_from_rates(2.,20.,unit='kg N ha-1 yr-1',known_zero_rate=True))
check('documented_zero_area_no_rate_needed',crop_mass_from_rates(0.,np.nan,unit='kg N ha-1 yr-1')==0)
check('explicit_absence_mask_not_nan_fill',crop_mass_from_rates(np.nan,np.nan,unit='kg N ha-1 yr-1',known_no_crop=True)==0)

# Recompute wet conflicts from monthly mass and full daily H1 rain, not old count.
mass=np.load(OLD/'deposition/monthly_land_components_kg_n.npy')
rain=np.load(OLD/'deposition/h1_precipitation_mm.npy',mmap_mode='r')
dates=pd.DatetimeIndex(np.load(OLD/'deposition/daily_dates.npy'))
reaches=np.load(OLD/'deposition/reach_ids.npy');rows=[]
for mi,period in enumerate(pd.period_range('1961-01','2024-12',freq='M')):
    sel=(dates.year==period.year)&(dates.month==period.month);rr=rain[sel]
    missing=~np.isfinite(rr).all(axis=0);zero=rr.sum(axis=0)==0
    for ri,ci in np.argwhere((mass[mi,:,2:]>0)&(missing|zero)[:,None]):
        rows.append(dict(year=period.year,month=period.month,reach_id=int(reaches[ri]),component=('wetnhx','wetnoy')[ci],unresolved_kg_n=float(mass[mi,ri,ci+2]),reason='missing_month_precipitation' if missing[ri] else 'zero_month_precipitation'))
conf=pd.DataFrame(rows);conf.to_csv(OUT/'wet_conflicts_independent.csv',index=False)
oldconf=pd.read_csv(OLD/'deposition/daily_wet_conflicts.csv')
keys=['year','month','reach_id','component'];join=conf.merge(oldconf,on=keys,suffixes=('_new','_old'),how='outer',indicator=True)
check('wet_conflict_identities_match',join['_merge'].eq('both').all())
err=float(abs(join.unresolved_kg_n_new-join.unresolved_kg_n_old).max());check('wet_conflict_mass_match',err<=1e-6)
stats=dict(component_cells=len(conf),unique_reach_months=len(conf.drop_duplicates(['year','month','reach_id'])),unallocated_mass_kg=float(conf.unresolved_kg_n.sum()),independent_mass_error_kg=err,rule='Blocked in H1-rain disaggregation; no automatic epsilon, deletion or cross-month shift.')

npp=pd.read_csv(OLD/'mod17_gee_reachcover/reach_clcd_class_annual_npp_qc_2001_2025.csv')
# CLCD 2 forest,3 shrub,4 grass,6 snow,7 barren,8 impervious,9 wetland.
veg=npp[npp.year.between(2001,2024)&npp.clcd_class.isin([2,3,4,9])]
summary=[]
for year,g in veg.groupby('year'):
    d={'year':int(year),'physical_class_area_m2':float(g.clcd_class_area_m2.sum()),'negative_npp_area_m2':float(g.npp_negative_area_m2.sum()),'missing_npp_area_m2':float(g.npp_missing_area_m2.sum())}
    for threshold in (0,10,20,50,100):
        d[f'qc_le_{threshold}_coverage']=float(g[f'qc_le_{threshold}_area_m2'].sum()/g.clcd_class_area_m2.sum())
    summary.append(d)
pd.DataFrame(summary).to_csv(OUT/'noncropland_npp_qc_coverage.csv',index=False)
sp=pd.read_csv(OLD/'spatial_topology/reach_input_spatial_join_230.csv')
sp[abs(sp.H1_area_rel_diff)>.01].to_csv(OUT/'H1_geometry_support_conflicts.csv',index=False)
fa=pd.read_csv(OLD/'faostat_crops/faostat_china_crop_activity_1961_2024.csv')
fa[fa.value.isna()].to_csv(OUT/'FAOSTAT_unknown_values.csv',index=False)
receipt=dict(passed=all(x['passed'] for x in checks),checks=checks,wet_deposition=stats,
    faostat_unknown_rows=int(fa.value.isna().sum()),H1_area_conflict_reaches=int((abs(sp.H1_area_rel_diff)>.01).sum()),
    npp_scope='noncropland forest/shrub/grass/wetland; crop excluded; no negative values clipped',
    source_hashes={str(p):sha(p) for p in [OLD/'deposition/monthly_land_components_kg_n.npy',OLD/'deposition/h1_precipitation_mm.npy',ROOT/'d29_platform/source_definitions.py']},
    NSE='not applicable: source quantities and software checks; not concentration skill',
    completed_real_model_run=False)
write_json(OUT/'definition_audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],'checks':len(checks),'wet':stats,'faostat_unknown':receipt['faostat_unknown_rows']},ensure_ascii=False),flush=True)
if not receipt['passed']:raise AssertionError('SOURCE_DEFINITION_AUDIT_FAILED')
