"""Read-only product reconstruction and synthetic historical readout checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from rebuild import rebuild_reach
from legacy_readout import concentration,fit_readout

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def same(actual,stored,keys):
    a=actual.sort_values(keys).reset_index(drop=True)
    b=stored.sort_values(keys).reset_index(drop=True)
    assert_frame_equal(a,b,check_exact=True)

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        p=ROOT/row['path']
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise RuntimeError('FILE_BYTE_MISMATCH: '+row['path'])
    read=lambda name:pd.read_parquet(ROOT/'data'/(name+'.parquet'))
    anchor=read('wwtp_reach_province_2012_tdn_anchor');factors=read('wang_province_year_temporal_factors_2006_2019')
    annual,reach=rebuild_reach(anchor,factors)
    stored_annual=read('prb_wwtp_tn_annual_reach_2006_2019')
    stored_reach=read('prb_wwtp_tn_monthly_reach_2006_2019')
    def mass_compare(a,b,keys,column):
        a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
        assert_frame_equal(a.drop(columns=column),b[a.columns].drop(columns=column),check_exact=True)
        error=float(np.max(np.abs(a[column].to_numpy()-b[column].to_numpy())))
        if error>1e-6:raise RuntimeError('REBUILT_MASS_EXCEEDS_ORIGINAL_ABSOLUTE_TOLERANCE')
        return error
    annual_error=mass_compare(annual,stored_annual,['model_reach_id','year','tn_scenario'],'tn_load_kg_n_yr')
    rebuild_error=mass_compare(reach,stored_reach,['model_reach_id','year','month','tn_scenario'],'tn_load_kg_n_month')
    if len(anchor)!=113 or len(annual)!=4704 or len(reach)!=56448:raise RuntimeError('WRONG_PRODUCT_COVERAGE')
    if not set(annual.year)==set(range(2006,2020)) or reach.model_reach_id.nunique()!=112:raise RuntimeError('WRONG_YEAR_REACH_SUPPORT')
    if (reach.tn_load_kg_n_month<0).any():raise RuntimeError('NEGATIVE_REACH_MASS')
    central=annual[(annual.year==2012)&(annual.tn_scenario=='tdn_equals_tn_lower_bound')]
    reference=anchor.groupby('model_reach_id').tdn_2012_t_n_yr.sum().reindex(central.model_reach_id)
    anchor_error=float(np.max(np.abs(central.tn_load_kg_n_yr.to_numpy()/1000-reference.to_numpy())))
    keys=['model_reach_id','year','tn_scenario']
    sums=reach.groupby(keys,as_index=False).tn_load_kg_n_month.sum()
    joined=sums.merge(annual,on=keys,validate='one_to_one')
    month_error=float((joined.tn_load_kg_n_month-joined.tn_load_kg_n_yr).abs().max())
    if anchor_error*1000>1e-6 or month_error>1e-6:raise RuntimeError('PRODUCT_MASS_CLOSURE')
    for year,days in [(2008,29),(2009,28)]:
        feb=reach[(reach.year==year)&(reach.month==2)]
        a=annual[annual.year==year]
        check=feb.merge(a,on=keys,validate='one_to_one')
        fraction=days/(366 if year==2008 else 365)
        if np.max(np.abs(check.tn_load_kg_n_month-check.tn_load_kg_n_yr*fraction))>1e-6:raise RuntimeError('LEAP_CALENDAR_ERROR')
    params=pd.read_parquet(ROOT/'fit/candidate_fold_readout_parameters.parquet')
    if len(params)!=240 or set(params.scenario_id)!={'PS0','PS1','PS2','PS3','PS09'}:raise RuntimeError('FIT_RECEIPT_COVERAGE')
    if not params[['eta_quick','eta_gw']].ge(0).all().all() or not params[['eta_quick','eta_gw']].le(1).all().all():raise RuntimeError('FIT_PARAMETER_BOUNDS')
    # Original-source runs were independently compared locally before release.
    # Public package retains synthetic expected fits, never real station records.
    expected_fits={'P1':[0.5029644728932853,0.7371124306824147],'P2':[0.5538139140748352,0.7685237731076479]}
    n=80;index=np.arange(n)
    frame=pd.DataFrame(dict(routed_quick_tn_kg_n=20+index%11,routed_gw_tn_kg_n=10+index%7,
        routed_wwtp_tn_kg_n=3+index%5,routed_water_volume_m3=np.full(n,20000.),station_key=np.where(index%2,'synthetic_A','synthetic_B')))
    direct=concentration(frame,np.zeros(2),'routed_wwtp_tn_kg_n')
    expected=frame.routed_wwtp_tn_kg_n.to_numpy()*1000/frame.routed_water_volume_m3.to_numpy()
    np.testing.assert_array_equal(direct,expected)
    for eta in [np.array([.1,.9]),np.array([.8,.3])]:
        delta=concentration(frame,eta,'routed_wwtp_tn_kg_n')-concentration(frame,eta)
        np.testing.assert_allclose(delta,expected,rtol=0,atol=1e-14)
    frame['tn_mg_l']=concentration(frame,np.array([.5,.7]),'routed_wwtp_tn_kg_n')
    results={}
    for layer in ['P1','P2']:
        a=fit_readout(frame,layer,'routed_wwtp_tn_kg_n')
        np.testing.assert_allclose(a[0],expected_fits[layer],rtol=0,atol=1e-6)
        if not a[2]['success']:raise RuntimeError('SYNTHETIC_READOUT_REGRESSION')
        results[layer]=dict(eta=a[0].tolist(),success=a[2]['success'],nfev=a[2]['nfev'])
    receipt=dict(passed=True,scope='historical_WWTP_product_and_synthetic_readout_not_current_training_acceptance',
        file_hashes_verified=len(manifest['files']),rebuild_scope=['reach_annual','reach_month'],
        counts=dict(reach_province_anchors=len(anchor),reach_annual=len(annual),reach_month=len(reach),fit_parameters=len(params)),
        central_2012_error_t_n=anchor_error,month_year_error_kg_n=month_error,annual_rebuild_error_kg_n=annual_error,monthly_rebuild_error_kg_n=rebuild_error,
        synthetic_fits=results,wwtp_unscaled_by_eta_verified=True,real_monitoring_read=False,current_science_changed=False,
        historical_fit_success_count=int(params.success.sum()),nse='not_applicable_to_product_mass_or_parameter_verification')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
