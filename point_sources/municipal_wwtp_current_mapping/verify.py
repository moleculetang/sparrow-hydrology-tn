"""Verify the published WWTP ID bridge; unknown years must stay unknown."""
import hashlib
import calendar
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def main():
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert hashlib.sha256((HERE/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    audit=json.loads((HERE/'mapping_audit.json').read_text(encoding='utf-8'))
    for path,key in [('spatial/H1_unified_20260915_v3/H1_unified_spatial_files.zip','h1_spatial_zip_sha256'),
                     ('point_sources/municipal_wwtp_2006_2019/data/wwtp_reach_province_2012_tdn_anchor.parquet','public_anchor_sha256')]:
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==audit['files'][key]
    reaches=pd.read_csv(HERE/'load_reach_id_to_current_h1.csv')
    fac=pd.read_csv(HERE/'facility_id_to_current_reach.csv',dtype={'source_chen_fids':str})
    units=pd.read_csv(HERE/'load_unit_to_current_reach.csv')
    avail=pd.read_csv(HERE/'independent_load_availability_2020_2021.csv')
    assert reaches.historical_load_reach_id.tolist()==reaches.current_h1_reach_id.tolist()==list(range(1,231))
    with zipfile.ZipFile(REPO/'spatial/H1_unified_20260915_v3/H1_unified_spatial_files.zip') as z:
        registry=pd.read_parquet(io.BytesIO(z.read('canonical/hydrology_reach_registry_v3.parquet'))).sort_values('reach_id')
    assert np.array_equal(reaches.topology_downstream_reach_id,registry.downstream_reach_id)
    assert np.max(np.abs(reaches.current_h1_catchment_area_km2.to_numpy()-registry.local_area_m2.to_numpy()/1e6))<1e-9
    assert len(fac)==267 and fac.chen_fid.is_unique
    originals=[int(x) for value in fac.source_chen_fids for x in value.split(',')]
    assert len(originals)==len(set(originals))==270
    assert all(int(r.chen_fid)==min(int(x) for x in r.source_chen_fids.split(',')) for r in fac.itertuples())
    historical=fac.historical_model_reach_id.notna()
    retained=fac.historical_inclusion_preserved
    assert historical.sum()==230 and retained.sum()==213
    assert fac.loc[retained,'current_candidate_reach_id'].eq(fac.loc[retained,'historical_model_reach_id']).all()
    assert fac.loc[retained,'current_candidate_reach_id'].eq(fac.loc[retained,'current_catchment_reach_id']).all()
    assert fac.loc[retained,'current_candidate_reach_id'].eq(fac.loc[retained,'nearest_archived_line_reach_id']).all()
    assert not fac.outfall_location_certified.any()
    lost=fac.loc[historical & ~retained]
    assert len(lost)==17 and lost.historical_model_reach_id.eq(16).all()
    assert lost.current_catchment_reach_id.isna().all() and lost.current_candidate_reach_id.isna().all()
    assert lost.mapping_status.eq('outside_current_domain').all()
    assert fac.loc[~retained,'current_candidate_reach_id'].isna().all()
    assert fac.mapping_status.value_counts().to_dict()==audit['counts']['facility_mapping_states']
    for r in reaches.itertuples():
        local=fac.loc[fac.historical_model_reach_id==r.historical_load_reach_id]
        assert len(local)==r.historical_accepted_facilities
        assert int(local.historical_inclusion_preserved.sum())==r.current_supported_historical_facilities
    anchor=pd.read_parquet(REPO/'point_sources/municipal_wwtp_2006_2019/data/wwtp_reach_province_2012_tdn_anchor.parquet')
    joined=units.merge(anchor,left_on=['historical_load_reach_id','province'],right_on=['model_reach_id','province'],validate='one_to_one',how='outer',indicator=True,suffixes=('_map','_anchor'))
    assert joined['_merge'].eq('both').all() and joined.facilities_map.eq(joined.facilities_anchor).all()
    assert len(units)==113 and units.load_unit_id.is_unique and units.current_h1_reach_id.nunique()==112
    assert units.historical_load_reach_id.eq(units.current_h1_reach_id).all()
    assert units.facilities.sum()==230 and units.current_supported_historical_facilities.sum()==213
    bad=units.current_supported_historical_facilities<units.facilities
    assert units.loc[bad,'spatial_support_status'].eq('blocked_some_anchors_outside_H1').all()
    assert not units.formal_input_enabled.any()
    assert len(avail)==460 and not avail.duplicated(['current_h1_reach_id','year']).any()
    assert sorted(avail.year.unique())==[2020,2021]
    assert all(sorted(group.current_h1_reach_id)==list(range(1,231)) for _,group in avail.groupby('year'))
    assert avail.tn_load_kg_n_yr.isna().all() and not avail.formal_input_enabled.any()
    assert avail.availability.eq('unknown_no_independent_product_located').all()
    assert audit['independent_requested_years_available'] is False
    annual=pd.read_parquet(HERE/'data/H1_wwtp_tn_annual_reach_2006_2019.parquet')
    monthly=pd.read_parquet(HERE/'data/H1_wwtp_tn_monthly_reach_2006_2019.parquet')
    current_anchor=pd.read_parquet(HERE/'data/H1_reach_province_2012_tdn_anchor.parquet')
    outside=pd.read_parquet(HERE/'data/outside_H1_historical_facility_loads_2006_2019.parquet')
    reconciliation=pd.read_csv(HERE/'remapping_mass_reconciliation.csv')
    legacy=REPO/'point_sources/municipal_wwtp_2006_2019'
    factors=pd.read_parquet(legacy/'data/wang_province_year_temporal_factors_2006_2019.parquet')
    spec=importlib.util.spec_from_file_location('independent_legacy_rebuild',legacy/'rebuild.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_a,expected_m=module.rebuild_reach(current_anchor.rename(columns={'current_h1_reach_id':'model_reach_id'}),factors)
    ak=['current_h1_reach_id','year','tn_scenario'];mk=ak+['month']
    def mass_error(actual,expected,keys,column):
        joined=actual.merge(expected.rename(columns={'model_reach_id':'current_h1_reach_id'}),on=keys,validate='one_to_one',how='outer',indicator=True,suffixes=('_actual','_expected'))
        assert joined['_merge'].eq('both').all()
        err=float(np.max(np.abs(joined[column+'_actual']-joined[column+'_expected'])))
        assert err<=1e-6,err
        return err
    annual_error=mass_error(annual,expected_a,ak,'tn_load_kg_n_yr')
    monthly_error=mass_error(monthly,expected_m,mk,'tn_load_kg_n_month')
    assert current_anchor.facilities.sum()==213 and len(current_anchor)==113
    assert len(annual)==4704 and len(monthly)==56448
    assert len(outside)==714 and set(outside.chen_fid)==set(lost.chen_fid)
    assert outside.historical_load_reach_id.eq(16).all()
    assert not outside.duplicated(['chen_fid','year','tn_scenario']).any()
    for table,column in [(annual,'tn_load_kg_n_yr'),(monthly,'tn_load_kg_n_month')]:
        assert sorted(table.year.unique())==list(range(2006,2020))
        assert table[column].ge(0).all() and np.isfinite(table[column]).all()
        assert table.spatial_version.eq('H1_unified_20260915_v3').all()
        assert table.input_status.eq('research_candidate_not_certified_outfall').all()
    year_error=float((monthly.groupby(ak).tn_load_kg_n_month.sum()-annual.set_index(ak).tn_load_kg_n_yr).abs().max())
    assert year_error<=1e-6
    old_annual=pd.read_parquet(legacy/'data/prb_wwtp_tn_annual_reach_2006_2019.parquet')
    old_monthly=pd.read_parquet(legacy/'data/prb_wwtp_tn_monthly_reach_2006_2019.parquet')
    source_keys=['year','tn_scenario']
    prior=old_annual.groupby(source_keys).tn_load_kg_n_yr.sum()
    remaining=annual.groupby(source_keys).tn_load_kg_n_yr.sum()
    excluded=outside.groupby(source_keys).tn_load_kg_n_yr.sum()
    conservation_error=float((prior-remaining-excluded).abs().max())
    assert conservation_error<=1e-6
    assert len(reconciliation)==42 and reconciliation.closure_error_kg_n.abs().max()<=1e-6
    registered=reconciliation.set_index(source_keys)
    assert (registered.historical_accepted_kg_n_yr-prior).abs().max()<=1e-6
    assert (registered.current_supported_kg_n_yr-remaining).abs().max()<=1e-6
    assert (registered.outside_H1_kg_n_yr-excluded).abs().max()<=1e-6
    old_month=old_monthly.groupby(['year','month','tn_scenario']).tn_load_kg_n_month.sum()
    new_month=monthly.groupby(['year','month','tn_scenario']).tn_load_kg_n_month.sum()
    residual=[]
    for (year,month,scenario),value in old_month.items():
        excluded_month=excluded.loc[(year,scenario)]*calendar.monthrange(int(year),int(month))[1]/(366 if calendar.isleap(int(year)) else 365)
        residual.append(float(value-new_month.loc[(year,month,scenario)]-excluded_month))
    month_conservation_error=float(np.max(np.abs(residual)))
    assert month_conservation_error<=1e-6
    result=dict(passed=True,network_reaches=230,load_units=113,facility_ids=267,
        retained_historical_candidates=213,outside_H1_historical_candidates=17,
        blocked_load_units=int(bad.sum()),unknown_reach_years=460,independent_2020_2021_load_available=False,
        current_training_modified=False,remapped_annual_rows=len(annual),remapped_monthly_rows=len(monthly),
        independent_annual_rebuild_error_kg=annual_error,independent_monthly_rebuild_error_kg=monthly_error,
        monthly_to_annual_error_kg=year_error,annual_remapping_conservation_error_kg=conservation_error,
        monthly_remapping_conservation_error_kg=month_conservation_error,
        NSE='not_applicable_to_input_mapping')
    print(json.dumps(result,ensure_ascii=False))
    return result

if __name__=='__main__':
    main()
